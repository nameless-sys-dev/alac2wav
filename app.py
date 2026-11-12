import base64
import json
import logging
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import modal


MAX_UPLOAD_BYTES = 1 << 30  # 1 GiB; adjust for your deployment.
PROBE_TIMEOUT_SECONDS = 15
DECODE_TIMEOUT_SECONDS = 110
COPY_CHUNK_BYTES = 4 << 20

# Explicitly use Ubuntu, not modal.Image.debian_slim().
image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("ffmpeg")  # Includes ffprobe.
    .pip_install(
        "fastapi>=0.115,<1",
        "python-multipart>=0.0.20,<1",
        "mutagen==1.47.0",
    )
)

app = modal.App("alac-to-wav")


@app.function(
    image=image,
    cpu=2,
    memory=2048,
    timeout=180,
    min_containers=0,       # Set to 1 to avoid cold starts; incurs idle cost.
    max_containers=20,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=1)  # Scale horizontally; avoid CPU contention.
@modal.asgi_app()
def api():
    # Imported here so only the Modal package is required locally.
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse
    from mutagen import MutagenError
    from mutagen.mp4 import MP4, MP4FreeForm
    from pydantic import BaseModel
    from starlette.background import BackgroundTask

    logger = logging.getLogger("alac-to-wav")

    web = FastAPI(
        title="ALAC to WAV",
        version="1.0.0",
        description=(
            "Convert one ALAC M4A into lossless 16-bit or 24-bit PCM WAV. "
            "Return a ZIP with optional MP4 tag and artwork sidecars."
        ),
    )

    class ErrorBody(BaseModel):
        # FastAPI validation errors use a list; application errors use a string.
        detail: str | list[dict[str, Any]]

    def integer(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def run_command(argv, timeout_seconds, error_status, error_message):
        # No shell: filenames and metadata cannot become shell commands.
        try:
            result = subprocess.run(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            # subprocess.run kills and waits for the child on timeout.
            raise HTTPException(
                status_code=504,
                detail="Media processing timed out; no partial result is returned.",
            ) from exc

        if result.returncode != 0:
            logger.warning(
                "%s\n%s",
                error_message,
                result.stderr.decode("utf-8", errors="replace")[-2000:],
            )
            raise HTTPException(
                status_code=error_status,
                detail=error_message,
            )

        return result.stdout

    def save_upload(upload, destination):
        size = 0
        try:
            upload.file.seek(0)
            with destination.open("wb") as output:
                while chunk := upload.file.read(COPY_CHUNK_BYTES):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File exceeds {MAX_UPLOAD_BYTES} bytes.",
                        )
                    output.write(chunk)

            if size == 0:
                raise HTTPException(status_code=415, detail="The upload is empty.")
        finally:
            # Release FastAPI's spooled upload before decoding.
            upload.file.close()

    def json_tag_value(value):
        """Export tag values without discarding binary/freeform payloads."""
        if isinstance(value, MP4FreeForm):
            result = {
                "type": "mp4_freeform",
                "dataformat": int(value.dataformat),
                "version": int(value.version),
                "base64": base64.b64encode(bytes(value)).decode("ascii"),
            }

            # Keep the original bytes above; add a readable view for UTF-8 tags.
            if int(value.dataformat) == 1:
                try:
                    result["text"] = bytes(value).decode("utf-8")
                except UnicodeDecodeError:
                    pass

            return result

        if isinstance(value, (bytes, bytearray)):
            return {
                "type": "binary",
                "base64": base64.b64encode(bytes(value)).decode("ascii"),
            }

        if isinstance(value, (list, tuple)):
            return [json_tag_value(item) for item in value]

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        raise TypeError(f"Unsupported tag value type: {type(value).__name__}")

    def image_extension(data):
        # Inspect signatures without decoding or re-encoding the image.
        if data.startswith(b"\xff\xd8"):
            return ".jpg"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if data.startswith((b"GIF87a", b"GIF89a")):
            return ".gif"
        if data.startswith(b"BM"):
            return ".bmp"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ".webp"
        return ".bin"  # Preserve unknown artwork bytes unchanged.

    def extract_sidecars(source, directory):
        try:
            media = MP4(str(source))
            tags = media.tags or {}
            files = []

            # covr contains artwork, which is exported separately.
            exported_tags = {
                key: json_tag_value(value)
                for key, value in tags.items()
                if key != "covr"
            }

            if exported_tags:
                tags_path = directory / "tags.json"
                tags_path.write_text(
                    json.dumps(
                        {
                            "tag_format": "MP4/iTunes",
                            "tags": exported_tags,
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                        indent=2,
                    ) + "\n",
                    encoding="utf-8",
                )
                files.append(tags_path)

            # Export every embedded cover, not just the first.
            for number, cover in enumerate(tags.get("covr", []), start=1):
                data = bytes(cover)
                path = directory / (
                    f"cover-{number:02d}{image_extension(data)}"
                )
                path.write_bytes(data)
                files.append(path)

            return files

        except (MutagenError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail="Cannot safely extract the MP4 tags or artwork.",
            ) from exc

    success_headers = {
        "X-Source-Bit-Depth": {
            "schema": {"type": "integer", "enum": [16, 20, 24]},
        },
        "X-WAV-Bit-Depth": {
            "schema": {"type": "integer", "enum": [16, 24]},
        },
        "X-Audio-Sample-Rate": {
            "schema": {"type": "integer", "minimum": 1},
        },
        "X-Audio-Channels": {
            "schema": {"type": "integer", "minimum": 1},
        },
    }

    error_descriptions = {
        413: "Uploaded file exceeds the configured size limit.",
        415: "Input is not a readable MP4/M4A containing ALAC audio.",
        422: (
            "Invalid request, unsupported depth, multiple audio streams, "
            "invalid metadata, or an ALAC decoding error."
        ),
        504: "Media inspection or decoding timed out.",
    }

    @web.post(
        "/convert",
        operation_id="convertAlac",
        summary="Convert ALAC M4A to WAV and extract tags/artwork",
        response_class=FileResponse,
        response_model=None,
        responses={
            200: {
                "description": (
                    "ZIP containing audio.wav, optional tags.json, "
                    "and optional cover-NN image files."
                ),
                "headers": success_headers,
                "content": {
                    "application/zip": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            },
            **{
                status: {
                    "description": description,
                    "model": ErrorBody,
                }
                for status, description in error_descriptions.items()
            },
        },
    )
    def convert(file: UploadFile = File(...)):
        # Do not use a 'with TemporaryDirectory()' block here: the directory
        # must survive until FileResponse finishes sending the archive.
        temporary = tempfile.TemporaryDirectory(
            prefix="alac-",
            ignore_cleanup_errors=True,
        )
        directory = Path(temporary.name)

        try:
            source = directory / "input.m4a"
            save_upload(file, source)

            probe = json.loads(
                run_command(
                    [
                        "ffprobe",
                        "-v", "error",
                        "-protocol_whitelist", "file",
                        "-f", "mov",
                        "-ignore_editlist", "1",
                        "-select_streams", "a",
                        "-show_entries",
                        (
                            "stream=index,codec_name,sample_rate,channels,"
                            "bits_per_raw_sample"
                        ),
                        "-of", "json",
                        str(source),
                    ],
                    PROBE_TIMEOUT_SECONDS,
                    415,
                    "Input is not a readable MP4/M4A file.",
                )
            )

            streams = probe.get("streams", [])
            if not streams:
                raise HTTPException(
                    status_code=415,
                    detail="No audio stream was found.",
                )
            if len(streams) != 1:
                raise HTTPException(
                    status_code=422,
                    detail="Exactly one audio stream is required.",
                )

            stream = streams[0]
            if stream.get("codec_name") != "alac":
                raise HTTPException(
                    status_code=415,
                    detail="The audio codec must be ALAC.",
                )

            # Use the ALAC-declared precision, not sample_fmt.
            # In particular, 24-bit ALAC normally decodes as s32p.
            source_bits = integer(stream.get("bits_per_raw_sample"))
            if source_bits not in (16, 20, 24):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"ALAC bit depth is {source_bits or 'unknown'}. "
                        "Lossless 16/24-bit output requires "
                        "16-, 20-, or 24-bit ALAC; refusing to truncate or guess."
                    ),
                )

            sample_rate = integer(stream.get("sample_rate"))
            channels = integer(stream.get("channels"))
            if sample_rate <= 0 or channels <= 0:
                raise HTTPException(
                    status_code=422,
                    detail="The source sample rate or channel count is invalid.",
                )

            wav_bits = 16 if source_bits == 16 else 24
            codec = "pcm_s16le" if wav_bits == 16 else "pcm_s24le"
            sample_format = "s16" if wav_bits == 16 else "s32"

            sidecars = extract_sidecars(source, directory)
            wav_path = directory / "audio.wav"

            run_command(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-nostdin",
                    "-y",
                    "-xerror",
                    "-filter_threads", "1",

                    # Input options.
                    "-protocol_whitelist", "file",
                    "-f", "mov",
                    "-ignore_editlist", "1",
                    "-err_detect", "explode",
                    "-threads:a", "1",
                    "-i", str(source),

                    # Only the verified ALAC stream.
                    "-map", f"0:{stream['index']}",
                    "-map_metadata", "-1",
                    "-map_chapters", "-1",

                    # Integer PCM at the source resolution or a wider one.
                    "-c:a", codec,
                    "-sample_fmt", sample_format,
                    "-threads:a", "1",

                    # At the unchanged rate/layout, this only handles
                    # planar-to-packed PCM conversion. Dither is disabled.
                    "-af", "aresample=dither_method=none",

                    # No -ar or -ac: preserve sample rate and channel count.
                    "-bitexact",
                    "-rf64", "auto",
                    "-f", "wav",
                    str(wav_path),
                ],
                DECODE_TIMEOUT_SECONDS,
                422,
                "ALAC decoding failed; no partial audio is returned.",
            )

            source.unlink()

            archive_path = directory / "converted.zip"
            members = [wav_path, *sidecars]

            # No DEFLATE compression: fast packaging, low CPU overhead.
            # ZIP64 and RF64 handle outputs exceeding ordinary 4 GiB limits.
            with zipfile.ZipFile(
                archive_path,
                mode="w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as archive:
                for member in members:
                    archive.write(member, arcname=member.name)

            # Keep only the archive while the client downloads it.
            for member in members:
                member.unlink()

            return FileResponse(
                path=archive_path,
                media_type="application/zip",
                filename="converted.zip",
                headers={
                    "X-Source-Bit-Depth": str(source_bits),
                    "X-WAV-Bit-Depth": str(wav_bits),
                    "X-Audio-Sample-Rate": str(sample_rate),
                    "X-Audio-Channels": str(channels),
                },
                # The bound method also keeps TemporaryDirectory alive.
                background=BackgroundTask(temporary.cleanup),
            )

        except Exception:
            temporary.cleanup()
            raise

    return web