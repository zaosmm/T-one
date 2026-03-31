"""Module contain simple website implementation for demo purposes."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse, FileResponse, HTMLResponse

from tone import read_example_audio, read_audio
from tone.demo.read_audio import read_audio_from_bytes
from tone.pipeline import StreamingCTCPipeline
from tone.project import VERSION

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_BYTES_PER_SAMPLE = 2


@dataclass
class Settings:
    """Global website settings.

    Can be modified using environment variables.
    """

    cors_allow_all: bool = False
    load_from_folder: Path | None = field(default_factory=lambda: os.getenv("LOAD_FROM_FOLDER", None))


class SingletonPipeline:
    """Singleton object to store a single ASR pipeline."""

    pipeline: StreamingCTCPipeline | None = None

    def __new__(cls) -> None:
        """Ensure the class is never created."""
        raise RuntimeError("This is class is a singleton!")

    @classmethod
    def init(cls, settings: Settings) -> None:
        """Initialize singleton object using settings."""
        if settings.load_from_folder is None:
            cls.pipeline = StreamingCTCPipeline.from_hugging_face()
        else:
            cls.pipeline = StreamingCTCPipeline.from_local(settings.load_from_folder)

    @classmethod
    def process_chunk(
            cls,
            audio_chunk: StreamingCTCPipeline.InputType,
            state: StreamingCTCPipeline.StateType | None = None,
            *,
            is_last: bool = False,
    ) -> tuple[StreamingCTCPipeline.OutputType, StreamingCTCPipeline.StateType]:
        """Process audio chunk using ASR pipeline.

        See `StreamingCTCPipeline.forward` for more info.
        """
        if cls.pipeline is None:
            raise RuntimeError("Pipeline is not initialized")
        return cls.pipeline.forward(audio_chunk, state, is_last=is_last)


router = APIRouter()


async def get_chunk_stream(ws: WebSocket) -> AsyncIterator[tuple[npt.NDArray[np.int16], bool]]:
    """Get audio chunks from websocket and return them as async iterator."""
    audio_data = bytearray()
    # See description of PADDING in StreamingCTCPipeline
    audio_data.extend(np.zeros((StreamingCTCPipeline.PADDING,), dtype=np.int16).tobytes())
    is_last = False
    while True:
        await ws.send_json({"event": "ready"})
        recv_bytes = await ws.receive_bytes()
        if len(recv_bytes) == 0:  # Last chunk of audio
            is_last = True
            audio_data.extend(np.zeros((StreamingCTCPipeline.PADDING,), dtype=np.int16).tobytes())
            fill_chunk_size = -(len(audio_data) // _BYTES_PER_SAMPLE) % StreamingCTCPipeline.CHUNK_SIZE
            audio_data.extend(np.zeros((fill_chunk_size,), dtype=np.int16).tobytes())
        else:
            audio_data.extend(recv_bytes)

        while len(audio_data) >= StreamingCTCPipeline.CHUNK_SIZE * _BYTES_PER_SAMPLE:
            chunk = np.frombuffer(audio_data[: StreamingCTCPipeline.CHUNK_SIZE * _BYTES_PER_SAMPLE], dtype=np.int16)
            del audio_data[: StreamingCTCPipeline.CHUNK_SIZE * _BYTES_PER_SAMPLE]
            yield chunk, is_last and (len(audio_data) == 0)

        if len(recv_bytes) == 0:
            return


async def get_chunk_stream_from_file(file: UploadFile) -> AsyncIterator[tuple[npt.NDArray[np.int16], bool]]:
    """Get audio chunks from websocket and return them as async iterator."""
    audio_data = bytearray()
    # See description of PADDING in StreamingCTCPipeline
    audio_data.extend(np.zeros((StreamingCTCPipeline.PADDING,), dtype=np.int16).tobytes())
    is_last = False
    while True:
        recv_bytes = await file.read()
        if len(recv_bytes) == 0:  # Last chunk of audio
            is_last = True
            audio_data.extend(np.zeros((StreamingCTCPipeline.PADDING,), dtype=np.int16).tobytes())
            fill_chunk_size = -(len(audio_data) // _BYTES_PER_SAMPLE) % StreamingCTCPipeline.CHUNK_SIZE
            audio_data.extend(np.zeros((fill_chunk_size,), dtype=np.int16).tobytes())
        else:
            audio_data.extend(recv_bytes)

        while len(audio_data) >= StreamingCTCPipeline.CHUNK_SIZE * _BYTES_PER_SAMPLE:
            chunk = np.frombuffer(audio_data[: StreamingCTCPipeline.CHUNK_SIZE * _BYTES_PER_SAMPLE], dtype=np.int16)
            del audio_data[: StreamingCTCPipeline.CHUNK_SIZE * _BYTES_PER_SAMPLE]
            yield chunk, is_last and (len(audio_data) == 0)

        if len(recv_bytes) == 0:
            return


@router.websocket("/ws")
async def websocket_stt(ws: WebSocket) -> None:
    """Websocket endpoint for streaming audio processing."""
    await ws.accept()
    payloads = []
    text = ''
    try:
        state: StreamingCTCPipeline.StateType | None = None
        async for audio_chunk, is_last in get_chunk_stream(ws):
            output, state = SingletonPipeline.process_chunk(audio_chunk.astype(np.int32), state, is_last=is_last)
            for phrase in output:
                payload = {
                    "event": "transcript",
                    "phrase": {"text": phrase.text, "start_time": phrase.start_time, "end_time": phrase.end_time},
                }
                text += f'{phrase.start_time}-{phrase.end_time}: {phrase.text}\n'
                payloads.append(payload)
                await ws.send_json(
                    payload,
                )

    except WebSocketDisconnect:
        pass
    finally:
        result_dir = os.path.join(RESULTS_DIR, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")}')

        os.makedirs(result_dir, exist_ok=True)

        result_file_json = os.path.join(result_dir, 'result.json')
        with open(result_file_json, 'w') as result_obj:
            result_obj.write(json.dumps(payloads, ensure_ascii=False))

        result_file_txt = os.path.join(result_dir, 'result.txt')
        with open(result_file_txt, 'w') as result_obj:
            result_obj.write(text)


@router.post("/asr")
async def http_stt(
        file: UploadFile = File(...),
) -> JSONResponse:
    """HTTP endpoint for non-streaming audio processing."""

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        all_phrases = []

        audio = read_audio_from_bytes(await file.read())
        pipeline = StreamingCTCPipeline.from_hugging_face()
        output = pipeline.forward_offline(audio)
        for phrase in output:
            all_phrases.append({"text": phrase.text, "start_time": phrase.start_time, "end_time": phrase.end_time})

        return JSONResponse(content={
            "success": True,
            "transcript": all_phrases,
            "num_phrases": len(all_phrases)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


def init_dir_results(root='.') -> str:
    results_dir = os.path.join(root, 'results')
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except Exception as e:
            print(e)
    return results_dir


@router.get("/results")
async def list_results() -> JSONResponse:
    """Get list of all result directories with their metadata."""
    results = []
    try:
        if os.path.exists(RESULTS_DIR):
            for item in sorted(os.listdir(RESULTS_DIR), reverse=True):
                item_path = os.path.join(RESULTS_DIR, item)
                if os.path.isdir(item_path):
                    json_path = os.path.join(item_path, 'result.json')
                    txt_path = os.path.join(item_path, 'result.txt')

                    # Get creation time from directory name or file stats
                    created_time = item

                    # Count transcripts in json file
                    transcript_count = 0
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r') as f:
                                data = json.load(f)
                                transcript_count = len(data)
                        except:
                            pass

                    results.append({
                        'name': item,
                        'path': item_path,
                        'created': created_time,
                        'has_json': os.path.exists(json_path),
                        'has_txt': os.path.exists(txt_path),
                        'transcript_count': transcript_count
                    })
    except Exception as e:
        print(f"Error listing results: {e}")

    return JSONResponse(content={'results': results})


@router.get("/results/download/{result_name}/{file_type}")
async def download_result_file(result_name: str, file_type: str):
    """Download specific file from a result directory."""
    if file_type not in ['json', 'txt']:
        raise HTTPException(status_code=400, detail="Invalid file type")

    result_dir = os.path.join(RESULTS_DIR, result_name)
    if not os.path.exists(result_dir):
        raise HTTPException(status_code=404, detail="Result directory not found")

    filename = f'result.{file_type}'
    file_path = os.path.join(result_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {filename} not found")

    return FileResponse(
        path=file_path,
        filename=f'{result_name}.{file_type}',
        media_type='application/octet-stream'
    )


@router.delete("/results/delete/{result_name}")
async def delete_result(result_name: str) -> JSONResponse:
    """Delete a specific result directory."""
    result_dir = os.path.join(RESULTS_DIR, result_name)

    if not os.path.exists(result_dir):
        raise HTTPException(status_code=404, detail="Result directory not found")

    try:
        shutil.rmtree(result_dir)
        return JSONResponse(content={'message': f'Successfully deleted {result_name}'})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting directory: {str(e)}")


@router.delete("/results/delete-all")
async def delete_all_results() -> JSONResponse:
    """Delete all result directories."""
    deleted_count = 0
    errors = []

    try:
        if os.path.exists(RESULTS_DIR):
            for item in os.listdir(RESULTS_DIR):
                item_path = os.path.join(RESULTS_DIR, item)
                if os.path.isdir(item_path):
                    try:
                        shutil.rmtree(item_path)
                        deleted_count += 1
                    except Exception as e:
                        errors.append(f"{item}: {str(e)}")

        if errors:
            return JSONResponse(
                content={'message': f'Deleted {deleted_count} directories with errors', 'errors': errors},
                status_code=207
            )
        else:
            return JSONResponse(content={'message': f'Successfully deleted {deleted_count} directories'})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting directories: {str(e)}")


def get_application() -> FastAPI:
    """Build and return FastAPI application."""
    app = FastAPI(title="T-one Streaming ASR", version=VERSION, docs_url=None, redoc_url=None)
    settings = Settings()
    if settings.cors_allow_all:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_event_handler("startup", lambda: SingletonPipeline.init(settings))

    app.include_router(router, prefix="/api")
    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="Main website page")

    return app


RESULTS_DIR = init_dir_results()
app = get_application()
