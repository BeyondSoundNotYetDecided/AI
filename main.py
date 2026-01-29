from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from src.services.audio_io import temp_audio_file
from src.services.speech_pipeline import analyze_speech_stream
from src.models.stt_whisper import get_whisperx_models

app = FastAPI(title="Speech Analysis API")

# 전역 변수
loaded_models = None

@app.on_event("startup")
async def startup_event():
    global loaded_models
    print("⏳ 모델 로딩 중...")
    loaded_models = get_whisperx_models(
        model_name="small.en",
        vad_method="silero"
    )
    print("✅ 모델 로딩 완료!")

@app.on_event("shutdown")
async def shutdown_event():
    global loaded_models
    loaded_models = None
    print("🛑 서버 종료")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        return {"error": "파일 이름이 없습니다"}
    
    # 1. 파일 읽기 (Bytes)
    audio_bytes = await file.read()
        
    # 2. 제너레이터 래퍼
    def stream_with_cleanup():
        with temp_audio_file(audio_bytes, suffix=".wav") as audio_path:
            
            # 파일 경로(audio_path)를 파이프라인에 넘김
            for chunk in analyze_speech_stream(
                audio_path=audio_path,
                loaded_models=loaded_models,
                mode="all"
            ):
                yield chunk

    # 3. StreamingResponse 반환
    return StreamingResponse(
        stream_with_cleanup(), 
        media_type="application/x-ndjson"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)