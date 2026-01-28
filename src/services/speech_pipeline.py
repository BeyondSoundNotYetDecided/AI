from __future__ import annotations

from typing import Any, Dict, List, Literal, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from src.models.stt_whisper import (
    get_whisperx_models,
    extract_word_timings,
)

from src.models.pitch_crepe import extract_pitch_crepe
from src.models.align_merge import merge_words_with_pitch_curve
from src.models.g2p import text_to_phonemes
from src.models.pronunciation import phonemes_to_hangul_ipa

Mode = Literal["pron", "inton", "all"]


def _process_pronunciation(words: List[str]) -> Dict[str, Any]:
    """발음 분석 실행"""
    pron: Dict[str, Any] = {}

    for idx, w in enumerate(words):
        # 1) 단어별 phonemes (ARPAbet) -> upl
        upl = text_to_phonemes(w)

        # 2) 단어별 한글/IPA
        ukor, _ipa_str, uipa = phonemes_to_hangul_ipa(upl)

        # 동일 단어 반복 대비: key를 유니크하게
        key = w  # 중복 가능성 있으면: f"{w}#{idx}"로 변경

        pron[key] = {
            "upl": upl,     # ARPAbet list
            "uipa": uipa,   # IPA list
            "ukor": ukor,   # 한글 발음 문자열
        }

    return pron


def _process_intonation(
    audio_path: str, 
    word_segments: List[Dict[str, Any]], 
    device: str
) -> List[Dict[str, Any]]:
    """인토네이션 분석 실행"""
    pitch_result = extract_pitch_crepe(audio_path, device=device)
    return merge_words_with_pitch_curve(word_segments, pitch_result)


def analyze_speech_stream(
    audio_path: str,
    # whisper_model_name: str = "small.en",
    # whisper_vad_method: str = "silero",
    loaded_models: tuple,   # 이미 로딩된 모델을 받음
    mode: Mode = "all",
) -> Generator[str, None, None]:
    """
    [기능] 음성 파이프라인 메인 함수
    1. WhisperX (공통)
    2. Pronunciation (G2P)
    3. Intonation (CREPE)
    """
    
    # 1. WhisperX 공통 단계: 모델 로드 -> 실행 -> 메모리 정리
    # model, model_a, metadata, device = get_whisperx_models(
    #     model_name=whisper_model_name,
    #     vad_method=whisper_vad_method,
    # )
    model, model_a, metadata, device = loaded_models
    
    try:
        word_segments = extract_word_timings(
            audio_path=audio_path,
            model=model,
            model_a=model_a,
            metadata=metadata,
            device=device,
            batch_size=16,
        )
    except Exception as e:
        yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        return

    words = [w["word"] for w in word_segments]

    # 실행할 작업 플래그 설정
    do_pron = mode in ("pron", "all")
    do_into = mode in ("inton", "all")

    # 2. 분석 실행
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_map = {}

        if do_pron:
            # 발음 분석 작업 제출
            f_pron = executor.submit(_process_pronunciation, words)
            future_map[f_pron] = "pron"
        
        if do_into:
            # 인토네이션 분석 작업 제출
            f_into = executor.submit(_process_intonation, audio_path, word_segments, device)
            future_map[f_into] = "inton"

        # 먼저 끝나는 작업부터 yield (as_completed)
        for future in as_completed(future_map):
            task_type = future_map[future]
            try:
                result_data = future.result()
                
                # 결과 전송 (type으로 구분)
                yield json.dumps({
                    "type": task_type,
                    "data": result_data
                }, ensure_ascii=False) + "\n"
                
            except Exception as e:
                yield json.dumps({
                    "type": "error",
                    "task": task_type,
                    "message": str(e)
                }, ensure_ascii=False) + "\n"

# 로컬 실행 테스트용
if __name__ == "__main__":
    import json
    import os
    # 모델 로더 함수 import 필요
    from src.models.stt_whisper import get_whisperx_models

    # 테스트 파일 경로 확인
    test_file = "./experiments/wav_data/i_like_to_dance.wav"
    
    if os.path.exists(test_file):
        print("⏳ [Test] 모델 로딩 중... (처음엔 시간이 좀 걸립니다)")
        
        # 1. 테스트를 위해 여기서 모델을 직접 로드합니다. (Main.py의 lifespan 역할)
        # 실제 서버에서는 이미 로드된 걸 쓰지만, 로컬 테스트에선 직접 준비해야 합니다.
        loaded_models_tuple = get_whisperx_models(
            model_name="small.en", 
            vad_method="silero"
        )
        print("✅ [Test] 모델 로딩 완료!")

        print("🎤 [Test] 분석 시작...")
        
        # 2. 로드된 모델을 인자로 넘겨줍니다.
        generator = analyze_speech_stream(
            audio_path=test_file, 
            mode="all", 
            loaded_models=loaded_models_tuple # <--- 핵심: 모델 전달
        )
        
        # 3. 결과 스트리밍 출력
        for chunk in generator:
            print(chunk.strip())
            
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {test_file}")