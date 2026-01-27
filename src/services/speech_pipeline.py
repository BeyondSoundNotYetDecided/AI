from __future__ import annotations

from typing import Any, Dict, List, Literal
from concurrent.futures import ThreadPoolExecutor

from src.models.stt_whisper import (
    load_whisperx_models,
    extract_word_timings,
    cleanup_whisperx_models,
)
from src.models.pitch_crepe import extract_pitch_crepe
from src.models.align_merge import merge_words_with_pitch_curve
from src.models.g2p import text_to_phonemes
from src.models.pronunciation import phonemes_to_hangul_ipa

Mode = Literal["pronunciation", "intonation", "all"]


def _process_pronunciation(words: List[str]) -> Dict[str, Any]:
    """발음 분석 실행"""
    text = " ".join(words).strip()
    phonemes = text_to_phonemes(text)
    hangul, ipa_str, ipa_list = phonemes_to_hangul_ipa(phonemes)
    
    return {
        "hangul": hangul,
        "ipa": ipa_str,
        "ipa_list": ipa_list,
        # "phonemes": phonemes, # 필요 시 주석 해제
    }


def _process_intonation(
    audio_path: str, 
    word_segments: List[Dict[str, Any]], 
    device: str
) -> List[Dict[str, Any]]:
    """인토네이션 분석 실행"""
    pitch_result = extract_pitch_crepe(audio_path, device=device)
    return merge_words_with_pitch_curve(word_segments, pitch_result)


def analyze_speech(
    audio_path: str,
    mode: Mode = "all",
    parallel: bool = True,
    whisper_model_name: str = "small.en",
    whisper_vad_method: str = "silero",
) -> Dict[str, Any]:
    """
    [기능] 음성 파이프라인 메인 함수
    1. WhisperX (공통)
    2. Pronunciation (G2P)
    3. Intonation (CREPE)
    """
    
    # 1. WhisperX 공통 단계: 모델 로드 -> 실행 -> 메모리 정리
    model, model_a, metadata, device = load_whisperx_models(
        model_name=whisper_model_name,
        vad_method=whisper_vad_method,
    )
    
    try:
        word_segments = extract_word_timings(
            audio_path=audio_path,
            model=model,
            model_a=model_a,
            metadata=metadata,
            device=device,
            batch_size=16,
        )
    finally:
        cleanup_whisperx_models(model, model_a)

    words = [w["word"] for w in word_segments]
    
    # 기본 결과 구조 생성
    result = {
        "mode": mode,
        "words": words,
        "word_segments": word_segments,
    }

    # 실행할 작업 플래그 설정
    do_pron = mode in ("pronunciation", "all")
    do_into = mode in ("intonation", "all")

    # 2. 분석 실행
    if parallel and do_pron and do_into:
        # 병렬 모드 (G2P는 CPU, CREPE는 GPU를 주로 쓰므로 효율적일 수 있음)
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_pron = executor.submit(_process_pronunciation, words)
            future_into = executor.submit(_process_intonation, audio_path, word_segments, device)
            
            result["pronunciation"] = future_pron.result()
            result["intonation"] = future_into.result()
    else:
        # 순차 처리
        if do_pron:
            result["pronunciation"] = _process_pronunciation(words)
        
        if do_into:
            result["intonation"] = _process_intonation(audio_path, word_segments, device)

    return result


# 로컬 실행 테스트용
if __name__ == "__main__":
    import json
    import os

    # 테스트 파일 경로 확인
    test_file = "./wav_data/i_like_to_dance.wav"
    
    if os.path.exists(test_file):
        out = analyze_speech(test_file, mode="all", parallel=False)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(f"파일을 찾을 수 없습니다: {test_file}")