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
        # "mode": mode,
        "words": words,
        # "word_segments": word_segments,
    }

    # 실행할 작업 플래그 설정
    do_pron = mode in ("pron", "all")
    do_into = mode in ("inton", "all")

    # 2. 분석 실행
    if parallel and do_pron and do_into:
        # 병렬 모드 (G2P는 CPU, CREPE는 GPU를 주로 쓰므로 효율적일 수 있음)
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_pron = executor.submit(_process_pronunciation, words)
            future_into = executor.submit(_process_intonation, audio_path, word_segments, device)
            
            result["pron"] = future_pron.result()
            result["inton"] = future_into.result()
    else:
        # 순차 처리
        if do_pron:
            result["pron"] = _process_pronunciation(words)
        
        if do_into:
            result["inton"] = _process_intonation(audio_path, word_segments, device)

    return result


# 로컬 실행 테스트용
if __name__ == "__main__":
    import json
    import os

    # 테스트 파일 경로 확인
    test_file = "./experiments/wav_data/i_like_to_dance.wav"
    
    if os.path.exists(test_file):
        out = analyze_speech(test_file, mode="all", parallel=False)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(f"파일을 찾을 수 없습니다: {test_file}")