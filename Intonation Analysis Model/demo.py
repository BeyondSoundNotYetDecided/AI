# pip install torchcrepe whisperx scipy numpy librosa
# conda install -c conda-forge ffmpeg -y

import whisperx
import torchcrepe
import librosa
import torch
import numpy as np

def analyze_intonation(audio_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 오디오 로드 (16kHz로 리샘플링)
    audio, sr = librosa.load(audio_path, sr=16000)

    # ==========================================
    # A. WhisperX로 단어 타이밍(Alignment) 구하기
    # ==========================================
    print("Alignment 수행 중...")
    # 모델 로드 (영어 모델, 크기는 small이나 medium 추천)
    model = whisperx.load_model("small", device, compute_type="float32") 
    result = model.transcribe(audio_path)
    
    # 정렬(Alignment) 모델 로드 및 수행
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # ==========================================
    # B. CREPE로 피치(Pitch) 추출하기
    # ==========================================
    print("Pitch 추출 중...")
    # 오디오를 텐서로 변환
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)
    
    # 10ms (0.01초) 간격으로 피치 추출
    hop_length = int(sr / 100) 
    fmin, fmax = 50, 2000 # 사람 목소리 범위
    model_type = 'tiny' # 빠르고 가벼움 (정확도 필요하면 'full')

    # CREPE 실행 (확률이 낮은 구간은 묵음 처리하기 위해 confidence도 받음)
    pitch, confidence = torchcrepe.predict(
        audio_tensor, 
        sr, 
        hop_length, 
        fmin, 
        fmax, 
        model=model_type, 
        batch_size=2048,
        device=device,
        return_periodicity=True
    )
    
    pitch = pitch.squeeze().cpu().numpy()
    confidence = confidence.squeeze().cpu().numpy()

    # ==========================================
    # C. 데이터 합치기 (단어별 평균 피치 or 그래프용 데이터)
    # ==========================================
    word_segments = []
    
    # WhisperX 결과에서 단어 단위로 루프
    for segment in aligned_result["segments"]:
        for word_info in segment["words"]:
            if "start" not in word_info: continue # 타이밍 못 찾은 단어 스킵
            
            start_t = word_info["start"]
            end_t = word_info["end"]
            word = word_info["word"]

            # 해당 시간대의 Pitch 데이터만 잘라내기
            start_idx = int(start_t * 100) # 10ms 단위이므로 * 100
            end_idx = int(end_t * 100)
            
            # 인덱스 범위 체크
            if start_idx >= len(pitch): continue
            segment_pitch = pitch[start_idx:end_idx]
            segment_conf = confidence[start_idx:end_idx]

            # 신뢰도가 낮은(잡음/묵음) 피치는 제외하고 평균 계산
            valid_indices = segment_conf > 0.4
            if np.any(valid_indices):
                avg_pitch = np.mean(segment_pitch[valid_indices])
            else:
                avg_pitch = 0 # 무성음(ex. s, t, k) 구간일 수 있음

            word_segments.append({
                "word": word,
                "start": start_t,
                "end": end_t,
                "avg_pitch": float(avg_pitch),
                "pitch_curve": segment_pitch.tolist() # 그래프 그릴 때 쓸 전체 데이터
            })

    return word_segments

# --- 실행 예시 ---
# data = analyze_intonation("my_voice.wav")
# for item in data:
#     print(f"단어: {item['word']:10} | 구간: {item['start']:.2f}~{item['end']:.2f} | 평균음높이: {item['avg_pitch']:.1f}Hz")