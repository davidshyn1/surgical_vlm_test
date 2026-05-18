# surgical_vlm_test

CholecT50 **triplet recognition** · Cholec80 **phase recognition** 벤치를 위한 독립 패키지입니다.  
`surgical_vlm_grounding`과 분리되어 있으며, bbox/localization/visualization은 사용하지 않습니다.

백엔드: `prismatic` · `cosmos` · `groot` (`backends.py`)

---

## 1. 구성

| 파일 | 역할 |
|------|------|
| `triplet_recognition_cholect50.py` | CholecT50 triplet 평가 |
| `phase_recognition_cholec80.py` | Cholec80 phase 평가 |
| `cholect50_data.py` | challenge-val 라벨·프레임 로딩 |
| `cholec80_data.py` | Cholec80 phase 라벨·비디오 프레임 로딩 |
| `utils.py` | 라벨 파싱, resume, 공통 상수 |
| `backends.py` | VLM 로드·추론 |
| `grounding_task.sh` | 실행 런처 (`uv` + 백엔드 venv) |
| `setup_backend.sh` | 백엔드 `.venv` 설치 (uv) |

---

## 2. 태스크 개요

**Triplet recognition** — 한 프레임에서 (instrument, verb, target)을 인식합니다.

- **단일 프롬프트**: 벤치 Figure 스타일  
  *"What tasks are the instruments accomplishing with the targets in this surgical image?"*  
  (프레임에 triplet이 여러 개 있을 수 있음)
- **모델 로드 1회** → **프레임당 VLM 1회** → 해당 프레임의 GT annotation마다 채점
- instrument별로 셸에서 스크립트를 반복 실행할 필요 없음

### 프롬프트 모드 (`--prompt-mode`)

| 모드 | 설명 |
|------|------|
| `mcq` (기본) | Instrument / Action / Target 각각 **A, B, C, …** 옵션 목록 제공 |
| `ov` | 옵션 없음 (open vocabulary) |

### 평가 지표 (`metrics` in JSON)

- **Component Accuracy**: Instrument, Verb, Target 각각
- **Triplet Accuracy**: 세 컴포넌트가 모두 맞은 비율
- **mAP**: Instrument / Verb / Target 각 component별 mean AP

예측이 여러 triplet을 나열하면, GT triplet이 그 목록 **안에 포함**되면 정답으로 처리합니다.

### Cholec80 Phase Recognition

**Phase recognition** — 담낭절제 영상 프레임을 7개 수술 단계 중 하나로 분류합니다.

- **프롬프트**: *"In the Cholecystectomy surgical image, what is the current Phase? The available phase options are …"* (A–G)
- **데이터**: `../data/Cholec80` (없으면 `../data/cholec80` 자동 탐색)
- **기본 split**: `eval` = **video41–video80** (EndoNet evaluation set)
- **지표**: Accuracy, 클래스별 Recall / Precision / Jaccard, macro 평균

비디오 MP4에서 프레임을 읽습니다. **OpenCV는 필수가 아닙니다** (numpy pin 환경 권장 경로 아래 참고).  
Cholec80은 **전 비디오 25 fps**이며, phase annotation은 **매 프레임**입니다.

| `--frame-stride` | 샘플링 | 용도 |
|----------------|--------|------|
| `25` (기본) | 약 **1 fps** (0, 25, 50, … 번 프레임) | 권장 eval |
| `1` | **25 fps** (모든 annotated frame) | 전체 프레임 eval (VLM 호출 매우 많음) |

7개 phase: Preparation, Calot Triangle Dissection, Clipping and Cutting, Gallbladder Dissection, Gallbladder Packaging, Cleaning and Coagulation, Gallbladder Retraction.

실행 예시는 **§4.1 Cholec80 phase**를 참고하세요.

---

## 3. 사전 준비

### 3.1 데이터

**CholecT50 (triplet)**

- 라벨 (기본): `<surgical repo>/eval/cholect50-challenge-val/labels`
- 프레임: `CHOLECT50_VIDEOS_ROOT`로 지정 (예: `.../CholecT50/videos` 또는 challenge-val 내 `videos`)

**Cholec80 (phase)**

- 루트: `<surgical repo>/data/cholec80` (`CHOLEC80_ROOT` 또는 `--dataset-root`)
- `videos/videoNN.mp4`, `phase_annotations/videoNN-phase.txt`
- 평가 기본: video **41–80**

### 3.2 HF 토큰

```bash
cp /path/to/.hf_token surgical_vlm_test/.hf_token
```

### 3.3 백엔드 가상환경

```bash
cd surgical_vlm_test
bash setup_backend.sh              # all
bash setup_backend.sh prismatic    # 하나만
```

백엔드 repo 기본 경로: `surgical/../backend` (`VLA_ROOT_OVERRIDE`로 변경 가능)

- `prismatic-vlms/.venv`
- `cosmos-reason2/.venv`
- `GR00T-H/.venv`

**Cholec80 프레임 로딩 (numpy pin 시 권장 순서)**

1. **`ffmpeg` on PATH** (기본, `auto`) — pip 추가 설치 없음, numpy와 무관  
2. **`--frames-root`** — 미리 뽑아 둔 PNG/JPG (`video41/000025.png`), PIL만 사용  
3. **OpenCV** (`--frame-reader opencv`) — venv에 이미 `cv2`가 있을 때만.  
   `opencv-python-headless`를 새로 깔면 **numpy 버전 충돌**이 날 수 있음 → 피하는 것을 권장

```bash
# (선택) eval 41–80, stride 25 프레임만 디스크에 추출 — 한 번만 실행
CHOLEC80_ROOT=../data/cholec80 OUT_ROOT=/path/to/cholec80_frames_stride25 STRIDE=25 \
  bash scripts/extract_cholec80_frames.sh

# 추출본으로 eval (가장 안전)
CHOLEC80_FRAMES_ROOT=/path/to/cholec80_frames_stride25 \
  bash grounding_task.sh phase_recognition_cholec80

# 또는 ffmpeg로 MP4에서 바로 (추출 없이)
bash grounding_task.sh phase_recognition_cholec80   # 시스템 ffmpeg 필요
```

### 3.4 기본 모델 ID

| Backend | 기본 `--model-id` |
|---------|-------------------|
| prismatic | `prism-dinosiglip+7b` |
| cosmos | `nvidia/Cosmos-Reason2-2B` |
| groot | `nvidia/GR00T-H` |

---

## 4. 실행

### 4.1 권장: `grounding_task.sh`

#### CholecT50 triplet

**전체 평가** (기본, MCQ):

```bash
export CHOLECT50_VIDEOS_ROOT=/path/to/CholecT50/videos   # 또는 challenge-val/videos

BACKEND=prismatic DEVICE_VISIBLE=0 \
  bash grounding_task.sh triplet_recognition_cholect50 --prompt-mode mcq
```

**Open vocabulary + 일부만 샘플링**:

```bash
bash grounding_task.sh triplet_recognition_cholect50 \
  --prompt-mode ov --samples-only --video VID68 --samples-per-instrument 10
```

**Cosmos 예**:

```bash
BACKEND=cosmos MODEL_ID=nvidia/Cosmos-Reason2-2B DEVICE_VISIBLE=0 \
  bash grounding_task.sh triplet_recognition_cholect50 --prompt-mode mcq
```

#### Cholec80 phase

`grounding_task.sh`가 기본으로 `--dataset-root $CHOLEC80_ROOT`, `--split eval`(video41–80)을 넣습니다.

**스모크 테스트** (video 41, sparse frames):

```bash
BACKEND=prismatic DEVICE_VISIBLE=0 \
  bash grounding_task.sh phase_recognition_cholec80 \
    --video 41 --frame-stride 250 --max-frames-per-video 5
```

**Eval 41–80, 1 fps** (기본 `frame-stride 25`):

```bash
BACKEND=prismatic DEVICE_VISIBLE=0 \
  bash grounding_task.sh phase_recognition_cholec80
# 동일: --frame-stride 25 (명시 생략 가능)
```

**전체 annotated frame (25 fps)**:

```bash
bash grounding_task.sh phase_recognition_cholec80 --frame-stride 1
```

**Train split (video01–40)**:

```bash
bash grounding_task.sh phase_recognition_cholec80 --split train
```

### 4.2 Python 직접 실행

**CholecT50**:

```bash
uv run --python ../backend/prismatic-vlms/.venv/bin/python \
  triplet_recognition_cholect50.py \
  --backend prismatic \
  --dataset-root ../eval/cholect50-challenge-val \
  --videos-root /path/to/CholecT50/videos \
  --prompt-mode mcq
```

**Cholec80**:

```bash
uv run --python ../backend/prismatic-vlms/.venv/bin/python \
  phase_recognition_cholec80.py \
  --backend prismatic \
  --dataset-root ../data/cholec80 \
  --split eval \
  --frame-stride 25
```

### 4.3 주요 CLI 인자

**CholecT50 (`triplet_recognition_cholect50.py`)**

| 인자 | 설명 |
|------|------|
| `--prompt-mode {mcq,ov}` | MCQ 옵션 vs open vocab (기본: `mcq`) |
| `--eval-all` | 모든 triplet annotation 평가 (기본) |
| `--samples-only` | `--samples-per-instrument`만큼 instrument별 샘플링 |
| `--video VID68` | 비디오 필터 |
| `--instrument grasper` | instrument 필터 |
| `--dataset-root` | challenge-val 루트 (기본: `../eval/cholect50-challenge-val`) |
| `--videos-root` | 프레임 이미지 루트 |
| `--force` | 기존 결과 무시하고 재추론 |
| `--output` | 결과 JSON 경로 |

**Cholec80 (`phase_recognition_cholec80.py`)**

| 인자 | 설명 |
|------|------|
| `--split {eval,train,all}` | `eval`=video41–80 (기본), `train`=01–40 |
| `--frame-stride N` | N프레임마다 1장 샘플 (기본 `25` ≈ 1 fps) |
| `--frames-root` | 추출 프레임 루트 (`video41/000123.png`) |
| `--frame-reader {auto,ffmpeg,opencv}` | MP4 디코드 방식 (기본 `auto` = ffmpeg 우선) |
| `--max-frames-per-video K` | 비디오당 최대 K장 (디버그용) |
| `--video 41` | 단일 비디오 (`41`, `video41` 모두 가능) |
| `--dataset-root` | Cholec80 루트 (기본: `../data/Cholec80`, `cholec80` 폴백) |
| `--force` / `--output` | triplet과 동일 |

**공통**: `--backend`, `--model-id`, `--device`, `--hf-token`, `--max-new-tokens`

### 4.4 기본 출력 경로

| 태스크 | 기본 JSON 경로 |
|--------|----------------|
| Triplet | `outputs/triplet_recognition_cholect50/triplet_{backend}_{model}_{mcq\|ov}/cholect50_challenge_val_triplet.json` |
| Phase | `outputs/phase_recognition_cholec80/phase_{backend}_{model}_{split}/cholec80_phase_stride{N}.json` |

JSON `metrics` 예 (phase): `accuracy`, `macro_recall`, `macro_precision`, `macro_jaccard`, `per_class` (클래스별 recall/precision/jaccard).

---

## 5. 환경 변수 (`grounding_task.sh`)

| 변수 | 설명 |
|------|------|
| `BACKEND` | `prismatic` \| `cosmos` \| `groot` |
| `DEVICE_VISIBLE` | `CUDA_VISIBLE_DEVICES` (기본 `0`) |
| `MODEL_ID` | `--model-id` 자동 주입 |
| `CHOLECT50_CHALLENGE_VAL_ROOT` | triplet `--dataset-root` (기본: `../eval/cholect50-challenge-val`) |
| `CHOLECT50_VIDEOS_ROOT` | triplet `--videos-root` |
| `CHOLEC80_ROOT` | phase `--dataset-root` (기본: `../data/Cholec80`, 없으면 `../data/cholec80`) |
| `CHOLEC80_FRAMES_ROOT` | phase `--frames-root` (추출 PNG 루트, 선택) |
| `PRISMATIC_PYTHON` / `COSMOS_PYTHON` / `GROOT_PYTHON` | venv python 경로 override |
| `GROUNDING_TASK_AUTO_BACKEND_SETUP` | `0`이면 uv 자동 설치 스킵 |

---

## 6. `surgical_vlm_grounding`과의 차이

| | **surgical_vlm_test** | **surgical_vlm_grounding** |
|---|------------------------|------------------------------|
| 목적 | CholecT50 triplet · Cholec80 phase VLM eval | Localization, language/visual grounding 등 |
| CholecT50 | `triplet_recognition_cholect50.py` | `localization_cholect50.py`, `language_grounding_v*`, … |
| Cholec80 | `phase_recognition_cholec80.py` | (별도 phase 스크립트 없음) |
| Bbox | 사용 안 함 | localization 등에서 사용 |
| 입력 | T50: 추출 프레임 이미지 / C80: MP4에서 프레임 읽기 | 주로 추출 프레임 |
| 프롬프트 | 태스크별 단일 질문 (triplet 또는 phase MCQ) | multi-step localization + action/phase |
| 추론 | 프레임당 1회 VLM | task별 상이 |

---

## 7. 문제 해결

1. **CholecT50 프레임 이미지 없음** — `CHOLECT50_VIDEOS_ROOT` 또는 `--videos-root` 확인  
2. **Cholec80 데이터 없음** — `CHOLEC80_ROOT` 또는 `--dataset-root`에 `phase_annotations/`, `videos/` 있는지 확인 (`Cholec80` vs `cholec80` 대소문자 폴백 지원)  
3. **프레임 로드 실패** — `ffmpeg -version` 확인, 또는 `scripts/extract_cholec80_frames.sh` 후 `CHOLEC80_FRAMES_ROOT` 사용 (`opencv` 설치보다 권장)  
4. **백엔드 import 실패** — `bash setup_backend.sh <backend>`  
5. **HF 401** — `.hf_token` 경로 및 권한  
6. **resume** — 동일 `--output` JSON에 이어서 실행; 전체 재실행은 `--force`  
7. **phase eval이 너무 느림** — 기본 `--frame-stride 25`(1 fps) 사용; `--frame-stride 1`은 전 프레임(약 98k+ calls / eval 40 videos)

```bash
python triplet_recognition_cholect50.py --help
python phase_recognition_cholec80.py --help
bash grounding_task.sh help
```
