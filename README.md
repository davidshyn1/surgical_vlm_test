# surgical_vlm_test

CholecT50 **triplet recognition** 벤치를 위한 독립 패키지입니다.  
`surgical_vlm_grounding`과 분리되어 있으며, bbox/localization/visualization은 사용하지 않습니다.

백엔드: `prismatic` · `cosmos` · `groot` (`backends.py`)

---

## 1. 구성

| 파일 | 역할 |
|------|------|
| `triplet_recognition_cholect50.py` | 메인 평가 스크립트 |
| `cholect50_data.py` | challenge-val 라벨·프레임 로딩, 샘플링 |
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

---

## 3. 사전 준비

### 3.1 데이터

- 라벨 (기본): `<surgical repo>/eval/cholect50-challenge-val/labels`
- 프레임: `CHOLECT50_VIDEOS_ROOT`로 지정 (예: `.../CholecT50/videos` 또는 challenge-val 내 `videos`)

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

### 3.4 기본 모델 ID

| Backend | 기본 `--model-id` |
|---------|-------------------|
| prismatic | `prism-dinosiglip+7b` |
| cosmos | `nvidia/Cosmos-Reason2-2B` |
| groot | `nvidia/GR00T-H` |

---

## 4. 실행

### 4.1 권장: `grounding_task.sh`

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

### 4.2 Python 직접 실행

```bash
uv run --python ../backend/prismatic-vlms/.venv/bin/python \
  triplet_recognition_cholect50.py \
  --backend prismatic \
  --dataset-root ../eval/cholect50-challenge-val \
  --videos-root /path/to/CholecT50/videos \
  --prompt-mode mcq
```

### 4.3 주요 CLI 인자

| 인자 | 설명 |
|------|------|
| `--prompt-mode {mcq,ov}` | MCQ 옵션 vs open vocab (기본: `mcq`) |
| `--eval-all` | 모든 triplet annotation 평가 (기본) |
| `--samples-only` | `--samples-per-instrument`만큼 instrument별 샘플링 |
| `--video VID68` | 비디오 필터 |
| `--instrument grasper` | instrument 필터 (셸 루프 대신 필터만 사용) |
| `--force` | 기존 결과 무시하고 재추론 |
| `--output` | 결과 JSON 경로 |

기본 출력:

`surgical_vlm_test/outputs/triplet_recognition_cholect50/triplet_{backend}_{model_name}_{mcq|ov}/cholect50_challenge_val_triplet.json`

---

## 5. 환경 변수 (`grounding_task.sh`)

| 변수 | 설명 |
|------|------|
| `BACKEND` | `prismatic` \| `cosmos` \| `groot` |
| `DEVICE_VISIBLE` | `CUDA_VISIBLE_DEVICES` (기본 `0`) |
| `MODEL_ID` | `--model-id` 자동 주입 |
| `CHOLECT50_CHALLENGE_VAL_ROOT` | `--dataset-root` (기본: `../eval/cholect50-challenge-val`) |
| `CHOLECT50_VIDEOS_ROOT` | `--videos-root` |
| `PRISMATIC_PYTHON` / `COSMOS_PYTHON` / `GROOT_PYTHON` | venv python 경로 override |
| `GROUNDING_TASK_AUTO_BACKEND_SETUP` | `0`이면 uv 자동 설치 스킵 |

---

## 6. `surgical_vlm_grounding`과의 차이

| | **surgical_vlm_test** | **surgical_vlm_grounding** |
|---|------------------------|------------------------------|
| 목적 | Triplet recognition (MCQ/OV) | Localization, language/visual grounding 등 |
| CholecT50 스크립트 | `triplet_recognition_cholect50.py` | `localization_cholect50.py`, `language_grounding_v*`, … |
| Bbox | 사용 안 함 | localization 등에서 사용 |
| 프롬프트 | 단일 triplet 질문 | multi-step localization + action/phase |
| 추론 | 프레임당 1회 VLM | task별 상이 |

---

## 7. 문제 해결

1. **프레임 이미지 없음** — `CHOLECT50_VIDEOS_ROOT` 또는 `--videos-root` 확인  
2. **백엔드 import 실패** — `bash setup_backend.sh <backend>`  
3. **HF 401** — `.hf_token` 경로 및 권한  
4. **resume** — 동일 `--output` JSON에 이어서 실행; 전체 재실행은 `--force`

```bash
python triplet_recognition_cholect50.py --help
bash grounding_task.sh help
```
