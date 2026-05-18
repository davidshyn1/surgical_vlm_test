# surgical_vlm_test

CholecT50 **triplet recognition** · Cholec80 **phase recognition** · EndoVis 2017 **instrument localization** 벤치를 위한 독립 패키지입니다.  
`surgical_vlm_grounding`과 분리되어 있으며, CholecT50/80은 분류·인식 중심이고 EndoVis 2017만 mask→bbox·시각화를 사용합니다.

백엔드 (`backends.py` · `backend_registry.py` · `hf_model_loader.py`):

| 경로 | 로드 방식 | 용도 |
|------|-----------|------|
| **`prismatic`** | `../backend/prismatic-vlms` + Hub id 또는 로컬 `.pt` + `config.json` | TRI-ML Prismatic 전용 |
| **`cosmos-*` / `qwen3-*` / `internvl*` / `paligemma*` / `groot`** | Hugging Face Hub → `AutoProcessor` + `model.generate()` | 로컬 GPU **추론(eval)만** |
| **`openai` / `gpt` / `gemini` / `claude`** | Cloud vision API (`api_backends.py`, JPEG base64) | API 키 필요, GPU 불필요 |

- `cosmos-32b` → `nvidia/Cosmos-Reason2-32B` (Qwen3-VL 계열; `cosmos-reason2` repo venv **불필요**)
- `qwen3-4b` / `qwen3-32b` → `Qwen/Qwen3-VL-*-Instruct`
- `internvl3.5` → `OpenGVLab/InternVL3_5-38B-HF` (**`-HF` 필수**, custom `InternVL3_5-38B` 아님)
- `paligemma2` → `google/paligemma2-28b-pt-224` (프롬프트 앞 `<image>` 자동 삽입)
- 크기·Hub id 전체: `backend_registry.py` · `BACKEND_CHOICES` / `DEFAULT_MODEL_IDS`

---

## 1. 구성

| 파일 | 역할 |
|------|------|
| `triplet_recognition_cholect50.py` | CholecT50 triplet 평가 |
| `phase_recognition_cholec80.py` | Cholec80 phase 평가 |
| `instrument_localization_endovis17.py` | EndoVis 2017 val instrument bbox localization |
| `endovis17_data.py` | mask→bbox 샘플 생성·프롬프트 |
| `scripts/build_endovis2017_bbox_annotations.py` | derived bbox JSON export |
| `cholect50_data.py` | challenge-val 라벨·프레임 로딩 |
| `cholec80_data.py` | Cholec80 phase 라벨·비디오 프레임 로딩 |
| `utils.py` | 라벨 파싱, bbox/IoU/mAP, resume, 시각화 |
| `backends.py` | VLM 로드·추론 (Prismatic / HF Auto) |
| `backend_registry.py` | `--backend` 별칭·기본 `model-id` |
| `hf_model_loader.py` | HF Hub `AutoProcessor` 로더 |
| `api_backends.py` | OpenAI / Gemini / Anthropic vision API |
| `grounding_task.sh` | 실행 런처 (`uv` + Python) |
| `setup_backend.sh` | **prismatic** `.venv` 설치 (uv) |
| `scripts/extract_cholec80_frames.sh` | Cholec80 0.1 fps 프레임 추출 → `../eval/cholec80/frames_0p1fps` |
| `scripts/build_cholec80_eval_phase_annotations.py` | eval frame용 `videoNN-phase.txt` manifest 생성 |

---

## 2. 태스크 개요

### CholecT50 Triplet recognition

한 프레임에서 (instrument, verb, target) triplet을 인식합니다.  
instrument별로 셸에서 스크립트를 반복 실행할 필요 없습니다.

#### 평가 프로토콜 (`--eval-protocol`)

| 프로토콜 | VLM 호출 | 설명 |
|----------|----------|------|
| `joint` (기본) | **프레임당 1회** | 한 질문에 모든 triplet. 같은 프레임의 GT annotation은 **응답 공유** |
| `sequential_gt` | **annotation당 3회** | instrument → verb → target 순서 질문. verb/target 질문의 context는 **GT** |
| `sequential_pred` | **annotation당 3회** | 순서 동일. verb/target 질문의 context는 **이전 단계 모델 예측** |

**`joint` — user prompt (MCQ 예)**

```
What tasks are the instruments accomplishing with the targets in this surgical image?
Answer with one triplet per line: <instrument, verb, target>

instrument: grasper, bipolar, hook, ...
verb: aspirate, clip, coagulate, ...
target: abd-wall/cavity, adhesion, ...
```

**기대 응답**

```
<grasper, grasp, gallbladder>
<hook, dissect, cystic-plate>
```

**`sequential_gt` / `sequential_pred` — 단계별 질문**

1. What is the **instrument**?
2. The instrument is {X}. What is the **verb**? — `X` = GT instrument (`gt`) 또는 1단계 예측 (`pred`)
3. The instrument is {X} and the verb is {Y}. What is the **target**?

각 단계는 해당 카테고리 이름만 답하도록 요청합니다 (MCQ면 해당 줄 옵션 목록 포함).

#### 프롬프트 모드 (`--prompt-mode`)

| 모드 | 설명 |
|------|------|
| `mcq` (기본) | instrument / verb / target **콤마 구분 옵션 목록** |
| `ov` | 옵션 없음 (open vocabulary) |

#### 평가 지표 (`metrics` in JSON)

- **Component Accuracy**: Instrument, Verb, Target 각각
- **Triplet Accuracy**: 세 컴포넌트가 모두 맞은 비율
- **mAP**: Instrument / Verb / Target 각 component별 mean AP

`joint`에서 예측 triplet이 여러 개면, GT triplet이 목록 **안에 포함**되면 정답.  
`sequential_*`에서는 최종 조합 1개 triplet으로 채점합니다.

### Cholec80 Phase Recognition

**Phase recognition** — 담낭절제 영상 프레임을 7개 수술 단계 중 하나로 분류합니다.

- **프롬프트**: *"In the Cholecystectomy surgical image, what is the current Phase? The available phase options are …"* (A–G)
- **기본 split**: `eval` = **video41–video80** (EndoNet evaluation set, 약 **9.8k** frame @ 0.1 fps)
- **지표**: Accuracy, 클래스별 Recall / Precision / Jaccard, macro 평균

#### 데이터 경로 (역할 분리)

| 경로 | 역할 |
|------|------|
| `../data/cholec80` (`CHOLEC80_ROOT`) | 원본 MP4, 25 fps `phase_annotations/` (추출·manifest 생성 시 참조) |
| `../eval/cholec80/frames_0p1fps` (`CHOLEC80_FRAMES_ROOT`) | **평가용** PNG + phase manifest (`phase_recognition` 기본 입력) |

**Eval frame 레이아웃** (`surgical_vlm_test` 기준 상대 경로 `../eval/cholec80/frames_0p1fps`):

```
frames_0p1fps/
  video41/
    000000.png          # 비디오 프레임 인덱스 0
    000250.png          # 인덱스 250 (0.1 fps, stride 250 @ 25 fps)
    ...
    video41-phase.txt   # Frame\tPhase (동일 인덱스)
  video42/
    ...
```

- PNG 파일명 `FFFFFF` = annotation `Frame` 컬럼 = 원본 MP4의 0-based frame index
- `phase_recognition`은 이 트리에서 **비디오 목록·라벨·이미지**를 모두 읽음 (MP4 디코드 불필요)
- 원본 phase annotation은 **25 fps**(매 프레임); eval manifest는 **0.1 fps** subsample (0, 250, 500, …)

| 설정 | 샘플링 | 용도 |
|------|--------|------|
| 기본 (`--frames-root` = eval manifest) | **0.1 fps** | 권장 eval |
| `--frame-stride 250` + MP4 (`--frames-root` 없음) | **0.1 fps** | 추출본 없을 때 |
| `--frame-stride 1` | **25 fps** | 매우 느림 (~98k+ calls / eval 40 videos) |

7개 phase: Preparation, Calot Triangle Dissection, Clipping and Cutting, Gallbladder Dissection, Gallbladder Packaging, Cleaning and Coagulation, Gallbladder Retraction.

실행 예시는 **§4.1 Cholec80 phase**를 참고하세요.

### EndoVis 2017 Instrument Localization

**Instrument localization** — `endovis2017` **val** split의 semantic mask에서 instrument별 tight bbox를 만들고, VLM이 **normalized bbox**를 예측합니다.

- **데이터**: `<surgical repo>/eval/endovis2017` — `val{1..10}/image/*.bmp`, `val*/label/*.bmp` (512×512)
- **GT 생성**: mask 픽셀 class id (1–7) → `instrument_type_mapping.json` → instrument별 axis-aligned bbox (pixel + **normalized [0,1]**)
- **Val에 등장하는 class**: 1 Bipolar, 2 Prograsp, 3 Large Needle Driver, 4 Vessel Sealer, 6 Monopolar Curved Scissors (5·7 없음)
- **샘플 수**: 프레임당 instrument 1개 = 1 query (image·label 쌍이 있는 val만; 현재 약 **1,310** samples). `val9`는 image 1장·label 300장으로 대부분 스킵됨
- **프롬프트**:

```
Where is the Large Needle Driver located? Answer the question with just a bounding box.
Format: [x_min, y_min, x_max, y_max]
Use normalized coordinates in [0, 1] relative to the image you see.
If the Large Needle Driver is not in the image, answer exactly: not present
```

- **VLM / viz**: 동일 square resize (`pil_side`, 예: 384×384). stretch resize이므로 normalized bbox는 native 512와 동일 비율
- **Cosmos**: 출력 0–1000 → 파서 **÷1000**
- **지표**: mIoU, mAP@50, mAP@75, COCO AP
- **시각화** (`--viz`): resize 이미지 위 GT(초록)/Pred(빨강), instrument 이름만

실행 예시는 **§4.1 EndoVis 2017 localization**을 참고하세요.

---

## 3. 사전 준비

### 3.1 데이터

**CholecT50 (triplet)**

- 라벨 (기본): `<surgical repo>/eval/cholect50-challenge-val/labels`
- 프레임: `CHOLECT50_VIDEOS_ROOT`로 지정 (예: `.../CholecT50/videos` 또는 challenge-val 내 `videos`)

**Cholec80 (phase)**

- 원본: `<surgical repo>/data/cholec80` — `videos/videoNN.mp4`, `phase_annotations/videoNN-phase.txt` (25 fps)
- **Eval frames (기본)**: `<surgical repo>/eval/cholec80/frames_0p1fps/`
  - `videoNN/{frame:06d}.png` + `videoNN/videoNN-phase.txt`
  - eval split: video **41–80**
- 추출 스크립트 출력 기본: `../eval/cholec80/frames_0p1fps` (`extract_cholec80_frames.sh`)

**EndoVis 2017 (localization)**

- 루트: `<surgical repo>/eval/endovis2017` (`ENDOVIS2017_ROOT`)
- 매핑: `instrument_type_mapping.json` (class id 1–7 → instrument name)
- val: `valN/image/seq_X_frameYYY.bmp`, `valN/label/seq_X_frameYYY.bmp` (mask, mode `L`)

### 3.2 인증 (HF Hub · Cloud API)

**Hugging Face** (로컬 HF / prismatic Hub):

```bash
cp /path/to/.hf_token surgical_vlm_test/.hf_token
```

**Cloud API** (`BACKEND=openai|gemini|claude` — `.hf_token` 불필요):

| Provider | 키 파일 (기본) | 환경 변수 |
|----------|----------------|-----------|
| OpenAI / GPT | `surgical_vlm_test/.openai_api_key` | `OPENAI_API_KEY` |
| Gemini | `.gemini_api_key` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| Claude / Anthropic | `.anthropic_api_key` | `ANTHROPIC_API_KEY` |

```bash
echo "$OPENAI_API_KEY" > surgical_vlm_test/.openai_api_key
chmod 600 surgical_vlm_test/.openai_api_key
```

또는 `--api-key-file /path/to/key` 로 지정.

### 3.3 백엔드 가상환경

```bash
cd surgical_vlm_test
bash setup_backend.sh              # all
bash setup_backend.sh prismatic    # 하나만
```

백엔드 repo 기본 경로: `surgical/../backend` (`VLA_ROOT_OVERRIDE`로 변경 가능)

- **prismatic**: `prismatic-vlms/.venv` (`bash setup_backend.sh prismatic`)
- **HF 모델**: `HF_PYTHON`으로 transformers가 설치된 인터프리터 지정 (conda `appgen`, `surgical` 등). 별도 `cosmos-reason2` / `GR00T-H` repo venv는 **필수 아님**.

### 3.4 HF Python 환경 (의존성)

`grounding_task.sh`는 `HF_PYTHON`이 없으면 conda `appgen` → `surgical` 순으로 탐색합니다.

**공통 (대부분의 HF VLM)**

```bash
export HF_PYTHON=/path/to/your/env/bin/python
$HF_PYTHON -m pip install torch transformers accelerate pillow
```

**모델군별 추가 패키지**

| 백엔드 | 추가 `pip install` | 비고 |
|--------|-------------------|------|
| `internvl` / `internvl3.5` | `timm` `einops` | `transformers>=4.52.1` 권장; 38B는 VRAM 큼 (A100×2 수준) |
| `paligemma2` | (없음) | chat template 없음 → `processor(text, images)` 경로; MCQ 품질은 **mix/instruct** checkpoint 권장 |
| `groot` | `gr00t` 패키지 (`../backend/GR00T-H`) | Eagle 번들 자동 보정 (`hf_model_loader`) |
| `cosmos-*` / `qwen3-*` | Qwen3-VL 지원 transformers | bbox localization 시 0–1000 → ÷1000 파싱 |

**HF_PYTHON 고정 예**

```bash
export HF_PYTHON=/NHNHOME/WORKSPACE/26msit001_T_B/KAIST-AIPRLab/.conda/envs/appgen/bin/python
```

### 3.5 HF 모델 캐시 (가중치)

스크립트는 **체크포인트를 새로 저장하지 않습니다.** Hub에서 받은 가중치는 `grounding_task.sh`가 설정하는 캐시에만 쌓입니다.

| 항목 | 기본 경로 |
|------|-----------|
| `HF_HOME` | `<surgical repo>/.cache/huggingface/` |
| **`HF_HUB_CACHE` (가중치)** | **`<surgical repo>/.cache/huggingface/hub/`** |
| Hub 스냅샷 예 | `.../hub/models--google--paligemma2-28b-pt-224/` |

`hf_model_loader.py` import 시 `configure_hf_cache()`로 env를 고정합니다.  
`grounding_task.sh`도 `HF_HUB_CACHE=$ROOT/../.cache/huggingface/hub` 를 export합니다.

> **레거시 캐시:** 예전에 `.../.cache/huggingface/models--*` ( **`hub/` 밖 상위 폴더** )에 받은 가중치가 있을 수 있습니다.  
> 새 설정은 `hub/` 를 우선합니다. 상위에만 있는 모델은 `hub/` 로 symlink 하거나 `export HF_HUB_CACHE` 를 상위로 맞추세요.

### 3.6 기본 Hub model-id · 출력 slug

| `--backend` | Hub `--model-id` (기본) | 출력 slug (`--model-name` 기본) |
|-------------|-------------------------|--------------------------------|
| `prismatic` | `prism-dinosiglip+7b` | `prismatic-7b` |
| `cosmos` / `cosmos-2b` | `nvidia/Cosmos-Reason2-2B` | `cosmos-reason2-2b` |
| `cosmos-32b` | `nvidia/Cosmos-Reason2-32B` | `cosmos-reason2-32b` |
| `qwen3` / `qwen3-4b` | `Qwen/Qwen3-VL-4B-Instruct` | `qwen3-vl-4b` |
| `qwen3-32b` | `Qwen/Qwen3-VL-32B-Instruct` | `qwen3-vl-32b` |
| `qwen2.5` | `Qwen/Qwen2.5-VL-32B-Instruct` | `qwen2.5-vl-32b` |
| `internvl` / `internvl3.5` | `OpenGVLab/InternVL3_5-38B-HF` | `internvl3.5-38b` |
| `paligemma` / `paligemma2` | `google/paligemma2-28b-pt-224` | `paligemma2-28b` |
| `groot` | `nvidia/GR00T-H` | `groot-h` |
| `gpt` / `openai` | `gpt-4o` | `gpt-4o` |
| `chatgpt` | `gpt-4o-mini` | `gpt-4o-mini` |
| `gemini` | `gemini-2.0-flash` | `gemini-2.0-flash` |
| `claude` / `anthropic` | `claude-sonnet-4-20250514` | `claude-sonnet-4` |

별칭: `BACKEND=cosmos2b` → `cosmos-2b`, `BACKEND=qwen3_32b` → `qwen3-32b`.

**InternVL:** Hub id에 **`-HF` suffix** 가 있는 transformers 표준 checkpoint만 `AutoProcessor` 경로와 호환됩니다.  
`OpenGVLab/InternVL3_5-38B`(HF 없음)는 OpenGVLab `internvl_chat` 코드용입니다.

**PaliGemma:** `*-pt-224` 는 pretrain; triplet MCQ에는 `google/paligemma2-10b-mix-448` 등 **mix/instruct** 를 `--model-id`로 지정하는 것을 권장합니다.

`--model-id`로 Hub id, `--model-name`으로 **결과 JSON이 들어가는 하위 폴더 이름**만 바꿀 수 있습니다.  
환경 변수 `MODEL_ID`, `MODEL_NAME`도 `grounding_task.sh`에서 동일하게 주입됩니다.

### 3.7 HF 추론 경로 (`backends.HfAutoBackend`)

| 모델군 | 입력 조립 |
|--------|-----------|
| Qwen-VL, Cosmos, InternVL `-HF` | `processor.apply_chat_template` (user + image) |
| PaliGemma 등 template 없음 | `processor(text="<image>\n" + prompt, images=...)` |
| Prismatic | `PurePromptBuilder` + `prismatic.generate` |

공통: 프레임 square resize → `model.generate` → 태스크 파서 → `outputs/...json` (resume 지원).

### 3.8 Cholec80 eval frame 준비 (최초 1회)

`extract_cholec80_frames.sh`는 비디오당 **ffmpeg 1회** 디코드로 0.1 fps PNG를 뽑고, 같은 폴더에 `videoNN-phase.txt`를 씁니다.

```bash
cd surgical_vlm_test

# eval 41–80 → ../eval/cholec80/frames_0p1fps/
CHOLEC80_ROOT=../data/cholec80 bash scripts/extract_cholec80_frames.sh

# PNG만 있고 manifest만 다시 만들 때
python3 scripts/build_cholec80_eval_phase_annotations.py \
  --dataset-root ../data/cholec80 --overwrite
```

**Cholec80 프레임 로딩 (eval 시 권장)**

1. **`--frames-root ../eval/cholec80/frames_0p1fps`** (기본) — PIL만, 가장 빠름·안정  
2. **`ffmpeg`** (`--frame-reader auto`, MP4 직접) — 추출본 없을 때  
3. **OpenCV** — venv에 `cv2`가 있을 때만 (새로 설치 시 numpy 충돌 주의)

```bash
# 기본 eval (grounding_task.sh가 frames-root·split 자동 주입)
bash grounding_task.sh phase_recognition_cholec80

# MP4에서 직접 (추출본 없을 때만)
bash grounding_task.sh phase_recognition_cholec80 --frame-stride 250
```

---

## 4. 실행

### 4.1 권장: `grounding_task.sh`

#### CholecT50 triplet

**Joint — 전체 평가** (기본, MCQ):

```bash
export CHOLECT50_VIDEOS_ROOT=/path/to/CholecT50/videos

BACKEND=prismatic DEVICE_VISIBLE=0 \
  bash grounding_task.sh triplet_recognition_cholect50 \
    --eval-protocol joint --prompt-mode mcq
```

**Sequential — GT context** (annotation당 VLM 3회, `--eval-all` 시 약 1319×3 forward):

```bash
export CHOLECT50_VIDEOS_ROOT=/path/to/CholecT50/videos

BACKEND=cosmos-32b DEVICE_VISIBLE=0 \
  bash grounding_task.sh triplet_recognition_cholect50 \
    --eval-protocol sequential_gt --prompt-mode mcq --eval-all
```

결과 JSON (기본):

`surgical_vlm_test/outputs/triplet_recognition_cholect50/triplet_cosmos-32b_cosmos-reason2-32b_mcq_sequential_gt/cholect50_challenge_val_triplet.json`

**Sequential — GT context** (짧은 테스트, backend만 바꿔도 됨):

```bash
bash grounding_task.sh triplet_recognition_cholect50 \
  --eval-protocol sequential_gt --prompt-mode mcq --video VID68
```

**Sequential — predicted context** (오류 전파):

```bash
bash grounding_task.sh triplet_recognition_cholect50 \
  --eval-protocol sequential_pred --prompt-mode mcq
```

**Open vocabulary + 샘플링**:

```bash
bash grounding_task.sh triplet_recognition_cholect50 \
  --eval-protocol joint --prompt-mode ov --samples-only --video VID68
```

**HF — 크기별 backend 예**:

```bash
# Qwen3-VL 4B (기본 qwen3 = qwen3-4b)
BACKEND=qwen3-4b HF_PYTHON=/path/to/python DEVICE_VISIBLE=0 \
  bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol joint

# Qwen3-VL 32B → outputs/.../triplet_qwen3-32b_qwen3-vl-32b_.../
BACKEND=qwen3-32b HF_PYTHON=/path/to/python DEVICE_VISIBLE=0 \
  bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol joint --max-samples 5

# Cosmos-Reason2 2B / 32B
BACKEND=cosmos-2b DEVICE_VISIBLE=0 bash grounding_task.sh phase_recognition_cholec80 --video 41
BACKEND=cosmos-32b DEVICE_VISIBLE=0 bash grounding_task.sh phase_recognition_cholec80 --video 41 --max-frames-per-video 5

# InternVL3.5 38B (HF) — timm/einops 필요, GPU 1장이면 OOM 가능
export HF_PYTHON=/path/to/env/bin/python
$HF_PYTHON -m pip install timm einops
BACKEND=internvl3.5 DEVICE_VISIBLE=1 \
  bash grounding_task.sh triplet_recognition_cholect50 \
    --eval-protocol sequential_gt --prompt-mode mcq --video VID68

# PaliGemma2 28B (로컬 캐시 있으면 hub에서 재다운로드 없음)
BACKEND=paligemma2 DEVICE_VISIBLE=0 \
  bash grounding_task.sh triplet_recognition_cholect50 \
    --eval-protocol sequential_gt --prompt-mode mcq --eval-all

# PaliGemma instruct/mix (품질 권장)
BACKEND=paligemma2 MODEL_ID=google/paligemma2-10b-mix-448 DEVICE_VISIBLE=0 \
  bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol joint --max-samples 10

# GPT-4o (OpenAI vision API)
BACKEND=gpt MODEL_ID=gpt-4o bash grounding_task.sh triplet_recognition_cholect50 \
  --eval-protocol sequential_gt --prompt-mode mcq --video VID68

# Gemini
BACKEND=gemini MODEL_ID=gemini-2.0-flash bash grounding_task.sh phase_recognition_cholec80 --video 41

# Claude
BACKEND=claude MODEL_ID=claude-3-5-sonnet-20241022 bash grounding_task.sh \
  instrument_localization_endovis17 --max-samples 5
```

> **주의:** `DEVICE_VISIBLE`(GPU index) 철자를 맞출 것 (`DEVISE_VISIBLE` 아님). API 백엔드는 GPU를 쓰지 않지만 `DEVICE_VISIBLE`은 무해합니다.

#### Cholec80 phase

`grounding_task.sh` 기본 주입:

- `--dataset-root` → `$CHOLEC80_ROOT` (`../data/cholec80`)
- `--frames-root` → `$CHOLEC80_FRAMES_ROOT` (`../eval/cholec80/frames_0p1fps`)
- `--split eval` (video41–80)

**스모크 테스트** (video 41, 5 frame):

```bash
BACKEND=prismatic DEVICE_VISIBLE=0 \
  bash grounding_task.sh phase_recognition_cholec80 \
    --video 41 --max-frames-per-video 5
```

**Eval 41–80, 0.1 fps** (eval frame dataset, 기본):

```bash
BACKEND=prismatic DEVICE_VISIBLE=0 \
  bash grounding_task.sh phase_recognition_cholec80
```

> **Train split (video01–40)** 또는 **25 fps** 전체 eval은 기본 파이프라인이 아닙니다.  
> train용 frame이 필요하면 `VID_START=1 VID_END=40 bash scripts/extract_cholec80_frames.sh` 후 `--split train`으로 실행하세요.

#### EndoVis 2017 localization

`grounding_task.sh`가 기본 `--dataset-root`를 `../eval/endovis2017`로 설정합니다.

**mask→bbox JSON만 export** (VLM 없음):

```bash
python3 scripts/build_endovis2017_bbox_annotations.py \
  --dataset-root ../eval/endovis2017 \
  --output ../eval/endovis2017/derived_bbox_annotations_val.json
```

**스모크 테스트** (5 samples):

```bash
BACKEND=prismatic DEVICE_VISIBLE=0 \
  bash grounding_task.sh instrument_localization_endovis17 --max-samples 5
```

**전체 val** (모든 val* split):

```bash
BACKEND=prismatic DEVICE_VISIBLE=0 \
  bash grounding_task.sh instrument_localization_endovis17
```

**단일 split** (예: val1):

```bash
bash grounding_task.sh instrument_localization_endovis17 --val-split val1 --max-samples 20
```

**Cosmos-Reason2 (HF Auto, bbox 0–1000 → ÷1000)**:

```bash
BACKEND=cosmos-2b DEVICE_VISIBLE=0 \
  bash grounding_task.sh instrument_localization_endovis17 --max-samples 10

BACKEND=cosmos-32b DEVICE_VISIBLE=0 \
  bash grounding_task.sh instrument_localization_endovis17 --max-samples 5
```

**시각화만 재생성**:

```bash
bash grounding_task.sh instrument_localization_endovis17 \
  --viz-only --force \
  --output outputs/instrument_localization_endovis17/loc_cosmos-32b_cosmos-reason2-32b/endovis2017_instrument_localization.json
```

**시각화 끄기**: `--no-viz`

### 4.2 Python 직접 실행

**CholecT50**:

```bash
uv run --python ../backend/prismatic-vlms/.venv/bin/python \
  triplet_recognition_cholect50.py \
  --backend prismatic \
  --dataset-root ../eval/cholect50-challenge-val \
  --videos-root /path/to/CholecT50/videos \
  --eval-protocol joint \
  --prompt-mode mcq
```

**Cholec80**:

```bash
uv run --python ../backend/prismatic-vlms/.venv/bin/python \
  phase_recognition_cholec80.py \
  --backend prismatic \
  --dataset-root ../data/cholec80 \
  --split eval \
  --frames-root ../eval/cholec80/frames_0p1fps
```

**EndoVis 2017**:

```bash
uv run --python ../backend/prismatic-vlms/.venv/bin/python \
  instrument_localization_endovis17.py \
  --backend prismatic \
  --dataset-root ../eval/endovis2017
```

### 4.3 주요 CLI 인자

**CholecT50 (`triplet_recognition_cholect50.py`)**

| 인자 | 설명 |
|------|------|
| `--eval-protocol {joint,sequential_gt,sequential_pred}` | 질문 방식 (기본: `joint`) |
| `--prompt-mode {mcq,ov}` | MCQ 콤마 옵션 vs open vocab (기본: `mcq`) |
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
| `--frame-stride N` | native 25 fps phase subsample (manifest 없을 때 `250` ≈ 0.1 fps) |
| `--frames-root` | eval frame 루트 (기본: `../eval/cholec80/frames_0p1fps`; `video41/000250.png` + `video41-phase.txt`) |
| `--frame-reader {auto,ffmpeg,opencv}` | MP4 디코드 방식 (기본 `auto` = ffmpeg 우선) |
| `--max-frames-per-video K` | 비디오당 최대 K장 (디버그용) |
| `--video 41` | 단일 비디오 (`41`, `video41` 모두 가능) |
| `--dataset-root` | Cholec80 루트 (기본: `../data/Cholec80`, `cholec80` 폴백) |
| `--force` / `--output` | triplet과 동일 |

**EndoVis 2017 (`instrument_localization_endovis17.py`)**

| 인자 | 설명 |
|------|------|
| `--dataset-root` | `endovis2017` 루트 (기본: `../eval/endovis2017`) |
| `--val-split` | `val1` 등 (반복 가능; 기본: 모든 `val*`) |
| `--instrument`, `--frame` | instrument slug / frame stem 필터 |
| `--min-mask-pixels` | GT bbox 최소 mask 픽셀 수 (기본 1) |
| `--export-annotations` | mask→bbox JSON만 저장 후 종료 |
| `--max-samples N` | 랜덤 subsample |
| `--viz` / `--no-viz` | VLM resize 이미지 위 GT/Pred/comparison (기본 on) |
| `--viz-only` / `--viz-side` | 시각화만 재생성 |
| `--force` / `--output` | resume·재추론·결과 경로 |

**공통**

| 인자 | 설명 |
|------|------|
| `--backend` | `backend_registry.BACKEND_CHOICES` (크기별: `cosmos-32b`, `qwen3-4b`, …) |
| `--model-id` | Hub model id override |
| `--model-name` | 출력 폴더 slug override (미지정 시 backend별 기본, 예: `cosmos-reason2-32b`) |
| `--device` | `0`, `cuda:0`, `cpu` → 내부 `torch.device` |
| `--hf-token` | 기본 `surgical_vlm_test/.hf_token` |
| `--max-new-tokens` | 생성 최대 토큰 수 |
| `--api-key-file` | Cloud API 키 파일 (openai/gemini/claude) |
| `--api-timeout-sec` | Cloud API HTTP timeout (기본 120) |

### 4.4 결과 저장 위치 (eval JSON)

모든 경로는 `surgical_vlm_test/` 기준 상대 경로입니다. `{model}` = `--model-name` 생략 시 `backend_registry.BACKEND_OUTPUT_SLUGS` (예: `cosmos-reason2-32b`).

| 태스크 | 디렉터리 패턴 | JSON 파일명 |
|--------|---------------|-------------|
| Triplet | `outputs/triplet_recognition_cholect50/triplet_{backend}_{model}_{prompt_mode}_{eval_protocol}/` | `cholect50_challenge_val_triplet.json` |
| Phase | `outputs/phase_recognition_cholec80/phase_{backend}_{model}_{split}/` | `cholec80_phase_0p1fps_manifest.json` (eval frames) |
| EndoVis 2017 | `outputs/instrument_localization_endovis17/loc_{backend}_{model}/` | `endovis2017_instrument_localization.json` |

**예시 (`BACKEND=cosmos-32b`, triplet, sequential_gt, mcq, eval-all):**

```
outputs/triplet_recognition_cholect50/
  triplet_cosmos-32b_cosmos-reason2-32b_mcq_sequential_gt/
    cholect50_challenge_val_triplet.json
```

**예시 (InternVL3.5 38B, triplet sequential_gt):**

```
outputs/triplet_recognition_cholect50/
  triplet_internvl3.5_internvl3.5-38b_mcq_sequential_gt/
    cholect50_challenge_val_triplet.json
```

**예시 (PaliGemma2 28B):**

```
outputs/triplet_recognition_cholect50/
  triplet_paligemma2_paligemma2-28b_mcq_sequential_gt/
    cholect50_challenge_val_triplet.json
```

`--output` / `--output-root`로 경로 override. 동일 JSON으로 재실행 시 **resume**; `--force`로 전부 재추론.

JSON 필드 요약:

- **triplet** `results[]`: `input` / `output` / `evaluation`; `sequential_*`는 `output.sequential_steps`
- **triplet** `metrics`: component accuracy, triplet accuracy, mAP
- **phase** `metrics`: `accuracy`, `macro_recall`, `macro_precision`, `macro_jaccard`, `per_class`
- **EndoVis** `results[]`: `label_context` (mask bbox GT), `output.parsed` (bbox); `metrics`: mIoU, mAP@50/75, COCO AP

---

## 5. 환경 변수 (`grounding_task.sh`)

| 변수 | 설명 |
|------|------|
| `BACKEND` | `prismatic` \| `qwen3-32b` \| `cosmos-32b` \| `gpt` \| `gemini` \| `claude` \| … |
| `API_PYTHON` | Cloud API용 Python (기본: `surgical` / `appgen` / `python3`) |
| `HF_PYTHON` | non-prismatic 백엔드용 Python (`torch`, `transformers` 필요) |
| `DEVICE_VISIBLE` | → `CUDA_VISIBLE_DEVICES` (기본 `0`). **철자 주의** (`DEVISE_VISIBLE` 아님) |
| `MODEL_ID` | `--model-id` 자동 주입 (Hub repo id) |
| `MODEL_NAME` | `--model-name` 자동 주입 (출력 하위 폴더 slug) |
| `HF_HOME` | 기본 `../.cache/huggingface` |
| `HF_HUB_CACHE` | 기본 `../.cache/huggingface/hub` (가중치 스냅샷) |
| `CHOLECT50_CHALLENGE_VAL_ROOT` | triplet `--dataset-root` (기본: `../eval/cholect50-challenge-val`) |
| `CHOLECT50_VIDEOS_ROOT` | triplet `--videos-root` |
| `CHOLEC80_ROOT` | phase `--dataset-root` (기본: `../data/Cholec80`, `cholec80` 폴백) |
| `CHOLEC80_EVAL_ROOT` | eval 데이터 루트 (기본: `../eval/cholec80`) |
| `CHOLEC80_FRAMES_ROOT` | phase `--frames-root` (기본: `$CHOLEC80_EVAL_ROOT/frames_0p1fps`) |
| `ENDOVIS2017_ROOT` | localization `--dataset-root` (기본: `../eval/endovis2017`) |
| `PRISMATIC_PYTHON` | prismatic venv python override |
| `GROUNDING_TASK_AUTO_BACKEND_SETUP` | `0`이면 prismatic uv 자동 설치 스킵 |

---

## 6. `surgical_vlm_grounding`과의 차이

| | **surgical_vlm_test** | **surgical_vlm_grounding** |
|---|------------------------|------------------------------|
| 목적 | CholecT50 triplet · Cholec80 phase · EndoVis 2017 bbox eval | Localization, language/visual grounding 등 |
| CholecT50 | `triplet_recognition_cholect50.py` | `localization_cholect50.py`, `language_grounding_v*`, … |
| Cholec80 | `phase_recognition_cholec80.py` | (별도 phase 스크립트 없음) |
| EndoVis 2017 | `instrument_localization_endovis17.py` | mask segmentation (별도) |
| Bbox | EndoVis 2017 mask→bbox + viz | CholecT50 localization 등 |
| 입력 | T50: 추출 프레임 / C80: `eval/cholec80/frames_0p1fps` PNG+manifest | 주로 추출 프레임 |
| Triplet | `joint` 1질문/프레임 또는 `sequential_*` 3질문/annotation | multi-step localization |
| Phase | 7-class MCQ (A–G) | localization + phase |
| 추론 | task·프로토콜별 (T50 joint=1회/프레임) | task별 상이 |

---

## 7. 문제 해결

1. **CholecT50 프레임 이미지 없음** — `CHOLECT50_VIDEOS_ROOT` 또는 `--videos-root` 확인  
2. **Cholec80 원본 없음** — `CHOLEC80_ROOT`에 `phase_annotations/`, `videos/` 확인 (`Cholec80` vs `cholec80` 폴백)  
3. **Cholec80 eval frame 없음** — `../eval/cholec80/frames_0p1fps/video41/000000.png` 및 `video41-phase.txt` 확인; 없으면 `bash scripts/extract_cholec80_frames.sh`  
4. **프레임 로드 실패** — eval PNG 경로·파일명(`{frame:06d}.png`)이 manifest `Frame`과 일치하는지 확인; MP4 fallback은 `ffmpeg` 필요  
5. **백엔드 import 실패** — prismatic: `bash setup_backend.sh prismatic` / HF: `HF_PYTHON`에 `torch`, `transformers` 확인  
5a. **InternVL `No module named timm`** — `pip install timm einops` in `HF_PYTHON` env  
5b. **InternVL wrong checkpoint** — `MODEL_ID=OpenGVLab/InternVL3_5-38B-HF` (`-HF` suffix). non-HF `InternVL3_5-38B`는 custom 포맷  
5c. **`apply_chat_template` / no chat template** — PaliGemma: 최신 `backends.py`가 `<image>` prefix + `processor(text, images)` fallback 사용  
5d. **PaliGemma `<image>` warning** — 동일; 프롬프트 앞 `<image>` 자동 추가됨 (재실행 시 최신 코드 필요)  
5e. **`TypeError: torch.device is not iterable`** — 최신 `hf_model_loader.py` 사용 (`torch.device` 지원)  
6. **HF 401** — `.hf_token` 경로 및 권한  
6b. **32B / 38B OOM** — `cosmos-32b`, `qwen3-32b`, `internvl3.5`는 VRAM 큼; `--video`, `--max-samples`, `--max-frames-per-video`로 축소  
6c. **캐시를 못 찾고 재다운로드** — 가중치가 `.../huggingface/models--*`(상위)에만 있으면 `hub/`로 symlink 또는 `HF_HUB_CACHE` 조정 (§3.5)  
6d. **API HTTP 401/403** — `.openai_api_key` / `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` 확인; `--model-id`가 vision 지원 모델인지 확인  
6e. **API 비용·속도** — `sequential_gt` + `--eval-all`은 호출 수 많음; 스모크는 `--video` / `--max-samples` 권장  
7. **resume** — 동일 `--output` JSON에 이어서 실행; 전체 재실행은 `--force`  
8. **triplet sequential이 느림** — annotation당 VLM 3회; 빠른 테스트는 `--eval-protocol joint` 또는 `--samples-only`  
9. **phase eval이 너무 느림** — 기본 `../eval/cholec80/frames_0p1fps`(0.1 fps, ~9.8k calls); 25 fps는 `--frame-stride 1`  
10. **프롬프트/프로토콜 변경 후** — 이전 JSON과 혼동 방지를 위해 `--force` 권장  
11. **EndoVis 2017 데이터 없음** — `eval/endovis2017/val*/image`, `val*/label` 확인  
12. **Qwen-VL / Cosmos bbox** — 0–1000 좌표 → 자동 ÷1000 (`bbox_coord_space=qwen_1000`)  
13. **EndoVis viz** — `--viz-only` + `endovis2017_instrument_localization.json`

```bash
python triplet_recognition_cholect50.py --help
python phase_recognition_cholec80.py --help
python instrument_localization_endovis17.py --help
bash grounding_task.sh help
```
