# surgical_vlm_test

CholecT50 **triplet recognition** · Cholec80 **phase recognition** · EndoVis 2017 **instrument localization** · EndoVis 2018 VQA **tissue/instrument recognition** · Endoscapes **CVS evaluation** 벤치를 위한 독립 패키지입니다.  
`surgical_vlm_grounding`과 분리되어 있으며, CholecT50/80·EndoVis 18 VQA는 분류·인식(MCQ) 중심이고 EndoVis 2017만 mask→bbox·시각화를 사용합니다.

백엔드 (`backends.py` · `backend_registry.py` · `hf_model_loader.py`):

| 경로 | 로드 방식 | 용도 |
|------|-----------|------|
| **`prismatic`** | `../backend/prismatic-vlms` + Hub id 또는 로컬 `.pt` + `config.json` | TRI-ML Prismatic 전용 |
| **`cosmos-*` / `qwen3-*` / `internvl*` / `paligemma*` / `groot`** | Hugging Face Hub → `AutoProcessor` + `model.generate()` | 로컬 GPU **추론(eval)만** |
| **PEFT LoRA** (`--model-id` = 어댑터 Hub/로컬 경로) | 베이스 VLM + `peft` (`hf_model_loader`) | 예: [surgsigma_qwen3vl_full](https://huggingface.co/khtks/Qwen3-VL/tree/main/surgsigma_qwen3vl_full) |
| **`openai` / `gpt` / `gemini` / `claude`** | Cloud vision API (`api_backends.py`, JPEG base64) | API 키 필요, GPU 불필요 |

- `cosmos-32b` → `nvidia/Cosmos-Reason2-32B` (Qwen3-VL 계열; `cosmos-reason2` repo venv **불필요**)
- `qwen3-4b` / `qwen3-32b` → `Qwen/Qwen3-VL-*-Instruct` (풀 weight; LoRA는 `--model-id`로 어댑터 지정)
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
| `tissue_instrument_recognition_endovis18.py` | EndoVis 2018 VQA Classification MCQ 평가 |
| `endovis17_data.py` | mask→bbox 샘플 생성·프롬프트 |
| `endovis18_vqa_data.py` | EndoVis 18 VQA Classification QA·보기·이미지 매핑 |
| `cvs_evaluation_endoscapes.py` | Endoscapes CVS (yes/no × 3 criteria) |
| `endoscapes_cvs_data.py` | Endoscapes COCO `ds` → binary GT·샘플 로드 |
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

### EndoVis 2018 VQA — Tissue / Instrument Recognition

**Tissue & instrument recognition** — EndoVis 2018 VQA **Classification** QA를 MCQ로 평가합니다 (논문의 open-vocab BLEU/ROUGE 대신 phase recognition과 동일한 분류 지표).

- **질문**: `eval/EndoVis-18-VQA/seq_*/vqa/Classification/frame*_QA.txt` 의 **모든** `질문|정답` 줄 (프레임당 최대 10문항)
- **이미지**: `eval/endovis2018/{val|train}/image/seq_{N}_frame{idx}.bmp` (`frame015_QA.txt` → `seq_N_frame015.bmp`)
- **프롬프트 (MCQ)**: 질문 + 해당 유형별 **키워드 옵션 목록** + **키워드 하나만** 답하도록 지시

**질문 유형 · 보기**

| 유형 | 질문 예 | 옵션 수 |
|------|---------|--------|
| organ | What organ is being operated? | 1 (`kidney`) |
| state | What is the state of {instrument}? | 13 (`Idle`, `Looping`, `Tissue_Manipulation`, …) |
| location | Where is {instrument} located? | 4 (`left-top`, `right-top`, `left-bottom`, `right-bottom`) |

**프롬프트 예 (state)**

```
What is the state of bipolar_forceps?

Answer with exactly one keyword from the options below.
Reply with the keyword only — no extra words, labels, or punctuation.

Options: Idle, Looping, Grasping, Retraction, Tissue_Manipulation, ...
```

**기대 응답**: `Looping` (옵션 중 키워드 하나)

#### 평가 지표 (`metrics` in JSON)

Cholec80 phase recognition과 동일 (macro는 **support > 0** 클래스만 평균):

- **Accuracy**: QA쌍 전체 exact match
- **macro_recall** / **macro_precision** / **macro_jaccard** (IoU)

`per_class` breakdown은 JSON에 **포함하지 않습니다**.

#### 데이터 규모 (`--image-split` 기본 `val`)

| split | QA쌍 | 프레임 | 비고 |
|-------|------|--------|------|
| `val` (기본) | ~1,396 | ~284 | seq 9·10 위주 + 일부 seq |
| `train` | ~6,525 | ~995 | |
| `both` | ~7,921 | ~1,279 | seq 11–16은 이미지 없음 → 제외 |

VLM 호출 수 = QA쌍 수 (질문마다 1회).

실행 예시는 **§4.1 EndoVis 2018 tissue/instrument** · **§3.1**을 참고하세요.

### Endoscapes CVS Evaluation

**Critical View of Safety** — 프레임마다 CVS **3기준**을 **yes/no**로 평가 (Endoscapes2023).

- **GT**: COCO `images[].ds` = `[C1, C2, C3]` (전문가 3명 평균 0~1) → `--gt-threshold 0.5`로 이진화
- **C1**: Only two tubular structures connect to the gallbladder.
- **C2**: Hepatocystic triangle cleared for visibility.
- **C3**: Lower gallbladder detached from liver bed.

| `--eval-protocol` | VLM 호출 | 설명 |
|-------------------|----------|------|
| `joint` (기본) | **프레임당 1회** | 3줄 `C1: yes` 형식 |
| `per_criterion` | **프레임당 3회** | 기준별 MCQ (`yes` / `no`) |

**지표** (논문 C.1.6): **Average Accuracy**, **Balanced Accuracy** (+ `per_criterion` 요약)

| `--split` | 프레임 수 (joint) |
|-----------|-------------------|
| `test` (기본) | 312 (CVS201 test) |
| `val` / `train` | validation / train |
| `test_seg` / `val_seg` / `train_seg` | 74 / 76 / 343 (Seg50 subset) |

데이터: `eval/endoscapes/{split}/annotation_coco.json` + 동일 폴더의 `.jpg`

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

**EndoVis 2018 VQA (tissue/instrument recognition)**

| 경로 | 역할 |
|------|------|
| `<surgical repo>/eval/EndoVis-18-VQA` (`ENDOVIS18_VQA_ROOT`) | `seq_*/vqa/Classification/frame*_QA.txt` |
| `<surgical repo>/eval/endovis2018` (`ENDOVIS2018_IMAGES_ROOT`) | `val/image/`, `train/image/` — `seq_{N}_frame{idx}.bmp` |

- QA 파일명 `frame015_QA.txt` → 이미지 `seq_{N}_frame015.bmp` (zero-padding 3자리·무패딩 모두 시도)
- `EndoVis_18_VQA.csv`는 참고용; eval 스크립트는 Classification QA + bmp만 사용

**Endoscapes (CVS)**

- 루트: `<surgical repo>/eval/endoscapes` (`ENDOSCAPES_ROOT`)
- split 폴더: `train_seg` / `val_seg` / `test_seg` (Seg50) 또는 `train` / `val` / `test` (CVS201)
- annotation: `annotation_coco.json` (없으면 `annotation_ds_coco.json`)
- 이미지: `{split}/{video_id}_{frame}.jpg` · GT: `images[].ds`

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
- **HF 모델**: `HF_PYTHON`으로 transformers가 설치된 인터프리터 지정 (기본: conda **`surgical`**). 별도 `cosmos-reason2` / `GR00T-H` repo venv는 **필수 아님**.

### 3.4 HF Python 환경 (의존성)

`grounding_task.sh`는 `HF_PYTHON`이 없으면 conda **`surgical`** 을 기본으로 사용합니다.

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
| PEFT LoRA adapter (`adapter_config.json`) | **`peft`** | `--model-id`가 어댑터 폴더/Hub 경로면 base 자동 로드 |

#### PEFT LoRA 어댑터 (베이스 + adapter 자동 로드)

일부 Hub checkpoint는 **풀 VLM weight가 아니라 LoRA 어댑터만** 올라와 있습니다 (`adapter_config.json`, `adapter_model.safetensors`, tokenizer 등).  
`hf_model_loader.load_hf_vlm()`은 `adapter_config.json`을 감지하면:

1. `base_model_name_or_path` (또는 `HF_BASE_MODEL_ID`)로 **베이스** `Qwen3VLForConditionalGeneration` 등 로드  
2. `PeftModel.from_pretrained(base, adapter_dir)` 적용  
3. `AutoProcessor`는 **어댑터 폴더 우선**, 없으면 베이스에서 로드  
4. 이후 `HfAutoBackend.generate()` — `BACKEND=qwen3-4b` 등 기존 HF 태스크와 동일

**지원 `MODEL_ID` 형식**

| 형식 | 예 |
|------|-----|
| Hub `org/repo/subfolder` | `khtks/Qwen3-VL/surgsigma_qwen3vl_full` |
| 로컬 디렉터리 | `/path/to/surgsigma_qwen3vl_full` (안에 `adapter_config.json`) |

**의존성**

```bash
export HF_PYTHON=/path/to/env/bin/python
$HF_PYTHON -m pip install peft
# 베이스 + 어댑터: torch, transformers, accelerate (공통 HF와 동일)
```

**실행 예 (SurgSigma Qwen3-VL LoRA on 4B base)**

```bash
BACKEND=qwen3-4b DEVICE_VISIBLE=0 \
  MODEL_ID=khtks/Qwen3-VL/surgsigma_qwen3vl_full \
  bash grounding_task.sh triplet_recognition_cholect50 \
    --eval-protocol joint --max-samples 5

# sequential GT (스모크) — 출력은 base(qwen3-vl-4b)와 다른 폴더
# outputs/.../triplet_qwen3-4b_surgsigma_qwen3vl_full_mcq_sequential_gt/
BACKEND=qwen3-4b DEVICE_VISIBLE=0 \
  MODEL_ID=khtks/Qwen3-VL/surgsigma_qwen3vl_full \
  bash grounding_task.sh triplet_recognition_cholect50 \
    --eval-protocol sequential_gt --prompt-mode mcq --video VID68
```

**환경 변수**

| 변수 | 설명 |
|------|------|
| `MODEL_ID` | 어댑터 Hub id 또는 로컬 경로 (`grounding_task.sh`가 `--model-id`로 전달) |
| `HF_BASE_MODEL_ID` | `adapter_config.json`의 `base_model_name_or_path` 대신 베이스 Hub id 강제 |
| `HF_PEFT_MERGE` | `1` / `true` 이면 추론 전 `merge_and_unload()` (LoRA를 베이스에 합침, VRAM·속도 trade-off) |

**출력 JSON `vlm_load` 예**

```json
{
  "source": "hf_peft_adapter",
  "hub_model_id": "khtks/Qwen3-VL/surgsigma_qwen3vl_full",
  "base_model_id": "Qwen/Qwen3-VL-4B-Instruct",
  "peft_type": "LORA",
  "peft_merged": false,
  "loader": "PeftModel"
}
```

**주의**

- `BACKEND`는 베이스 **아키텍처**에 맞춰야 합니다 (4B LoRA → `qwen3-4b`, 32B base LoRA → `qwen3-32b`).  
- 어댑터만 `MODEL_ID`에 넣고 **풀 모델 id**(`Qwen/Qwen3-VL-4B-Instruct`)를 그대로 쓰면 LoRA가 적용되지 않습니다.  
- `cosmos-reason2` / `prismatic` **repo `.venv`는 불필요** — `HF_PYTHON` + Hub 캐시만 사용.

**HF_PYTHON 고정 예** (기본값과 동일; 다른 env 쓸 때만 명시)

```bash
export HF_PYTHON=/NHNHOME/WORKSPACE/26msit001_T_B/KAIST-AIPRLab/.conda/envs/surgical/bin/python
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

**PEFT 어댑터** — `--model-id`로 어댑터를 지정하고, `--model-name`으로 출력 slug를 지정합니다 (베이스 기본 slug와 별개).

| `--backend` | `--model-id` (예) | `--model-name` (권장 slug) |
|-------------|-------------------|---------------------------|
| `qwen3-4b` | `khtks/Qwen3-VL/surgsigma_qwen3vl_full` | `surgsigma-qwen3vl-full` |

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
| PEFT LoRA on Qwen3-VL | 베이스와 동일 (`PeftModel` + Qwen chat template) |
| PaliGemma 등 template 없음 | `processor(text="<image>\n" + prompt, images=...)` |
| Prismatic | `PurePromptBuilder` + `prismatic.generate` |

공통: 프레임 square resize → `model.generate` → 태스크 파서 → `outputs/...json` (resume 지원).  
PEFT 경로는 `load_hf_vlm` 진입 시 `adapter_config.json` 유무로 자동 분기 (`load_hf_vlm_peft_adapter`).

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

# PEFT LoRA (SurgSigma on Qwen3-VL-4B) — pip install peft 필요
export HF_PYTHON=/path/to/env/bin/python
$HF_PYTHON -m pip install peft
BACKEND=qwen3-4b DEVICE_VISIBLE=0 \
  MODEL_ID=khtks/Qwen3-VL/surgsigma_qwen3vl_full \
  MODEL_NAME=surgsigma-qwen3vl-full \
  bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol joint --max-samples 5

# LoRA merge 후 추론 (선택)
HF_PEFT_MERGE=1 BACKEND=qwen3-4b MODEL_ID=khtks/Qwen3-VL/surgsigma_qwen3vl_full \
  bash grounding_task.sh phase_recognition_cholec80 --video 41 --max-frames-per-video 3

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

#### EndoVis 2018 tissue / instrument recognition

`grounding_task.sh`가 기본 주입:

- `--vqa-root` → `$ENDOVIS18_VQA_ROOT` (`../eval/EndoVis-18-VQA`)
- `--images-root` → `$ENDOVIS2018_IMAGES_ROOT` (`../eval/endovis2018`)
- `--image-split val` (스크립트 기본)

**스모크 테스트** (seq 2, 5 QA쌍):

```bash
BACKEND=qwen2.5 DEVICE_VISIBLE=0 \
  bash grounding_task.sh tissue_instrument_recognition_endovis18 \
    --seq 2 --max-samples 5
```

**Val 전체** (~1,396 VLM 호출):

```bash
BACKEND=cosmos-32b DEVICE_VISIBLE=0 \
  bash grounding_task.sh tissue_instrument_recognition_endovis18
```

**Train + val**:

```bash
BACKEND=gemini MODEL_ID=gemini-2.0-flash \
  bash grounding_task.sh tissue_instrument_recognition_endovis18 \
    --image-split both
```

**Cloud API 스모크**:

```bash
BACKEND=gpt MODEL_ID=gpt-4o \
  bash grounding_task.sh tissue_instrument_recognition_endovis18 \
    --seq 10 --max-samples 10
```

#### Endoscapes CVS

```bash
# 스모크 (test, 3프레임, joint)
BACKEND=qwen2.5 DEVICE_VISIBLE=0 \
  bash grounding_task.sh cvs_evaluation_endoscapes \
    --max-samples 3

# test 전체 (312 VLM calls, joint)
BACKEND=cosmos-32b DEVICE_VISIBLE=0 \
  bash grounding_task.sh cvs_evaluation_endoscapes

# 기준별 3회 호출
BACKEND=prismatic DEVICE_VISIBLE=0 \
  bash grounding_task.sh cvs_evaluation_endoscapes \
    --eval-protocol per_criterion --split val_seg

# CVS201 full test split
bash grounding_task.sh cvs_evaluation_endoscapes --split test --max-samples 20
```

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

**EndoVis 2018 VQA**:

```bash
uv run --python ../backend/prismatic-vlms/.venv/bin/python \
  tissue_instrument_recognition_endovis18.py \
  --backend prismatic \
  --vqa-root ../eval/EndoVis-18-VQA \
  --images-root ../eval/endovis2018 \
  --image-split val \
  --seq 2 --max-samples 5
```

**PEFT LoRA (SurgSigma Qwen3-VL)** — `HF_PYTHON`에 `peft` 설치 필요:

```bash
uv run --python /path/to/hf-env/bin/python \
  triplet_recognition_cholect50.py \
  --backend qwen3-4b \
  --model-id khtks/Qwen3-VL/surgsigma_qwen3vl_full \
  --model-name surgsigma-qwen3vl-full \
  --eval-protocol joint \
  --max-samples 5
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

**EndoVis 2018 VQA (`tissue_instrument_recognition_endovis18.py`)**

| 인자 | 설명 |
|------|------|
| `--vqa-root` | `EndoVis-18-VQA` 루트 (기본: `../eval/EndoVis-18-VQA`) |
| `--images-root` | `endovis2018` 루트 (기본: `../eval/endovis2018`) |
| `--image-split {val,train,both}` | bmp split (기본: `val`) |
| `--seq 2` | 단일 sequence (`2` 또는 `seq_2`) |
| `--max-samples N` | 최대 N개 QA쌍만 평가 |
| `--force` / `--output` | resume·재추론·결과 경로 |

**Endoscapes CVS (`cvs_evaluation_endoscapes.py`)**

| 인자 | 설명 |
|------|------|
| `--dataset-root` | `endoscapes` 루트 (기본: `../eval/endoscapes`) |
| `--split` | `train` / `val` / `test` / `train_seg` / `val_seg` / `test_seg` (기본: `test`) |
| `--eval-protocol {joint,per_criterion}` | 1회 vs 기준별 3회 (기본: `joint`) |
| `--gt-threshold` | `ds[k] >= threshold` → yes (기본 `0.5`) |
| `--annotation-file` | COCO JSON 이름 override |
| `--video` | `video_id` 필터 |
| `--max-samples` | 최대 샘플 수 |

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
| EndoVis 2018 VQA | `outputs/tissue_instrument_recognition_endovis18/tir_{backend}_{model}_mcq_{split}/` | `endovis18_tir_{split}.json` (`split` = `val` / `train` / `both`) |
| Endoscapes CVS | `outputs/cvs_evaluation_endoscapes/cvs_{backend}_{model}_{protocol}_{split}/` | `endoscapes_cvs_{split}.json` |

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

**예시 (PEFT LoRA, `MODEL_ID=khtks/Qwen3-VL/surgsigma_qwen3vl_full`)** — 베이스 기본 id와 다르면 Hub subfolder 이름으로 폴더 분리:

```
outputs/triplet_recognition_cholect50/
  triplet_qwen3-4b_surgsigma_qwen3vl_full_mcq_sequential_gt/
    cholect50_challenge_val_triplet.json
```

(`MODEL_NAME`으로 slug를 덮어쓸 수 있음.)

`--output` / `--output-root`로 경로 override. 동일 JSON으로 재실행 시 **resume**; `--force`로 전부 재추론.

JSON 필드 요약:

- **triplet** `results[]`: `input` / `output` / `evaluation`; `sequential_*`는 `output.sequential_steps`
- **triplet** `metrics`: component accuracy, triplet accuracy, mAP
- **phase** `metrics`: `accuracy`, `macro_recall`, `macro_precision`, `macro_jaccard`, `per_class`
- **EndoVis 2017** `results[]`: `label_context` (mask bbox GT), `output.parsed` (bbox); `metrics`: mIoU, mAP@50/75, COCO AP
- **EndoVis 2018 VQA** `results[]`: `label_context` (question, gold_keyword, options), `output.parsed.keyword`; `metrics`: `accuracy`, `macro_recall`, `macro_precision`, `macro_jaccard` (no `per_class`)
- **Endoscapes CVS** `metrics`: `average_accuracy`, `balanced_accuracy`, `per_criterion` (C1–C3 accuracy/BA); joint은 `evaluation.per_criterion[]`

---

## 5. 환경 변수 (`grounding_task.sh`)

| 변수 | 설명 |
|------|------|
| `BACKEND` | `prismatic` \| `qwen3-32b` \| `cosmos-32b` \| `gpt` \| `gemini` \| `claude` \| … |
| `API_PYTHON` | Cloud API용 Python (기본: `HF_PYTHON` → conda `surgical` / `python3`) |
| `HF_PYTHON` | non-prismatic 백엔드용 Python (`torch`, `transformers` 필요) |
| `DEVICE_VISIBLE` | → `CUDA_VISIBLE_DEVICES` (기본 `0`). **철자 주의** (`DEVISE_VISIBLE` 아님) |
| `MODEL_ID` | `--model-id` 자동 주입 (풀 Hub id **또는** PEFT 어댑터 경로) |
| `MODEL_NAME` | `--model-name` 자동 주입 (출력 slug; 생략 시 `MODEL_ID`가 백엔드 기본과 다르면 Hub tail 사용) |
| `HF_BASE_MODEL_ID` | PEFT 어댑터의 베이스 Hub id override (`adapter_config`보다 우선) |
| `HF_PEFT_MERGE` | `1`이면 LoRA `merge_and_unload()` 후 추론 |
| `HF_HOME` | 기본 `../.cache/huggingface` |
| `HF_HUB_CACHE` | 기본 `../.cache/huggingface/hub` (가중치 스냅샷) |
| `CHOLECT50_CHALLENGE_VAL_ROOT` | triplet `--dataset-root` (기본: `../eval/cholect50-challenge-val`) |
| `CHOLECT50_VIDEOS_ROOT` | triplet `--videos-root` |
| `CHOLEC80_ROOT` | phase `--dataset-root` (기본: `../data/Cholec80`, `cholec80` 폴백) |
| `CHOLEC80_EVAL_ROOT` | eval 데이터 루트 (기본: `../eval/cholec80`) |
| `CHOLEC80_FRAMES_ROOT` | phase `--frames-root` (기본: `$CHOLEC80_EVAL_ROOT/frames_0p1fps`) |
| `ENDOVIS2017_ROOT` | localization `--dataset-root` (기본: `../eval/endovis2017`) |
| `ENDOVIS18_VQA_ROOT` | tissue/instrument `--vqa-root` (기본: `../eval/EndoVis-18-VQA`) |
| `ENDOVIS2018_IMAGES_ROOT` | tissue/instrument `--images-root` (기본: `../eval/endovis2018`) |
| `ENDOSCAPES_ROOT` | CVS `--dataset-root` (기본: `../eval/endoscapes`) |
| `PRISMATIC_PYTHON` | prismatic venv python override |
| `GROUNDING_TASK_AUTO_BACKEND_SETUP` | `0`이면 prismatic uv 자동 설치 스킵 |

---

## 6. `surgical_vlm_grounding`과의 차이

| | **surgical_vlm_test** | **surgical_vlm_grounding** |
|---|------------------------|------------------------------|
| 목적 | CholecT50 triplet · Cholec80 phase · EndoVis 2017 bbox · EndoVis 18 VQA MCQ | Localization, language/visual grounding 등 |
| CholecT50 | `triplet_recognition_cholect50.py` | `localization_cholect50.py`, `language_grounding_v*`, … |
| Cholec80 | `phase_recognition_cholec80.py` | (별도 phase 스크립트 없음) |
| EndoVis 2017 | `instrument_localization_endovis17.py` | mask segmentation (별도) |
| EndoVis 2018 VQA | `tissue_instrument_recognition_endovis18.py` | — |
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
5f. **PEFT / `No module named peft`** — `HF_PYTHON` env에 `pip install peft`  
5g. **PEFT load failed / adapter not found** — `MODEL_ID`가 어댑터 디렉터리인지 확인 (`adapter_config.json` 존재). Hub는 `org/repo/subfolder` 형식 (예: `khtks/Qwen3-VL/surgsigma_qwen3vl_full`)  
5h. **PEFT OOM** — 베이스 4B+LoRA도 GPU 필요; `HF_PEFT_MERGE=1`은 merge 시 추가 메모리 사용 가능  
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
14. **EndoVis 18 VQA 이미지 없음** — `eval/endovis2018/val/image/seq_N_frameIDX.bmp` 확인; `--image-split both`로 train fallback; seq 11–16은 bmp 없음
15. **EndoVis 18 VQA 샘플 0개** — `--vqa-root`, `--images-root`, `--seq` 확인
16. **EndoVis 18 VQA 호출 수** — val 전체 ~1,396회; API 비용·rate limit 주의 (`--max-samples`, `--seq`로 스모크)

```bash
python triplet_recognition_cholect50.py --help
python phase_recognition_cholec80.py --help
python instrument_localization_endovis17.py --help
python tissue_instrument_recognition_endovis18.py --help
python cvs_evaluation_endoscapes.py --help
bash grounding_task.sh help
```
