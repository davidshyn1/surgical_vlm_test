# surgical_vlm_test

CholecT50 **triplet recognition** · Cholec80 **phase recognition** · EndoVis-17 **instrument localization** 벤치를 위한 독립 패키지입니다.  
`surgical_vlm_grounding`과 분리되어 있으며, CholecT50/80은 분류·인식 중심이고 EndoVis-17만 bbox 출력·시각화를 사용합니다.

백엔드: `prismatic` · `cosmos` · `groot` (`backends.py`)

---

## 1. 구성

| 파일 | 역할 |
|------|------|
| `triplet_recognition_cholect50.py` | CholecT50 triplet 평가 |
| `phase_recognition_cholec80.py` | Cholec80 phase 평가 |
| `instrument_localization_endovis17.py` | EndoVis-17-VQLA instrument bbox localization |
| `endovis17_data.py` | EndoVis-17 `vqla/*.txt` 샘플·프롬프트 로딩 |
| `cholect50_data.py` | challenge-val 라벨·프레임 로딩 |
| `cholec80_data.py` | Cholec80 phase 라벨·비디오 프레임 로딩 |
| `utils.py` | 라벨 파싱, bbox/IoU/mAP, resume, 시각화 |
| `backends.py` | VLM 로드·추론 |
| `grounding_task.sh` | 실행 런처 (`uv` + 백엔드 venv) |
| `setup_backend.sh` | 백엔드 `.venv` 설치 (uv) |
| `scripts/extract_cholec80_frames.sh` | Cholec80 프레임 추출 (ffmpeg, 기본 video41–80) |

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
- **데이터**: `../data/Cholec80` (없으면 `../data/cholec80` 자동 탐색)
- **기본 split**: `eval` = **video41–video80** (EndoNet evaluation set)
- **지표**: Accuracy, 클래스별 Recall / Precision / Jaccard, macro 평균

비디오 MP4에서 프레임을 읽습니다. **OpenCV는 필수가 아닙니다** (numpy pin 환경 권장 경로 아래 참고).  
Cholec80은 **전 비디오 25 fps**이며, phase annotation은 **매 프레임**입니다.  
기본 eval은 **0.1 fps** (stride **250**: 0, 250, 500, …)이며, `frames_0p1fps/`에 PNG + `videoNN-phase.txt` manifest를 둡니다.

| 설정 | 샘플링 | 용도 |
|------|--------|------|
| 기본 (`frames_0p1fps` manifest) | **0.1 fps** | 권장 eval |
| `--frame-stride 250` (MP4 직접) | **0.1 fps** | manifest 없이 동일 간격 |
| `--frame-stride 1` | **25 fps** (전 annotated frame) | 매우 느림 |

7개 phase: Preparation, Calot Triangle Dissection, Clipping and Cutting, Gallbladder Dissection, Gallbladder Packaging, Cleaning and Coagulation, Gallbladder Retraction.

실행 예시는 **§4.1 Cholec80 phase**를 참고하세요.

### EndoVis-17 Instrument Localization

**Instrument localization** — EndoVis-17-VQLA `Where is {instrument} located?` 질의에 대해 bbox를 예측합니다.

- **데이터**: `<surgical repo>/eval/EndoVis-17-VQLA` (`left_frames/`, `vqla/*.txt`)
- **이미지**: 1280×1024 JPEG (원본). VLM 입력은 backend `pil_side`로 square resize (예: 384×384)
- **프롬프트** (instrument 이름이 샘플마다 치환됨):

```
Where is the Large Needle Driver located? Answer the question with just a bounding box.
Format: [x_min, y_min, x_max, y_max]
Use normalized coordinates in [0, 1] relative to the image you see.
If the Large Needle Driver is not in the image, answer exactly: not present
```

- **GT bbox**: annotation 픽셀 좌표 → 원본 W/H 기준 normalized xyxy 저장
- **Cosmos**: 모델 출력이 0–1000 스케일이면 파서에서 **÷1000** 후 [0,1]로 metric 계산
- **지표** (`metrics` in JSON): **mIoU**, **mAP@50**, **mAP@75**, **COCO AP** (IoU 0.5:0.05:0.95)
- **시각화** (기본 `--viz`): `visualizations/gt/`, `pred/`, `comparison/` — 박스 위·좌상단에 **instrument 이름만** 표시

5종 instrument: Bipolar Forceps, Large Needle Driver, Monopolar Curved Scissors, Prograsp Forceps, Ultrasound Probe.  
전체 localization 쿼리 **236개** (`Where is … located?` 행만 사용).

실행 예시는 **§4.1 EndoVis-17 localization**을 참고하세요.

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

**EndoVis-17 (localization)**

- 루트: `<surgical repo>/eval/EndoVis-17-VQLA` (`ENDOVIS17_VQLA_ROOT`)
- 프레임: `left_frames/{seq}_frame{NNN}.jpg`
- 질의·GT: `vqla/{stem}.txt` (`question|region|xmin,ymin,xmax,ymax`)

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
2. **`--frames-root`** — 미리 뽑아 둔 PNG/JPG (`video41/000250.png`), PIL만 사용  
3. **OpenCV** (`--frame-reader opencv`) — venv에 이미 `cv2`가 있을 때만.  
   `opencv-python-headless`를 새로 깔면 **numpy 버전 충돌**이 날 수 있음 → 피하는 것을 권장

```bash
# (선택) eval 41–80, 0.1 fps 프레임 + phase manifest 추출 — 한 번만 실행
CHOLEC80_ROOT=../data/cholec80 bash scripts/extract_cholec80_frames.sh

# manifest만 생성
python3 scripts/build_cholec80_eval_phase_annotations.py --dataset-root ../data/cholec80

# 추출본으로 eval (가장 안전)
CHOLEC80_FRAMES_ROOT=../data/cholec80/frames_0p1fps \
  bash grounding_task.sh phase_recognition_cholec80

# 또는 ffmpeg로 MP4에서 바로 (추출 없이, stride 250 ≈ 0.1 fps)
bash grounding_task.sh phase_recognition_cholec80 --frame-stride 250
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

**Joint — 전체 평가** (기본, MCQ):

```bash
export CHOLECT50_VIDEOS_ROOT=/path/to/CholecT50/videos

BACKEND=prismatic DEVICE_VISIBLE=0 \
  bash grounding_task.sh triplet_recognition_cholect50 \
    --eval-protocol joint --prompt-mode mcq
```

**Sequential — GT context** (annotation당 VLM 3회):

```bash
bash grounding_task.sh triplet_recognition_cholect50 \
  --eval-protocol sequential_gt --prompt-mode mcq
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

**Cosmos 예**:

```bash
BACKEND=cosmos MODEL_ID=nvidia/Cosmos-Reason2-2B DEVICE_VISIBLE=0 \
  bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol joint --prompt-mode mcq
```

#### Cholec80 phase

`grounding_task.sh`가 기본으로 `--dataset-root $CHOLEC80_ROOT`, `--split eval`(video41–80)을 넣습니다.

**스모크 테스트** (video 41, sparse frames):

```bash
BACKEND=prismatic DEVICE_VISIBLE=0 \
  bash grounding_task.sh phase_recognition_cholec80 \
    --video 41 --frame-stride 250 --max-frames-per-video 5
```

**Eval 41–80, 0.1 fps** (기본 `frames_0p1fps`):

```bash
BACKEND=prismatic DEVICE_VISIBLE=0 \
  bash grounding_task.sh phase_recognition_cholec80
```

**전체 annotated frame (25 fps)**:

```bash
bash grounding_task.sh phase_recognition_cholec80 --frame-stride 1
```

**Train split (video01–40)**:

```bash
bash grounding_task.sh phase_recognition_cholec80 --split train
```

#### EndoVis-17 localization

`grounding_task.sh`가 기본으로 `--dataset-root`, `--frames-root`, `--annotations-root`를 `../eval/EndoVis-17-VQLA` 하위로 넣습니다.

**스모크 테스트** (5 samples):

```bash
BACKEND=prismatic DEVICE_VISIBLE=0 \
  bash grounding_task.sh instrument_localization_endovis17 --max-samples 5
```

**전체 236 queries**:

```bash
BACKEND=prismatic DEVICE_VISIBLE=0 \
  bash grounding_task.sh instrument_localization_endovis17
```

**Cosmos** (bbox ÷1000 파싱):

```bash
BACKEND=cosmos MODEL_ID=nvidia/Cosmos-Reason2-2B DEVICE_VISIBLE=0 \
  bash grounding_task.sh instrument_localization_endovis17
```

**시각화만 재생성** (기존 JSON 필요):

```bash
bash grounding_task.sh instrument_localization_endovis17 \
  --viz-only --force \
  --output outputs/instrument_localization_endovis17/loc_prismatic_original/endovis17_instrument_localization.json
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
  --frames-root ../data/cholec80/frames_0p1fps
```

**EndoVis-17**:

```bash
uv run --python ../backend/prismatic-vlms/.venv/bin/python \
  instrument_localization_endovis17.py \
  --backend prismatic \
  --dataset-root ../eval/EndoVis-17-VQLA
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
| `--frame-stride N` | native phase에서 N프레임마다 1장 (manifest 없을 때 기본 `250` ≈ 0.1 fps) |
| `--frames-root` | 추출 프레임 루트 (`video41/000123.png`) |
| `--frame-reader {auto,ffmpeg,opencv}` | MP4 디코드 방식 (기본 `auto` = ffmpeg 우선) |
| `--max-frames-per-video K` | 비디오당 최대 K장 (디버그용) |
| `--video 41` | 단일 비디오 (`41`, `video41` 모두 가능) |
| `--dataset-root` | Cholec80 루트 (기본: `../data/Cholec80`, `cholec80` 폴백) |
| `--force` / `--output` | triplet과 동일 |

**EndoVis-17 (`instrument_localization_endovis17.py`)**

| 인자 | 설명 |
|------|------|
| `--dataset-root` | EndoVis-17-VQLA 루트 (기본: `../eval/EndoVis-17-VQLA`) |
| `--frames-root` | `left_frames/` (기본: dataset-root/left_frames) |
| `--annotations-root` | `vqla/` (기본: dataset-root/vqla) |
| `--instrument`, `--region`, `--frame` | 필터 (instrument id, region id, frame stem) |
| `--max-samples N` | 랜덤 subsample (디버그; 생략 시 236개 전체) |
| `--viz` / `--no-viz` | GT/Pred/comparison JPEG (기본: viz 켜짐) |
| `--viz-only` | VLM 생략, 기존 JSON에서 시각화만 생성 (`--output` 필수) |
| `--force` / `--output` | resume·재추론·결과 경로 |

**공통**: `--backend`, `--model-id`, `--device`, `--hf-token`, `--max-new-tokens`

### 4.4 기본 출력 경로

| 태스크 | 기본 JSON 경로 |
|--------|----------------|
| Triplet | `outputs/triplet_recognition_cholect50/triplet_{backend}_{model}_{mcq\|ov}_{joint\|sequential_*}/cholect50_challenge_val_triplet.json` |
| Phase | `outputs/phase_recognition_cholec80/phase_{backend}_{model}_{split}/cholec80_phase_stride{N}.json` |
| EndoVis-17 | `outputs/instrument_localization_endovis17/loc_{backend}_{model}/endovis17_instrument_localization.json` |

EndoVis-17 시각화 (같은 폴더):

```
loc_{backend}_{model}/
  endovis17_instrument_localization.json
  visualizations/
    gt/          {frame}_{instrument}_{region}_gt.jpg
    pred/        ..._pred.jpg
    comparison/  ..._gt_pred.jpg   # GT=green, Pred=red, IoU
```

JSON `results[]` 항목 요약:

- `input.label_context`: GT bbox (`label_bbox_xyxy_px`, `label_bbox_xyxy_norm`), instrument/region
- `output.parsed`: `bbox_xyxy_norm`, `bbox_xyxy_px`, `not_present`, `raw` model text
- `evaluation`: `iou`, `gt_bbox_norm`, `pred_bbox_norm`
- `visualization_*_path`: 생성된 JPEG 경로

JSON `metrics` (EndoVis-17): `mIoU`, `mAP@50`, `mAP@75`, `COCO_AP`, `per_class_ap`, `n_parsed_bbox`, `n_not_present`.

JSON `output` (triplet `sequential_*`): `parsed.triplets` + `sequential_steps` (단계별 prompt/text/parsed).  
JSON `metrics` (phase): `accuracy`, `macro_recall`, `macro_precision`, `macro_jaccard`, `per_class`.

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
| `ENDOVIS17_VQLA_ROOT` | localization `--dataset-root` (기본: `../eval/EndoVis-17-VQLA`) |
| `PRISMATIC_PYTHON` / `COSMOS_PYTHON` / `GROOT_PYTHON` | venv python 경로 override |
| `GROUNDING_TASK_AUTO_BACKEND_SETUP` | `0`이면 uv 자동 설치 스킵 |

---

## 6. `surgical_vlm_grounding`과의 차이

| | **surgical_vlm_test** | **surgical_vlm_grounding** |
|---|------------------------|------------------------------|
| 목적 | CholecT50 triplet · Cholec80 phase · EndoVis-17 bbox eval | Localization, language/visual grounding 등 |
| CholecT50 | `triplet_recognition_cholect50.py` | `localization_cholect50.py`, `language_grounding_v*`, … |
| Cholec80 | `phase_recognition_cholec80.py` | (별도 phase 스크립트 없음) |
| EndoVis-17 | `instrument_localization_endovis17.py` | (동일 벤치, 다른 패키지) |
| Bbox | EndoVis-17만 (normalized xyxy + viz) | CholecT50 localization 등 |
| 입력 | T50: 추출 프레임 / C80: MP4·추출 PNG | 주로 추출 프레임 |
| Triplet | `joint` 1질문/프레임 또는 `sequential_*` 3질문/annotation | multi-step localization |
| Phase | 7-class MCQ (A–G) | localization + phase |
| 추론 | task·프로토콜별 (T50 joint=1회/프레임) | task별 상이 |

---

## 7. 문제 해결

1. **CholecT50 프레임 이미지 없음** — `CHOLECT50_VIDEOS_ROOT` 또는 `--videos-root` 확인  
2. **Cholec80 데이터 없음** — `CHOLEC80_ROOT` 또는 `--dataset-root`에 `phase_annotations/`, `videos/` 있는지 확인 (`Cholec80` vs `cholec80` 대소문자 폴백 지원)  
3. **프레임 로드 실패** — `ffmpeg -version` 확인, 또는 `scripts/extract_cholec80_frames.sh` 후 `CHOLEC80_FRAMES_ROOT` 사용 (`opencv` 설치보다 권장)  
4. **백엔드 import 실패** — `bash setup_backend.sh <backend>`  
5. **HF 401** — `.hf_token` 경로 및 권한  
6. **resume** — 동일 `--output` JSON에 이어서 실행; 전체 재실행은 `--force`  
7. **triplet sequential이 느림** — annotation당 VLM 3회; 빠른 테스트는 `--eval-protocol joint` 또는 `--samples-only`  
8. **phase eval이 너무 느림** — 기본 `frames_0p1fps`(0.1 fps); `--frame-stride 1`은 eval 40 videos 기준 약 98k+ calls  
9. **프롬프트/프로토콜 변경 후** — 이전 JSON과 혼동 방지를 위해 `--force` 권장  
10. **EndoVis-17 데이터 없음** — `eval/EndoVis-17-VQLA/left_frames`, `vqla/` 확인  
11. **Cosmos bbox 이상** — 0–1000 출력은 자동 ÷1000; [0,1]로 직접 내면 그대로 사용  
12. **EndoVis viz만 다시** — `--viz-only --output <기존 json>` (`--force`로 JPEG 덮어쓰기)

```bash
python triplet_recognition_cholect50.py --help
python phase_recognition_cholec80.py --help
python instrument_localization_endovis17.py --help
bash grounding_task.sh help
```
