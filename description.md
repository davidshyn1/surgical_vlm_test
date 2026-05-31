# surgical_vlm_test — 태스크 요약

CholecT50 · Cholec80 · EndoVis 2017/2018 · Endoscapes · SAR-RARP50 벤치마크를 **하나의 VLM 백엔드**로 돌리는 평가 패키지입니다.  
실행 진입점은 `grounding_task.sh`이며, 각 태스크는 별도 Python 스크립트가 담당합니다.

```bash
BACKEND=<backend> bash grounding_task.sh <task_name> [추가 인자...]
```

- **이미지 입력**: triplet · phase · localization · EndoVis18 VQA · CVS · SAR-RARP50 action  
- **텍스트만**: `language_grounding_surgical_prompts` (이미지 없음)  
- **백엔드**: `prismatic`, HF 계열(`qwen3-*`, `cosmos-*`, `internvl3.5`, `paligemma2`, …), API(`gpt`, `gemini`, `claude`)  
- **결과**: `outputs/<task>/.../*.json`에 프롬프트·예측·채점 저장

---

## 1. CholecT50 Triplet Recognition

**스크립트**: `triplet_recognition_cholect50.py`

| 항목 | 내용 |
|------|------|
| **데이터** | `eval/cholect50-challenge-val` 라벨 + `CHOLECT50_VIDEOS_ROOT` 프레임 이미지 |
| **단위** | 프레임당 triplet (instrument, verb, target) — annotation마다 질문 방식이 다를 수 있음 |

**프롬프트**

- **`joint` (기본)**: 한 장의 이미지에 대해 *"What tasks are the instruments accomplishing…"* 한 번 질문. MCQ면 instrument/verb/target **옵션 목록(쉼표 구분)** 포함.
- **`sequential_gt` / `sequential_pred`**: instrument → verb → target 순으로 **3번** 질문. verb/target 문맥은 GT 또는 이전 단계 **모델 예측**.
- **`--prompt-mode`**: `mcq`(옵션 목록) / `ov`(open vocabulary)

**기대 답변**

- `joint`: `<grasper, grasp, gallbladder>` 형태, 줄마다 하나의 triplet (여러 줄 가능)
- `sequential_*`: 각 단계에서 해당 카테고리 이름만 (예: `grasper`, `dissect`)

**지표**

- Component Accuracy (instrument / verb / target 각각)
- Triplet Accuracy (세 컴포넌트 모두 일치)
- Component별 **mAP**

---

## 2. Cholec80 Phase Recognition

**스크립트**: `phase_recognition_cholec80.py`

| 항목 | 내용 |
|------|------|
| **데이터** | `eval/cholec80/frames_0p1fps` (video41–80, 0.1 fps PNG + phase manifest). 원본 MP4는 `data/cholec80` |
| **단위** | 프레임 1장 → 수술 **phase** 1개 (7단계, A–G) |

**프롬프트**

- *"In the Cholecystectomy surgical image, what is the current Phase?"*
- 7개 phase를 **문자 옵션(A–G) + 이름**으로 나열 (MCQ)

**기대 답변**

- phase 키워드 1개 (예: `gallbladder-dissection`, 또는 `D`)

**지표**

- **Accuracy**
- 클래스별 Recall / Precision / Jaccard
- **macro** Recall / Precision / Jaccard

---

## 3. EndoVis 2017 Instrument Localization

**스크립트**: `instrument_localization_endovis17.py` · `endovis17_data.py`

| 항목 | 내용 |
|------|------|
| **데이터** | `eval/endovis2017` — val split mask (`label/`) → **instrument 클래스별 tight bbox** GT |
| **단위** | (프레임, instrument 종류) 샘플 1개 — 이미지 1장 + 해당 도구 1개 위치 질문 |

**프롬프트**

- *"Where is the {Instrument Name} located? … Format: [x_min, y_min, x_max, y_max]"*
- 좌표는 **이미지 기준 [0,1] 정규화**
- 없으면 `not present`

**기대 답변**

- `[0.12, 0.34, 0.56, 0.78]` 또는 `not present`
- LoRA/파인튜닝 모델은 `<answer>…</answer>` 태그가 붙을 수 있음 → 파서가 태그 제거 후 파싱

**지표**

- **mIoU** (파싱된 bbox vs mask GT)
- **mAP@50**, **mAP@75**, **COCO AP** (detection 스타일)

---

## 4. EndoVis 2018 VQA — Tissue / Instrument Recognition

**스크립트**: `tissue_instrument_recognition_endovis18.py` · `endovis18_vqa_data.py`

| 항목 | 내용 |
|------|------|
| **데이터** | `eval/EndoVis-18-VQA` (Classification QA txt) + `eval/endovis2018` (bmp 이미지) |
| **단위** | QA 1쌍 (organ / instrument state·location / multi-tool 등) |

**프롬프트**

- VQA 파일의 **질문 문장** 그대로 사용
- **`mcq`**: 보기 목록을 쉼표로 붙임
- **`ov`**: 보기 없이 키워드만 답하라고 지시
- multi-select 문항은 쉼표 구분 복수 키워드

**기대 답변**

- 단일 MCQ: 키워드 1개 (예: `kidney`, `left-top`)
- 도구 집합 문항: `grasper, hook` 등

**지표**

- **Tissue accuracy** (organ Q1)
- **Instrument accuracy** (state/location Q)
- **Overall accuracy** (단일 선택 문항 합산)
- **Tools AUROC** (프레임별 multi-label instrument set)
- **Tools exact set accuracy** (집합 완전 일치)

---

## 5. Endoscapes CVS Evaluation

**스크립트**: `cvs_evaluation_endoscapes.py` · `endoscapes_cvs_data.py`

| 항목 | 내용 |
|------|------|
| **데이터** | `eval/endoscapes` — COCO `annotation_coco.json`, 이미지별 **CVS 3기준** GT (`ds` 점수, threshold 0.5 → yes/no) |
| **단위** | 프레임 1장, Critical View of Safety **3 criteria** |

**프롬프트**

- **`joint`**: 3개 기준 질문을 한 번에, `C1: yes` 형식으로 3줄 답 요청
- **`per_criterion`**: 기준마다 yes/no MCQ 1회 (VLM 3호출)

**기대 답변**

- `yes` / `no` (키워드만)

**지표**

- **Average accuracy** (3기준 합산)
- **Balanced accuracy** (기준별·전체)
- 기준별(C1–C3) accuracy / balanced accuracy

---

## 6. SAR-RARP50 Action Recognition

**스크립트**: `action_recognition_sarrarp50.py` · `sarrarp50_data.py`

| 항목 | 내용 |
|------|------|
| **데이터** | `eval/sarrarp50` — `video_XX/segmentation/*.png` 인덱스 + `action_discrete.txt` GT. **사전에** `scripts/extract_sarrarp50_frames.py`로 PNG 추출 필요 |
| **단위** | segmentation 프레임 1장 → **needle/suture action** 1클래스 (8-class MCQ) |

**프롬프트**

- *"What Action related to the needle and suture is the surgeon focusing on right now?"*
- **`mcq`**: 8개 action **A–H 옵션 목록**
- **`ov`**: 옵션 없음

**기대 답변**

- canonical action id 1개 (예: `a3` → display name)

**지표**

- **Accuracy**
- 클래스별 Recall / Precision / Jaccard
- **macro** Recall / Precision / Jaccard

---

## 7. CholecT50 Language Grounding (Surgical Prompts)

**스크립트**: `language_grounding_surgical_prompts.py`  
**데이터 JSON**: `eval/prompts/surgical_prompts.json` (148문항, triplet completion)

| 항목 | 내용 |
|------|------|
| **입력** | **이미지 없음** — phase + triplet 2필드만 텍스트로 주고 나머지 1필드 예측 |
| **서브타입** | `pvt_to_instrument` / `pit_to_verb` / `piv_to_target` |

**프롬프트** (`eval/prompts/build_surgical_prompt.py`)

- 영어 질문: *"During the '{phase}' phase … which instrument(s) / what action(s) / which anatomical structure(s) …?"*
- 예측 필드의 **전체 vocab을 MCQ 옵션**으로 나열 (instrument 6 · verb 9 · target 12)
- *"comma-separated list, at most 3 labels"* 지시

**기대 답변**

- 쉼표 구분 라벨 최대 3개 (예: `grasper` 또는 `bipolar, hook`)
- 파인튜닝 모델의 `<answer>…</answer>`는 채점 전 **태그 제거** (`strip_lora_answer_tags`)

**지표**

- Sample-averaged **multi-label F1** (≤3 term)
- **macro AUROC** · **macro mAP** (`instrument` / `verb` / `target` vocab + synthetic `others`)
- OOV 토큰 → `others=1` (GT others는 항상 0)

**백엔드**

- 로컬: `generate_text()` (텍스트만 forward)
- API: GPT / Gemini / Claude text endpoint

---

## 공통 사항

| 항목 | 설명 |
|------|------|
| **Resume** | 대부분 태스크는 기존 JSON에 유효한 예측이 있으면 스킵 (`--force`로 재추론) |
| **출력 경로** | `outputs/<task_name>/<backend>_<model>/…json` — 자세한 패턴은 `README.md` §4.4 |
| **LoRA (SurgSigma 등)** | `--model-id`에 어댑터 경로 지정 · vision 태스크와 language 태스크 모두 `<answer>` 태그 처리 가능 |
| **상세 문서** | 설치, 데이터 경로, CLI 전체 목록 → [`README.md`](README.md) |

---

## 태스크 ↔ `grounding_task.sh` 이름

| `grounding_task.sh` 인자 | 스크립트 |
|--------------------------|----------|
| `triplet_recognition_cholect50` | `triplet_recognition_cholect50.py` |
| `phase_recognition_cholec80` | `phase_recognition_cholec80.py` |
| `instrument_localization_endovis17` | `instrument_localization_endovis17.py` |
| `tissue_instrument_recognition_endovis18` | `tissue_instrument_recognition_endovis18.py` |
| `cvs_evaluation_endoscapes` | `cvs_evaluation_endoscapes.py` |
| `action_recognition_sarrarp50` | `action_recognition_sarrarp50.py` |
| `language_grounding_surgical_prompts` | `language_grounding_surgical_prompts.py` |
