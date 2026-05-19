# BACKEND=gemini MODEL_ID=gemini-2.0-flash API_WORKERS=16  bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol joint --prompt-mode mcq --eval-all --max-new-tokens 32
# BACKEND=qwen3-4b DEVICE_VISIBLE=2 MODEL_ID=khtks/Qwen3-VL/surgsigma_qwen3vl_full   bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol sequential_gt --prompt-mode mcq --eval-all --max-new-tokens 32
# BACKEND=qwen3-4b DEVICE_VISIBLE=2 MODEL_ID=khtks/Qwen3-VL/surgsigma_qwen3vl_full   bash grounding_task.sh instrument_localization_endovis17 --viz
# BACKEND=qwen3-4b DEVICE_VISIBLE=2 MODEL_ID=khtks/Qwen3-VL/surgsigma_qwen3vl_full   bash grounding_task.sh phase_recognition_cholec80
# BACKEND=qwen3-4b DEVICE_VISIBLE=2 MODEL_ID=khtks/Qwen3-VL/surgsigma_qwen3vl_full   bash grounding_task.sh tissue_instrument_recognition_endovis18
# BACKEND=qwen3-4b DEVICE_VISIBLE=2 MODEL_ID=khtks/Qwen3-VL/surgsigma_qwen3vl_full   bash grounding_task.sh cvs_evaluation_endoscapes


BACKEND=cosmos-32b DEVICE_VISIBLE=2 bash grounding_task.sh  cvs_evaluation_endoscapes --force
BACKEND=internvl3.5 DEVICE_VISIBLE=2 bash grounding_task.sh  cvs_evaluation_endoscapes --force
BACKEND=paligemma2 DEVICE_VISIBLE=2 bash grounding_task.sh  cvs_evaluation_endoscapes --force
BACKEND=qwen2.5 DEVICE_VISIBLE=2 bash grounding_task.sh  cvs_evaluation_endoscapes --force
BACKEND=qwen3-4b DEVICE_VISIBLE=2 bash grounding_task.sh  cvs_evaluation_endoscapes --force
BACKEND=qwen3-32b DEVICE_VISIBLE=2 bash grounding_task.sh  cvs_evaluation_endoscapes --force
BACKEND=cosmos-2b DEVICE_VISIBLE=2 bash grounding_task.sh  cvs_evaluation_endoscapes --force