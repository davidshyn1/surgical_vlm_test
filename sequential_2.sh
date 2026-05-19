# BACKEND=cosmos-2b DEVICE_VISIBLE=1 bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol sequential_gt --prompt-mode mcq --eval-all
# BACKEND=cosmos-32b DEVICE_VISIBLE=1 bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol sequential_gt --prompt-mode mcq --eval-all
# BACKEND=internvl3.5 DEVICE_VISIBLE=1 bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol sequential_gt --prompt-mode mcq --eval-all
# BACKEND=paligemma2 DEVICE_VISIBLE=1 bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol sequential_gt --prompt-mode mcq --eval-all
# BACKEND=qwen2.5 DEVICE_VISIBLE=1 bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol sequential_gt --prompt-mode mcq --eval-all
# BACKEND=qwen3-4b DEVICE_VISIBLE=1 bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol sequential_gt --prompt-mode mcq --eval-all
# BACKEND=qwen3-32b DEVICE_VISIBLE=1 bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol sequential_gt --prompt-mode mcq --eval-all

# BACKEND=cosmos-32b DEVICE_VISIBLE=1 bash grounding_task.sh  tissue_instrument_recognition_endovis18
# BACKEND=internvl3.5 DEVICE_VISIBLE=1 bash grounding_task.sh  tissue_instrument_recognition_endovis18
# BACKEND=paligemma2 DEVICE_VISIBLE=1 bash grounding_task.sh  tissue_instrument_recognition_endovis18
# BACKEND=qwen2.5 DEVICE_VISIBLE=1 bash grounding_task.sh  tissue_instrument_recognition_endovis18
# BACKEND=qwen3-4b DEVICE_VISIBLE=1 bash grounding_task.sh  tissue_instrument_recognition_endovis18
# BACKEND=qwen3-32b DEVICE_VISIBLE=1 bash grounding_task.sh  tissue_instrument_recognition_endovis18
# BACKEND=cosmos-2b DEVICE_VISIBLE=1 bash grounding_task.sh  tissue_instrument_recognition_endovis18
# BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=khtks/Qwen3-VL/surgsigma_qwen3vl_full   bash grounding_task.sh tissue_instrument_recognition_endovis18

# BACKEND=cosmos-32b DEVICE_VISIBLE=1 bash grounding_task.sh language_grounding_surgical_prompts --force
# BACKEND=internvl3.5 DEVICE_VISIBLE=1 bash grounding_task.sh language_grounding_surgical_prompts --force
# BACKEND=paligemma2 DEVICE_VISIBLE=1 bash grounding_task.sh language_grounding_surgical_prompts --force
# BACKEND=qwen3-4b DEVICE_VISIBLE=1 bash grounding_task.sh language_grounding_surgical_prompts --force
# BACKEND=qwen3-32b DEVICE_VISIBLE=1 bash grounding_task.sh language_grounding_surgical_prompts --force
# BACKEND=cosmos-2b DEVICE_VISIBLE=1 bash grounding_task.sh language_grounding_surgical_prompts --force
BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=khtks/Qwen3-VL/surgsigma_qwen3vl_full bash grounding_task.sh language_grounding_surgical_prompts --max-new-tokens 128

# BACKEND=gemini MODEL_ID=gemini-2.5-flash bash grounding_task.sh language_grounding_surgical_prompts --max-new-tokens 128
# BACKEND=gpt MODEL_ID=gpt-4o bash grounding_task.sh language_grounding_surgical_prompts