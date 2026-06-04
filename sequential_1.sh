# endoscapes
BACKEND=qwen3-32b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-32b-surgsigma-lora-ft-v1 \
  bash grounding_task.sh cvs_evaluation_endoscapes

BACKEND=qwen3-32b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-32b-augmented-lora-ft-v0.1/checkpoint-53000 \
  bash grounding_task.sh cvs_evaluation_endoscapes

BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-surgsigma-lora-ft-v1 \
  bash grounding_task.sh cvs_evaluation_endoscapes

BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-augmented-lora-ft-v1 \
  bash grounding_task.sh cvs_evaluation_endoscapes

BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-pretrained-dense-vision-lora-ft-v1 \
  bash grounding_task.sh cvs_evaluation_endoscapes

BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-surgsigma-dense-vision-lora-ft-v1 \
  bash grounding_task.sh cvs_evaluation_endoscapes

BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-augmented-dense-vision-lora-ft-v1 \
  bash grounding_task.sh cvs_evaluation_endoscapes


#RARP
BACKEND=qwen3-32b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-32b-surgsigma-lora-ft-v1 \
  bash grounding_task.sh action_recognition_sarrarp50

BACKEND=qwen3-32b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-32b-augmented-lora-ft-v0.1/checkpoint-53000 \
  bash grounding_task.sh action_recognition_sarrarp50

BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-surgsigma-lora-ft-v1 \
  bash grounding_task.sh action_recognition_sarrarp50

BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-augmented-lora-ft-v1 \
  bash grounding_task.sh action_recognition_sarrarp50

BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-pretrained-dense-vision-lora-ft-v1 \
  bash grounding_task.sh action_recognition_sarrarp50

BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-surgsigma-dense-vision-lora-ft-v1 \
  bash grounding_task.sh action_recognition_sarrarp50

BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-augmented-dense-vision-lora-ft-v1 \
  bash grounding_task.sh action_recognition_sarrarp50


#RARP Action Planning Language
# BACKEND=cosmos-32b DEVICE_VISIBLE=0 bash grounding_task.sh instrument_localization_endovis17 --viz
# BACKEND=internvl3.5 DEVICE_VISIBLE=0 bash grounding_task.sh instrument_localization_endovis17 --viz
# BACKEND=paligemma2 DEVICE_VISIBLE=0 bash grounding_task.sh instrument_localization_endovis17 --viz
# BACKEND=qwen2.5 DEVICE_VISIBLE=0 bash grounding_task.sh instrument_localization_endovis17 --viz
# BACKEND=qwen3-4b DEVICE_VISIBLE=0 bash grounding_task.sh instrument_localization_endovis17 --viz
# BACKEND=qwen3-32b DEVICE_VISIBLE=0 bash grounding_task.sh instrument_localization_endovis17 --viz
# BACKEND=cosmos-2b DEVICE_VISIBLE=0 bash grounding_task.sh instrument_localization_endovis17 --viz

##Language Grounding Sarrarp50 Next Action
BACKEND=cosmos-32b DEVICE_VISIBLE=1 bash grounding_task.sh language_grounding_sarrarp50_next_action --filter-template-id 0
BACKEND=internvl3.5 DEVICE_VISIBLE=1 bash grounding_task.sh language_grounding_sarrarp50_next_action --filter-template-id 0
BACKEND=paligemma2 DEVICE_VISIBLE=1 bash grounding_task.sh language_grounding_sarrarp50_next_action --filter-template-id 0
BACKEND=qwen3-4b DEVICE_VISIBLE=1 bash grounding_task.sh language_grounding_sarrarp50_next_action --filter-template-id 0
BACKEND=qwen3-32b DEVICE_VISIBLE=1 bash grounding_task.sh language_grounding_sarrarp50_next_action --filter-template-id 0
BACKEND=cosmos-2b DEVICE_VISIBLE=1 bash grounding_task.sh language_grounding_sarrarp50_next_action --filter-template-id 0
BACKEND=qwen3-32b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-32b-surgsigma-lora-ft-v1 \
  bash grounding_task.sh language_grounding_sarrarp50_next_action --filter-template-id 0
BACKEND=qwen3-32b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-32b-augmented-lora-ft-v0.1/checkpoint-53000 \
  bash grounding_task.sh language_grounding_sarrarp50_next_action --filter-template-id 0
BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-surgsigma-lora-ft-v1 \
  bash grounding_task.sh language_grounding_sarrarp50_next_action --filter-template-id 0
BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-augmented-lora-ft-v1 \
  bash grounding_task.sh language_grounding_sarrarp50_next_action --filter-template-id 0
BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-pretrained-dense-vision-lora-ft-v1 \
  bash grounding_task.sh language_grounding_sarrarp50_next_action --filter-template-id 0
BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-surgsigma-dense-vision-lora-ft-v1 \
  bash grounding_task.sh language_grounding_sarrarp50_next_action --filter-template-id 0
BACKEND=qwen3-4b DEVICE_VISIBLE=1 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-augmented-dense-vision-lora-ft-v1 \
  bash grounding_task.sh language_grounding_sarrarp50_next_action --filter-template-id 0