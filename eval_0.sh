# endovis2017

BACKEND=cosmos-32b DEVICE_VISIBLE=0 bash grounding_task.sh instrument_localization_endovis17 --bbox-mode filtered_union --viz
BACKEND=internvl3.5 DEVICE_VISIBLE=0 bash grounding_task.sh instrument_localization_endovis17 --bbox-mode filtered_union --viz
BACKEND=paligemma2 DEVICE_VISIBLE=0 bash grounding_task.sh instrument_localization_endovis17 --bbox-mode filtered_union --viz
BACKEND=qwen3-4b DEVICE_VISIBLE=0 bash grounding_task.sh instrument_localization_endovis17 --bbox-mode filtered_union --viz
BACKEND=qwen3-32b DEVICE_VISIBLE=0 bash grounding_task.sh instrument_localization_endovis17 --bbox-mode filtered_union --viz
BACKEND=cosmos-2b DEVICE_VISIBLE=0 bash grounding_task.sh instrument_localization_endovis17 --bbox-mode filtered_union --viz

BACKEND=qwen3-32b DEVICE_VISIBLE=0 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-32b-surgsigma-lora-ft-v1 \
    bash grounding_task.sh instrument_localization_endovis17 --bbox-mode filtered_union --viz

BACKEND=qwen3-32b DEVICE_VISIBLE=0 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-32b-augmented-lora-ft-v0.1/checkpoint-53000 \
    bash grounding_task.sh instrument_localization_endovis17 --bbox-mode filtered_union --viz

BACKEND=qwen3-4b DEVICE_VISIBLE=0 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-surgsigma-lora-ft-v1 \
    bash grounding_task.sh instrument_localization_endovis17 --bbox-mode filtered_union --viz

BACKEND=qwen3-4b DEVICE_VISIBLE=0 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-augmented-lora-ft-v1 \
    bash grounding_task.sh instrument_localization_endovis17 --bbox-mode filtered_union --viz

BACKEND=qwen3-4b DEVICE_VISIBLE=0 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-pretrained-dense-vision-lora-ft-v1 \
    bash grounding_task.sh instrument_localization_endovis17 --bbox-mode filtered_union --viz

BACKEND=qwen3-4b DEVICE_VISIBLE=0 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-surgsigma-dense-vision-lora-ft-v1 \
    bash grounding_task.sh instrument_localization_endovis17 --bbox-mode filtered_union --viz

BACKEND=qwen3-4b DEVICE_VISIBLE=0 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-augmented-dense-vision-lora-ft-v1 \
    bash grounding_task.sh instrument_localization_endovis17 --bbox-mode filtered_union --viz

# endovis2018
BACKEND=qwen3-32b DEVICE_VISIBLE=0 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-32b-surgsigma-lora-ft-v1 \
    bash grounding_task.sh tissue_instrument_recognition_endovis18 --bbox-mode filtered_union --viz

# BACKEND=qwen3-32b DEVICE_VISIBLE=0 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-32b-augmented-lora-ft-v0.1/checkpoint-53000 \
#     bash grounding_task.sh tissue_instrument_recognition_endovis18

BACKEND=qwen3-4b DEVICE_VISIBLE=0 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-surgsigma-lora-ft-v1 \
    bash grounding_task.sh tissue_instrument_recognition_endovis18

BACKEND=qwen3-4b DEVICE_VISIBLE=0 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-augmented-lora-ft-v1 \
    bash grounding_task.sh tissue_instrument_recognition_endovis18

BACKEND=qwen3-4b DEVICE_VISIBLE=0 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-pretrained-dense-vision-lora-ft-v1/qwen_lora \
    bash grounding_task.sh tissue_instrument_recognition_endovis18

BACKEND=qwen3-4b DEVICE_VISIBLE=0 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-surgsigma-dense-vision-lora-ft-v1/qwen_lora \
    bash grounding_task.sh tissue_instrument_recognition_endovis18

BACKEND=qwen3-4b DEVICE_VISIBLE=0 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-augmented-dense-vision-lora-ft-v1/qwen_lora \
    bash grounding_task.sh tissue_instrument_recognition_endovis18