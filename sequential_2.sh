
BACKEND=qwen3-32b DEVICE_VISIBLE=2 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-32b-surgsigma-lora-ft-v1 \
    bash grounding_task.sh language_grounding_surgical_prompts --force
BACKEND=qwen3-32b DEVICE_VISIBLE=2 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-32b-augmented-lora-ft-v0.1/checkpoint-53000 \
    bash grounding_task.sh language_grounding_surgical_prompts --force
BACKEND=qwen3-4b DEVICE_VISIBLE=2 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-surgsigma-lora-ft-v1 \
    bash grounding_task.sh language_grounding_surgical_prompts --force
BACKEND=qwen3-4b DEVICE_VISIBLE=0 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-augmented-lora-ft-v1 \
    bash grounding_task.sh language_grounding_surgical_prompts --force
BACKEND=qwen3-4b DEVICE_VISIBLE=2 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-pretrained-dense-vision-lora-ft-v1 \
    bash grounding_task.sh language_grounding_surgical_prompts --force
BACKEND=qwen3-4b DEVICE_VISIBLE=2 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-surgsigma-dense-vision-lora-ft-v1 \
    bash grounding_task.sh language_grounding_surgical_prompts --force
BACKEND=qwen3-4b DEVICE_VISIBLE=2 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-augmented-dense-vision-lora-ft-v1 \
    bash grounding_task.sh language_grounding_surgical_prompts --force

#Language Grounding Sarrarp50 Next Action
BACKEND=gemini MODEL_ID=gemini-2.5-flash DEVICE_VISIBLE=2 bash grounding_task.sh language_grounding_sarrarp50_next_action --filter-template-id 0
BACKEND=gpt MODEL_ID=gpt-4o DEVICE_VISIBLE=2 bash grounding_task.sh language_grounding_sarrarp50_next_action --filter-template-id 0
