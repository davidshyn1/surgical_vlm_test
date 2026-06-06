# BACKEND=qwen3-32b DEVICE_VISIBLE=1 \
#   MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-32b-surgsigma-lora-ft-v1 \
#   bash grounding_task.sh triplet_recognition_cholect50 \
#     --eval-protocol joint --prompt-mode mcq

BACKEND=qwen3-32b DEVICE_VISIBLE=1 \
  MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-32b-augmented-lora-ft-v0.1/checkpoint-53000 \
  bash grounding_task.sh triplet_recognition_cholect50 \
    --eval-protocol joint --prompt-mode mcq 

# BACKEND=qwen3-4b DEVICE_VISIBLE=1 \
#   MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-surgsigma-lora-ft-v1 \
#   bash grounding_task.sh triplet_recognition_cholect50 \
#     --eval-protocol joint --prompt-mode mcq 

# BACKEND=qwen3-4b DEVICE_VISIBLE=1 \
#   MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-augmented-lora-ft-v1 \
#   bash grounding_task.sh triplet_recognition_cholect50 \
#     --eval-protocol joint --prompt-mode mcq 