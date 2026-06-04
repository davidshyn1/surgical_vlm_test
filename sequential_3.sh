

BACKEND=cosmos-32b DEVICE_VISIBLE=3 \
  bash grounding_task.sh visual_cross_attention_endovis2017 \
--feature-backbone hf \
--query-from-gt-crop \
--dataset-root /NHNHOME/WORKSPACE/26msit001_T_B/KAIST-AIPRLab/surgical/eval/endovis2017 \
--eval-all

BACKEND=internvl3.5 DEVICE_VISIBLE=3 \
  bash grounding_task.sh visual_cross_attention_endovis2017  \
--feature-backbone hf \
--query-from-gt-crop \
--dataset-root /NHNHOME/WORKSPACE/26msit001_T_B/KAIST-AIPRLab/surgical/eval/endovis2017 \
--eval-all

BACKEND=paligemma2 DEVICE_VISIBLE=3 \
  bash grounding_task.sh visual_cross_attention_endovis2017  \
--feature-backbone hf \
--query-from-gt-crop \
--dataset-root /NHNHOME/WORKSPACE/26msit001_T_B/KAIST-AIPRLab/surgical/eval/endovis2017 \
--eval-all

BACKEND=qwen3-4b DEVICE_VISIBLE=3 \
  bash grounding_task.sh visual_cross_attention_endovis2017  \
--feature-backbone hf \
--query-from-gt-crop \
--dataset-root /NHNHOME/WORKSPACE/26msit001_T_B/KAIST-AIPRLab/surgical/eval/endovis2017 \
--eval-all

BACKEND=qwen3-32b DEVICE_VISIBLE=3 \
  bash grounding_task.sh visual_cross_attention_endovis2017  \
--feature-backbone hf \
--query-from-gt-crop \
--dataset-root /NHNHOME/WORKSPACE/26msit001_T_B/KAIST-AIPRLab/surgical/eval/endovis2017 \
--eval-all

BACKEND=qwen3-32b DEVICE_VISIBLE=3 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-32b-surgsigma-lora-ft-v1 \
  bash grounding_task.sh visual_cross_attention_endovis2017  \
--feature-backbone hf \
--query-from-gt-crop \
--dataset-root /NHNHOME/WORKSPACE/26msit001_T_B/KAIST-AIPRLab/surgical/eval/endovis2017 \
--eval-all

BACKEND=qwen3-32b DEVICE_VISIBLE=3 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-32b-augmented-lora-ft-v0.1/checkpoint-53000 \
  bash grounding_task.sh visual_cross_attention_endovis2017  \
--feature-backbone hf \
--query-from-gt-crop \
--dataset-root /NHNHOME/WORKSPACE/26msit001_T_B/KAIST-AIPRLab/surgical/eval/endovis2017 \
--eval-all

BACKEND=qwen3-4b DEVICE_VISIBLE=3 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-surgsigma-lora-ft-v1 \
  bash grounding_task.sh visual_cross_attention_endovis2017  \
--feature-backbone hf \
--query-from-gt-crop \
--dataset-root /NHNHOME/WORKSPACE/26msit001_T_B/KAIST-AIPRLab/surgical/eval/endovis2017 \
--eval-all

BACKEND=qwen3-4b DEVICE_VISIBLE=3 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-augmented-lora-ft-v1 \
  bash grounding_task.sh visual_cross_attention_endovis2017  \
--feature-backbone hf \
--query-from-gt-crop \
--dataset-root /NHNHOME/WORKSPACE/26msit001_T_B/KAIST-AIPRLab/surgical/eval/endovis2017 \
--eval-all

BACKEND=qwen3-4b DEVICE_VISIBLE=3 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-pretrained-dense-vision-lora-ft-v1 \
  bash grounding_task.sh visual_cross_attention_endovis2017  \
--feature-backbone hf \
--query-from-gt-crop \
--dataset-root /NHNHOME/WORKSPACE/26msit001_T_B/KAIST-AIPRLab/surgical/eval/endovis2017 \
--eval-all

BACKEND=qwen3-4b DEVICE_VISIBLE=3 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-surgsigma-dense-vision-lora-ft-v1 \
  bash grounding_task.sh visual_cross_attention_endovis2017  \
--feature-backbone hf \
--query-from-gt-crop \
--dataset-root /NHNHOME/WORKSPACE/26msit001_T_B/KAIST-AIPRLab/surgical/eval/endovis2017 \
--eval-all

BACKEND=qwen3-4b DEVICE_VISIBLE=3 MODEL_ID=SurgVLA-Foundry/Qwen3-VL/qwen3vl-4b-augmented-dense-vision-lora-ft-v1 \
  bash grounding_task.sh visual_cross_attention_endovis2017  \
--feature-backbone hf \
--query-from-gt-crop \
--dataset-root /NHNHOME/WORKSPACE/26msit001_T_B/KAIST-AIPRLab/surgical/eval/endovis2017 \
--eval-all