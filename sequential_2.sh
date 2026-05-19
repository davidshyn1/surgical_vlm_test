BACKEND=cosmos-2b DEVICE_VISIBLE=1 bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol sequential_gt --prompt-mode mcq --eval-all
BACKEND=cosmos-32b DEVICE_VISIBLE=1 bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol sequential_gt --prompt-mode mcq --eval-all
BACKEND=internvl3.5 DEVICE_VISIBLE=1 bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol sequential_gt --prompt-mode mcq --eval-all
BACKEND=paligemma2 DEVICE_VISIBLE=1 bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol sequential_gt --prompt-mode mcq --eval-all
BACKEND=qwen2.5 DEVICE_VISIBLE=1 bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol sequential_gt --prompt-mode mcq --eval-all
BACKEND=qwen3-4b DEVICE_VISIBLE=1 bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol sequential_gt --prompt-mode mcq --eval-all
BACKEND=qwen3-32b DEVICE_VISIBLE=1 bash grounding_task.sh triplet_recognition_cholect50 --eval-protocol sequential_gt --prompt-mode mcq --eval-all