#!/bin/bash -e

python3 launch.py --config configs/dreamdpo/mvdream-sd21-reward.yaml \
    --train --gpu 0 --eval-imagereward \
    system.prompt_processor.prompt="A pair of hiking boots caked with mud at the doorstep of a cabin" \
    system.guidance.beta_dpo=0.01 system.guidance.reward_model="hpsv2-score" \
    tag="original-reward-only"

python3 launch.py --config configs/dreamdpo/mvdream-sd21-lmm.yaml \
    --train --gpu 0 --eval-imagereward \
    system.prompt_processor.prompt="A pair of hiking boots caked with mud at the doorstep of a cabin" \
    system.guidance.ai_start_iter=1200 system.guidance.ai_prob=0.1 \
    tag="original-qwen"

python3 launch.py --config configs/dreamdpo/mvdream-sd21-reward.yaml \
    --train --gpu 0 --eval-imagereward \
    system.prompt_processor.prompt="A pair of hiking boots caked with mud at the doorstep of a cabin" \
    system.guidance.beta_dpo=0.01 system.guidance.reward_model="hpsv2-score" system.guidance.smooth_dpo=True \
    tag="ablation-soft-gap"

python3 launch.py --config configs/dreamdpo/mvdream-sd21-lmm.yaml \
    --train --gpu 0 --eval-imagereward \
    system.prompt_processor.prompt="A pair of hiking boots caked with mud at the doorstep of a cabin" \
    system.guidance.ai_start_iter=1200 system.guidance.ai_prob=0.1 system.guidance.ai_feedback_class="lmm-score" \
    tag="ablation-new-lmm"
