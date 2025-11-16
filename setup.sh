#!/bin/bash -e

conda create -n dreamdpo python=3.9
conda activate dreamdpo
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118
pip install ninja
git clone https://github.com/bytedance/MVDream extern/MVDream
pip install -e extern/MVDream 
pip install -r requirements.txt

pip install hpsv2 image-reward brisque libsvm-official==3.30.0 fairscale dashscope kiui "numpy<2.0" git+https://github.com/openai/CLIP.git

grep -rl "RedbeardNZ/stable-diffusion-2-1-base" . \
  | xargs sed -i 's|RedbeardNZ/stable-diffusion-2-1-base|RedbeardNZ/stable-diffusion-2-1-base|g'

mkdir -p model_weights/HPSv2
huggingface-cli download xswu/HPSv2 --local-dir model_weights/HPSv2

mkdir -p model_weights/ImageReward
huggingface-cli download THUDM/ImageReward --local-dir model_weights/ImageReward

huggingface-cli download RedbeardNZ/stable-diffusion-2-1-base
huggingface-cli download google-bert/bert-base-uncased
