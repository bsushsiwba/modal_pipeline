git clone https://github.com/bsushsiwba/IDM-VTON

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

cd ckpt
cd densepose
rm model_final_162be9.pkl
wget https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/densepose/model_final_162be9.pkl
cd ..

cd humanparsing
rm parsing_atr.onnx
rm parsing_lip.onnx
wget https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx
wget https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx
cd ..

cd openpose
cd ckpts
rm body_pose_model.pth
wget https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth
cd ..
cd ..
cd ..

conda env create -f environment.yaml
conda activate idm

pip install huggingface_hub==0.25.1
pip install pydantic==2.8.2
pip install pydantic-core==2.20.1
pip install fastapi==0.112.4

python gradio_demo/app.py

# SAM Commands
python -m venv sam
source sam/bin/activate

pip install torch==2.5.1
pip install packaging hydra-core scikit-learn wheel iopath onnxruntime rembg
pip install transformers==4.45.2

pip install git+https://github.com/bsushsiwba/grounded-sam-pypi.git