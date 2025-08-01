# Belgium_Traffic_Signs_Classification_using_PyTorch_with_OpenVINO

Step to run

1. Train PyTorch model
python train.py

2. Convert to OpenVINO FP16
python convert.py

3. Run inference

python infer.py  [-h] [--device {CPU,GPU,NPU,AUTO,ALL}] [--batch-sizes BATCH_SIZES]

python eval.py  [-h] [--device {CPU,GPU,NPU,AUTO,ALL}] [--batch-sizes BATCH_SIZES]

python benchmark.py [-h] [--device {CPU,GPU,NPU,AUTO,ALL}] [--batch-sizes BATCH_SIZES]


This project required
openvino
torch
tqdm
pillow


For Intel pytorch extension:

python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/xpu
python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

Read more:
https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.7.10%2Bxpu&os=windows&package=pip

