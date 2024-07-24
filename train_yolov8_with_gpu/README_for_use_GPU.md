## For CUDA Toolkit 11.8

read https://medium.com/@sabrimol91/how-to-use-yolov8-for-object-detection-fc3aa17ab860

- Install GeForce Experience and update NVIDIA driver to the latest version

- Download CUDA Toolkit 11.8.0 (October 2022) and install CUDA Toolkit
  https://developer.nvidia.com/cuda-toolkit-archive

- Download cuDNN v8.9.7 (December 5th, 2023), for CUDA 11.x
  https://developer.nvidia.com/rdp/cudnn-archive

- Adjust cuDNN 'C:\ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v11.8'

- restart computer

- install lib
    
      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
      pip3 install ultralytics