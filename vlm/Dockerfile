# example deep learning VM
# for a full list see us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# and for details see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# pip gives a warning if you install packages as root
# set this flag to just ignore the warning
ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install -U pip
WORKDIR /workspace

# install other requirements
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --upgrade ultralytics

# copy the rest of the files into the container
COPY src .
COPY models/yolov8m-v2-300/weights/best.pt ./models/yolov8/weights/
COPY models/clip-finetune/best.pt ./models/clip-finetune/

# start model service
CMD uvicorn api_service:app --port 5004 --host 0.0.0.0