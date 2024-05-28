# example deep learning VM
# for a full list see us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# and for details see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310

RUN apt install -y -q ffmpeg

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
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the files into the container
COPY src .
ARG MODEL_ROOT="./models/whisper-small.en-til-24"
COPY ${MODEL_ROOT}/processor  ${MODEL_ROOT}/
COPY ${MODEL_ROOT}/config.json ${MODEL_ROOT}/generation_config.json ${MODEL_ROOT}/pytorch_model.bin ${MODEL_ROOT}/

# start model service
CMD uvicorn api_service:app --port 5001 --host 0.0.0.0