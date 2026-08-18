FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/app \
    GRAND_CHALLENGE_MAX_WORKERS=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app
COPY requirements.txt /opt/app/requirements.txt
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt

COPY main.py inference.py preprocessing.py /opt/app/
COPY modeling /opt/app/modeling
COPY ta36 /opt/app/ta36
COPY vendor/nnunetv2 /opt/app/nnunetv2
COPY vendor/dynamic_network_architectures /opt/app/dynamic_network_architectures

RUN chown -R user:user /opt/app
USER user

ENTRYPOINT ["python", "main.py"]
