FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
RUN apt update && apt install -y python3.10 python3-pip libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY ./models /app/models
COPY ./main.py /app/main.py
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
# RUN pip3 install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
CMD uvicorn main:app --reload