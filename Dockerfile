FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        jq \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "fugashi==1.1.1" \
        "ipadic==1.0.0" \
        "scikit-learn==1.0.1"

WORKDIR /app
COPY model model
COPY utils utils
COPY train.py .
COPY predict.py .
COPY submission.sh .
