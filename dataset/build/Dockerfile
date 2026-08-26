FROM python:3.11-slim

WORKDIR /dataset

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# torch + torchvision must come from the same index — docling needs torchvision
# ops (e.g. torchvision::nms) for its layout model.
RUN pip install --no-cache-dir \
    torch==2.13.0+cpu torchvision==0.28.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "build/build.py"]
