FROM python:3.11-slim

WORKDIR /app

# CPU-only torch: sentence-transformers pulls it in, and the default wheel
# drags CUDA along for nothing.
RUN pip install --no-cache-dir torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY load.py .

ENTRYPOINT ["python", "load.py"]
