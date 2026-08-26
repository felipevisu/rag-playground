FROM python:3.11-slim

WORKDIR /eval

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "run.py"]
