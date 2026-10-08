FROM python:3.12-slim

WORKDIR /dataset
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# build:  docker compose run --rm build configs/legal-1200.yaml
# ui:     docker compose up ui   ->  http://localhost:8765
ENTRYPOINT ["python", "build/build.py"]
