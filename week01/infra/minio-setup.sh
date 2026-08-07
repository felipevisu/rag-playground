#!/bin/sh
# MinIO bootstrap: alias, bucket, webhook subscription, seed PDFs.
# Idempotent — safe to re-run on every compose up.

set -e

ALIAS=local
BUCKET=documents
WEBHOOK_ARN=arn:minio:sqs::WORKER:webhook
SOURCE_DIR=/source-pdfs

echo "→ Configuring mc alias..."
mc alias set $ALIAS http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"

echo "→ Waiting for MinIO to be ready..."
until mc ready $ALIAS; do
  sleep 2
done

# NOTE: worker container is started before us via depends_on. If worker is
# still loading models when an event fires, MinIO retries the webhook —
# no explicit readiness probe needed here (and mc image lacks curl/wget).

echo "Creating bucket: $BUCKET"
mc mb --ignore-existing $ALIAS/$BUCKET

echo "Subscribing bucket to webhook ($WEBHOOK_ARN, PUT, *.pdf)..."
# Remove any stale subscription first (ignore error if none), then add.
mc event remove $ALIAS/$BUCKET $WEBHOOK_ARN --event put --suffix .pdf 2>/dev/null || true
mc event add    $ALIAS/$BUCKET $WEBHOOK_ARN --event put --suffix .pdf

echo "Seeding bucket with existing PDFs from $SOURCE_DIR..."
if [ -d "$SOURCE_DIR" ]; then
  for f in "$SOURCE_DIR"/*.pdf; do
    [ -f "$f" ] || continue
    name=$(basename "$f")
    if mc stat $ALIAS/$BUCKET/"$name" >/dev/null 2>&1; then
      echo "  [exists] $name"
    else
      echo "  [upload] $name"
      mc cp "$f" $ALIAS/$BUCKET/"$name"
    fi
  done
else
  echo "  no source dir, skipping seed."
fi

echo "MinIO setup complete."
