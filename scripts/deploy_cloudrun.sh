#!/bin/bash
# Deploy the web app to Google Cloud Run.
#
#   ./scripts/deploy_cloudrun.sh
#
# Cloud Run builds the image itself from the Dockerfile, so there is no local
# docker daemon involved. What gets uploaded is governed by .gcloudignore, which
# keeps the 11 GB of training data out of the build context.
#
# Sizing notes, since the defaults are wrong for this workload:
#   --memory 2Gi     the 512Mi default OOMs; torch plus a 3 megapixel image
#                    needs room, and an OOM on Cloud Run looks like a 503 with
#                    nothing useful in the log
#   --concurrency 2  encoding is CPU bound and takes seconds, so piling requests
#                    onto one instance makes everyone wait rather than helping
#   --max-instances  a hard cap, so a burst of traffic cannot quietly run up a
#                    bill on an account that has billing enabled
#   --min-instances 0  scale to zero when idle, which is what keeps it free.
#                    The cost is a cold start of roughly half a minute while
#                    torch loads.

set -euo pipefail

SERVICE="${SERVICE:-json-camera}"
REGION="${REGION:-europe-west2}"
MEMORY="${MEMORY:-2Gi}"
CPU="${CPU:-1}"
CONCURRENCY="${CONCURRENCY:-2}"
MAX_INSTANCES="${MAX_INSTANCES:-3}"

cd "$(dirname "$0")/.."

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is not installed."
  echo "  brew install --cask google-cloud-sdk"
  exit 1
fi

PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
  echo "No gcloud project is set."
  echo "  gcloud projects create YOUR-PROJECT-ID"
  echo "  gcloud config set project YOUR-PROJECT-ID"
  exit 1
fi

if [ ! -f checkpoints/stable/jc-final.pt ]; then
  echo "No checkpoint at checkpoints/stable/jc-final.pt."
  echo "  .venv/bin/jsoncam export checkpoints/jc.best.pt -o checkpoints/stable/jc-final.pt"
  exit 1
fi

echo "project   $PROJECT"
echo "service   $SERVICE  ($REGION)"
echo "sizing    $CPU vCPU, $MEMORY, concurrency $CONCURRENCY, max $MAX_INSTANCES instances"
echo "uploading $(du -sh --exclude=.git . 2>/dev/null | cut -f1 || echo '~8 MB') of build context"
echo

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory "$MEMORY" \
  --cpu "$CPU" \
  --concurrency "$CONCURRENCY" \
  --min-instances 0 \
  --max-instances "$MAX_INSTANCES" \
  --timeout 300 \
  --port 8080 \
  --set-env-vars "HOST=0.0.0.0,JSONCAM_MODELS=/home/user/app/checkpoints"

echo
echo "URL:"
gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
