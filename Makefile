# GCP Cloud Run deployment
#
# One-time GCP setup (run once per project):
#   gcloud services enable run.googleapis.com artifactregistry.googleapis.com
#   gcloud artifacts repositories create llmops-rag \
#     --repository-format=docker --location=us-central1
#   gcloud auth configure-docker us-central1-docker.pkg.dev
#
# Typical workflow:
#   source .env
#   make push             # build + tag + push all images
#   make deploy-serving
#   make deploy-ui        # reads the serving Cloud Run URL automatically
#   make deploy-ingestion # create (or update) the Cloud Run Job
#   make run-ingestion    # execute one ingestion run

GIT_SHA         := $(shell git rev-parse --short HEAD)
REGION          ?= us-central1
REGISTRY        ?= $(REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/llmops-rag
VERSION         ?= $(GIT_SHA)

SERVING_IMAGE   := $(REGISTRY)/rag-serving:$(VERSION)
UI_IMAGE        := $(REGISTRY)/rag-ui:$(VERSION)
INGESTION_IMAGE := $(REGISTRY)/rag-ingestion:$(VERSION)

.PHONY: build push deploy-serving deploy-ui deploy-ingestion run-ingestion

# ── Build ─────────────────────────────────────────────────────────────────────
build:
	docker compose -f docker/docker-compose.yml build --no-cache base
	docker compose -f docker/docker-compose.yml build serving ingestion ui

# ── Push ──────────────────────────────────────────────────────────────────────
push: build
	docker tag rag-serving   $(SERVING_IMAGE)
	docker tag rag-ui        $(UI_IMAGE)
	docker tag rag-ingestion $(INGESTION_IMAGE)
	docker push $(SERVING_IMAGE)
	docker push $(UI_IMAGE)
	docker push $(INGESTION_IMAGE)

# ── Cloud Run Services ────────────────────────────────────────────────────────
deploy-serving:
	gcloud run deploy rag-serving \
	  --image=$(SERVING_IMAGE) \
	  --region=$(REGION) \
	  --allow-unauthenticated \
	  --port=8000 \
	  --memory=512Mi \
	  --min-instances=0 \
	  --set-env-vars="\
OPENAI_API_KEY=$$OPENAI_API_KEY,\
ZILLIZ_URI=$$ZILLIZ_URI,\
ZILLIZ_TOKEN=$$ZILLIZ_TOKEN,\
LANGFUSE_PUBLIC_KEY=$$LANGFUSE_PUBLIC_KEY,\
LANGFUSE_SECRET_KEY=$$LANGFUSE_SECRET_KEY,\
LANGFUSE_HOST=$$LANGFUSE_HOST"

deploy-ui:
	$(eval SERVING_URL := $(shell gcloud run services describe rag-serving \
	  --region=$(REGION) --format='value(status.url)'))
	gcloud run deploy rag-ui \
	  --image=$(UI_IMAGE) \
	  --region=$(REGION) \
	  --allow-unauthenticated \
	  --port=8080 \
	  --memory=512Mi \
	  --min-instances=0 \
	  --set-env-vars="\
SERVING_URL=$(SERVING_URL),\
OPENAI_API_KEY=$$OPENAI_API_KEY"

# ── Cloud Run Job (ingestion) ─────────────────────────────────────────────────
deploy-ingestion:
	gcloud run jobs create rag-ingestion \
	  --image=$(INGESTION_IMAGE) \
	  --region=$(REGION) \
	  --memory=1Gi \
	  --task-timeout=1800 \
	  --set-env-vars="\
OPENAI_API_KEY=$$OPENAI_API_KEY,\
ZILLIZ_URI=$$ZILLIZ_URI,\
ZILLIZ_TOKEN=$$ZILLIZ_TOKEN" \
	2>/dev/null \
	|| gcloud run jobs update rag-ingestion \
	  --image=$(INGESTION_IMAGE) \
	  --region=$(REGION) \
	  --memory=1Gi \
	  --task-timeout=1800 \
	  --set-env-vars="\
OPENAI_API_KEY=$$OPENAI_API_KEY,\
ZILLIZ_URI=$$ZILLIZ_URI,\
ZILLIZ_TOKEN=$$ZILLIZ_TOKEN"

run-ingestion:
	gcloud run jobs execute rag-ingestion --region=$(REGION) --wait
