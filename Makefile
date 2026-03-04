# GCP Cloud Run deployment
#
# One-time GCP setup (run once per project):
#   gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
#   gcloud artifacts repositories create llmops-rag \
#     --repository-format=docker --location=us-central1
#   gcloud auth configure-docker us-central1-docker.pkg.dev
#
# Typical workflow:
#   source .env
#   make create-secrets      # upload secrets to Secret Manager (once, or to rotate)
#   make grant-secret-access # grant Cloud Run SA access to secrets (once)
#   make push                # build + tag + push all images
#   make deploy-serving
#   make deploy-ui           # reads the serving Cloud Run URL automatically
#   make deploy-ingestion    # create (or update) the Cloud Run Job
#   make run-ingestion       # execute one ingestion run

GIT_SHA         := $(shell git rev-parse --short HEAD)
REGION          ?= us-central1
REGISTRY        ?= $(REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/llmops-rag
VERSION         ?= $(GIT_SHA)

SERVING_IMAGE   := $(REGISTRY)/rag-serving:$(VERSION)
UI_IMAGE        := $(REGISTRY)/rag-ui:$(VERSION)
INGESTION_IMAGE := $(REGISTRY)/rag-ingestion:$(VERSION)

.PHONY: build push create-secrets grant-secret-access deploy-serving deploy-ui deploy-ingestion run-ingestion

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

# ── Secrets ───────────────────────────────────────────────────────────────────
# Idempotent: creates the secret on first run; adds a new version on rotation.
# printf '%s' avoids the trailing newline that echo adds (SM stores raw bytes).
create-secrets:
	@printf '%s' "$$OPENAI_API_KEY"      | gcloud secrets create openai-api-key      --data-file=- 2>/dev/null \
	  || printf '%s' "$$OPENAI_API_KEY"      | gcloud secrets versions add openai-api-key      --data-file=-
	@printf '%s' "$$ZILLIZ_URI"          | gcloud secrets create zilliz-uri          --data-file=- 2>/dev/null \
	  || printf '%s' "$$ZILLIZ_URI"          | gcloud secrets versions add zilliz-uri          --data-file=-
	@printf '%s' "$$ZILLIZ_TOKEN"        | gcloud secrets create zilliz-token        --data-file=- 2>/dev/null \
	  || printf '%s' "$$ZILLIZ_TOKEN"        | gcloud secrets versions add zilliz-token        --data-file=-
	@printf '%s' "$$LANGFUSE_PUBLIC_KEY" | gcloud secrets create langfuse-public-key --data-file=- 2>/dev/null \
	  || printf '%s' "$$LANGFUSE_PUBLIC_KEY" | gcloud secrets versions add langfuse-public-key --data-file=-
	@printf '%s' "$$LANGFUSE_SECRET_KEY" | gcloud secrets create langfuse-secret-key --data-file=- 2>/dev/null \
	  || printf '%s' "$$LANGFUSE_SECRET_KEY" | gcloud secrets versions add langfuse-secret-key --data-file=-

# Grants the default Cloud Run service account read access to all secrets.
grant-secret-access:
	$(eval PROJECT_NUMBER := $(shell gcloud projects describe $(GCP_PROJECT_ID) \
	  --format='value(projectNumber)'))
	gcloud projects add-iam-policy-binding $(GCP_PROJECT_ID) \
	  --member="serviceAccount:$(PROJECT_NUMBER)-compute@developer.gserviceaccount.com" \
	  --role="roles/secretmanager.secretAccessor"

# ── Cloud Run Services ────────────────────────────────────────────────────────
deploy-serving:
	gcloud run deploy rag-serving \
	  --image=$(SERVING_IMAGE) \
	  --region=$(REGION) \
	  --allow-unauthenticated \
	  --port=8000 \
	  --memory=512Mi \
	  --min-instances=0 \
	  --set-env-vars="LANGFUSE_HOST=$$LANGFUSE_HOST" \
	  --set-secrets="\
OPENAI_API_KEY=openai-api-key:latest,\
ZILLIZ_URI=zilliz-uri:latest,\
ZILLIZ_TOKEN=zilliz-token:latest,\
LANGFUSE_PUBLIC_KEY=langfuse-public-key:latest,\
LANGFUSE_SECRET_KEY=langfuse-secret-key:latest"

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
	  --set-env-vars="SERVING_URL=$(SERVING_URL)" \
	  --set-secrets="OPENAI_API_KEY=openai-api-key:latest"

# ── Cloud Run Job (ingestion) ─────────────────────────────────────────────────
deploy-ingestion:
	gcloud run jobs create rag-ingestion \
	  --image=$(INGESTION_IMAGE) \
	  --region=$(REGION) \
	  --memory=1Gi \
	  --task-timeout=1800 \
	  --set-secrets="\
OPENAI_API_KEY=openai-api-key:latest,\
ZILLIZ_URI=zilliz-uri:latest,\
ZILLIZ_TOKEN=zilliz-token:latest" \
	2>/dev/null \
	|| gcloud run jobs update rag-ingestion \
	  --image=$(INGESTION_IMAGE) \
	  --region=$(REGION) \
	  --memory=1Gi \
	  --task-timeout=1800 \
	  --set-secrets="\
OPENAI_API_KEY=openai-api-key:latest,\
ZILLIZ_URI=zilliz-uri:latest,\
ZILLIZ_TOKEN=zilliz-token:latest"

run-ingestion:
	gcloud run jobs execute rag-ingestion --region=$(REGION) --wait
