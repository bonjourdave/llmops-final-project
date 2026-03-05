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
#   make create-service-accounts  # create dedicated SAs (once)
#   make create-secrets           # upload secrets to Secret Manager (once, or to rotate)
#   make grant-secret-access      # grant each SA access to only its secrets (once)
#   make push                     # build + tag + push all images
#   make deploy-serving
#   make deploy-ui           	  # reads the serving Cloud Run URL automatically
#   make deploy-ingestion    	  # create (or update) the Cloud Run Job
#   make run-ingestion       	  # execute one ingestion run

-include .env   # the leading dash tells Make to silently skip if file not found
export

GIT_SHA         := $(shell git rev-parse --short HEAD)
REGION          ?= us-central1
REGISTRY        ?= $(REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/llmops-rag
VERSION         ?= $(GIT_SHA)

SERVING_IMAGE   := $(REGISTRY)/rag-serving:$(VERSION)
UI_IMAGE        := $(REGISTRY)/rag-ui:$(VERSION)
INGESTION_IMAGE := $(REGISTRY)/rag-ingestion:$(VERSION)

# Dedicated, least-privilege service accounts (one per service)
SA_SERVING   := rag-serving-sa@$(GCP_PROJECT_ID).iam.gserviceaccount.com
SA_UI        := rag-ui-sa@$(GCP_PROJECT_ID).iam.gserviceaccount.com
SA_INGESTION := rag-ingestion-sa@$(GCP_PROJECT_ID).iam.gserviceaccount.com

.PHONY: build push create-service-accounts create-secrets grant-secret-access \
        deploy-serving deploy-ui deploy-ingestion run-ingestion

# ── Build ─────────────────────────────────────────────────────────────────────
# Uses two compose files: base config + prod overlay (which pins platform to
# linux/amd64 required by Cloud Run). Without the overlay, Docker defaults to
# the host machine's native architecture (arm64 on Apple Silicon), which Cloud
# Run rejects. The -f flags are additive — prod overlay merges into base config.
build:
	docker compose \
	  -f docker/docker-compose.yml \
	  -f docker/docker-compose.prod.yml \
	  build serving ingestion ui

# ── Push ──────────────────────────────────────────────────────────────────────
push: build
	docker tag rag-serving   $(SERVING_IMAGE)
	docker tag rag-ui        $(UI_IMAGE)
	docker tag rag-ingestion $(INGESTION_IMAGE)
	docker push $(SERVING_IMAGE)
	docker push $(UI_IMAGE)
	docker push $(INGESTION_IMAGE)

# ── Service Accounts ──────────────────────────────────────────────────────────
# Idempotent: the `|| true` swallows the "already exists" error on re-runs.
# Each SA is born with zero permissions; access is granted explicitly below.
create-service-accounts:
	gcloud iam service-accounts create rag-serving-sa \
	  --display-name="RAG Serving Service Account" \
	  --project=$(GCP_PROJECT_ID) 2>/dev/null || true
	gcloud iam service-accounts create rag-ui-sa \
	  --display-name="RAG UI Service Account" \
	  --project=$(GCP_PROJECT_ID) 2>/dev/null || true
	gcloud iam service-accounts create rag-ingestion-sa \
	  --display-name="RAG Ingestion Service Account" \
	  --project=$(GCP_PROJECT_ID) 2>/dev/null || true

# ── Secrets ───────────────────────────────────────────────────────────────────
# Uploads each secret to GCP Secret Manager for use by Cloud Run at runtime.
# Idempotent: attempts to create the secret first; if it already exists,
# adds a new version instead (useful for key rotation).
# --replication-policy=user-managed --locations=$(REGION) is required because
# the org policy (constraints/gcp.resourceLocations) blocks Secret Manager's
# default global replication — secrets must be pinned to an explicit region.
create-secrets:
	@printf '%s' "$$OPENAI_API_KEY"      | gcloud secrets create openai-api-key      --data-file=- --replication-policy=user-managed --locations=$(REGION) 2>/dev/null \
	  || printf '%s' "$$OPENAI_API_KEY"      | gcloud secrets versions add openai-api-key      --data-file=-
	@printf '%s' "$$ZILLIZ_URI"          | gcloud secrets create zilliz-uri          --data-file=- --replication-policy=user-managed --locations=$(REGION) 2>/dev/null \
	  || printf '%s' "$$ZILLIZ_URI"          | gcloud secrets versions add zilliz-uri          --data-file=-
	@printf '%s' "$$ZILLIZ_TOKEN"        | gcloud secrets create zilliz-token        --data-file=- --replication-policy=user-managed --locations=$(REGION) 2>/dev/null \
	  || printf '%s' "$$ZILLIZ_TOKEN"        | gcloud secrets versions add zilliz-token        --data-file=-
	@printf '%s' "$$LANGFUSE_PUBLIC_KEY" | gcloud secrets create langfuse-public-key --data-file=- --replication-policy=user-managed --locations=$(REGION) 2>/dev/null \
	  || printf '%s' "$$LANGFUSE_PUBLIC_KEY" | gcloud secrets versions add langfuse-public-key --data-file=-
	@printf '%s' "$$LANGFUSE_SECRET_KEY" | gcloud secrets create langfuse-secret-key --data-file=- --replication-policy=user-managed --locations=$(REGION) 2>/dev/null \
	  || printf '%s' "$$LANGFUSE_SECRET_KEY" | gcloud secrets versions add langfuse-secret-key --data-file=-

# ── Secret Access (per-SA, per-secret) ───────────────────────────────────────
# Each SA only gets access to the secrets it actually needs at runtime.
# Serving: OpenAI + Zilliz (for retrieval) + Langfuse (for tracing)
# UI:      OpenAI only (for any client-side calls; no vector DB or tracing)
# Ingestion: OpenAI + Zilliz only (writes embeddings; no Langfuse)
grant-secret-access:
	@for secret in openai-api-key zilliz-uri zilliz-token langfuse-public-key langfuse-secret-key; do \
	  gcloud secrets add-iam-policy-binding $$secret \
	    --member="serviceAccount:$(SA_SERVING)" \
	    --role="roles/secretmanager.secretAccessor"; \
	done
	@for secret in openai-api-key; do \
	  gcloud secrets add-iam-policy-binding $$secret \
	    --member="serviceAccount:$(SA_UI)" \
	    --role="roles/secretmanager.secretAccessor"; \
	done
	@for secret in openai-api-key zilliz-uri zilliz-token; do \
	  gcloud secrets add-iam-policy-binding $$secret \
	    --member="serviceAccount:$(SA_INGESTION)" \
	    --role="roles/secretmanager.secretAccessor"; \
	done

# ── Cloud Run Services ────────────────────────────────────────────────────────
deploy-serving:
	gcloud run deploy rag-serving \
	  --image=$(SERVING_IMAGE) \
	  --region=$(REGION) \
	  --allow-unauthenticated \
	  --port=8000 \
	  --memory=1Gi \
	  --min-instances=0 \
	  --service-account=$(SA_SERVING) \
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
	  --service-account=$(SA_UI) \
	  --set-env-vars="SERVING_URL=$(SERVING_URL)" \
	  --set-secrets="OPENAI_API_KEY=openai-api-key:latest"

# ── Cloud Run Job (ingestion) ─────────────────────────────────────────────────
deploy-ingestion:
	gcloud run jobs create rag-ingestion \
	  --image=$(INGESTION_IMAGE) \
	  --region=$(REGION) \
	  --memory=1Gi \
	  --task-timeout=1800 \
	  --service-account=$(SA_INGESTION) \
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
	  --service-account=$(SA_INGESTION) \
	  --set-secrets="\
OPENAI_API_KEY=openai-api-key:latest,\
ZILLIZ_URI=zilliz-uri:latest,\
ZILLIZ_TOKEN=zilliz-token:latest"

run-ingestion:
	gcloud run jobs execute rag-ingestion --region=$(REGION) --wait