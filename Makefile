# RAG POC — run-order wrapper
# The GPU is single-tenant: the smoke tests, the bench, and `vllm serve`
# all want full VRAM. Stop one before starting another.

SHELL := /bin/bash
LLM_MODEL ?= meta-llama/Llama-3.1-8B-Instruct
QDRANT_CONTAINER ?= qdrant-poc

.PHONY: help install typecheck smoke smoke-tiny smoke-llama bench qdrant qdrant-stop vllm rag clean

help:
	@echo "Targets:"
	@echo "  install      npm install + check ROCm visibility"
	@echo "  typecheck    tsc --noEmit"
	@echo "  smoke-tiny   in-process opt-125m sanity"
	@echo "  smoke-llama  in-process Llama-3.1-8B fp16 load"
	@echo "  smoke        smoke-tiny then smoke-llama"
	@echo "  bench        AWQ-INT4 batched throughput probe"
	@echo "  qdrant       launch Qdrant container (detached)"
	@echo "  qdrant-stop  stop and remove Qdrant container"
	@echo "  vllm         vllm serve $(LLM_MODEL) (foreground)"
	@echo "  rag          run the orchestrator (assumes Qdrant + vLLM up)"

install:
	npm install
	@rocminfo | grep -q gfx1100 || (echo "gfx1100 not visible — check ROCm install" && exit 1)

typecheck:
	npx tsc --noEmit

smoke-tiny:
	python scripts/bench/smoke_opt125m.py

smoke-llama:
	python scripts/bench/smoke_llama3_8b.py

smoke: smoke-tiny smoke-llama

bench:
	python scripts/bench/bench_llama31_awq_int4.py

qdrant:
	docker run -d --name $(QDRANT_CONTAINER) -p 6333:6333 -p 6334:6334 \
		-v $(PWD)/qdrant_storage:/qdrant/storage \
		qdrant/qdrant

qdrant-stop:
	-docker stop $(QDRANT_CONTAINER)
	-docker rm $(QDRANT_CONTAINER)

vllm:
	vllm serve $(LLM_MODEL) \
		--dtype float16 \
		--enforce-eager \
		--gpu-memory-utilization 0.92 \
		--max-model-len 4096 \
		--max-num-seqs 4

rag:
	npm start

clean:
	rm -rf dist node_modules
