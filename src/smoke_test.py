"""Smoke test for runtime configs and RAG initialization (no external calls).
This script loads config/models.yaml, creates RuntimeModelConfig objects and
initializes the RAG_module in a dry-run mode (won't force rebuild embeddings).
"""
from agent_import import load_config_from_file, RuntimeModelConfig, create_client
from rag import RAG_module
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT, "config", "models.yaml")

conf = load_config_from_file(CONFIG_PATH)
print("Loaded config:", conf)

llm_conf = conf.get("llm", {})
emb_conf = conf.get("embeddings", {})

llm_runtime = RuntimeModelConfig(
    provider=llm_conf.get("provider", "openai"),
    api_key=llm_conf.get("api_key") or None,
    api_base=llm_conf.get("api_base") or None,
    model_name=llm_conf.get("model_name", "gpt-4o"),
    temperature=llm_conf.get("temperature", 0.3),
)

emb_runtime = RuntimeModelConfig(
    provider=emb_conf.get("provider", "openai"),
    api_key=emb_conf.get("api_key") or None,
    api_base=emb_conf.get("api_base") or None,
    model_name=emb_conf.get("model_name", "text-embedding-3-small"),
    is_embedding=True,
)

print("LLM runtime config:", llm_runtime)
print("Embedding runtime config:", emb_runtime)

# Create clients but avoid making API calls in smoke test
try:
    emb_client = create_client(emb_runtime, client_type="embeddings")
    print("Embeddings client created:", type(emb_client))
except Exception as e:
    print("Could not create embeddings client (this may be expected without keys):", e)

try:
    llm_client = create_client(llm_runtime, client_type="llm")
    print("LLM client created:", type(llm_client))
except Exception as e:
    print("Could not create LLM client (this may be expected without keys):", e)

# Attempt to create RAG_module in a doc folder; create sample docs if missing
DOCS_DIR = os.path.join(ROOT, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)
SAMPLE_DOC = os.path.join(DOCS_DIR, "sample.txt")
if not os.path.exists(SAMPLE_DOC):
    with open(SAMPLE_DOC, "w", encoding="utf-8") as f:
        f.write("This is a small sample document for smoke testing.\n")

try:
    rag = RAG_module(DOCS_DIR, embedding_config=emb_runtime, llm_config=llm_runtime, rebuild_embeddings=False)
    print("RAG_module initialized. Retriever ready.")
except Exception as e:
    print("RAG_module initialization failed (likely due to missing API keys).", e)

print("Smoke test complete.")
