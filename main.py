#main.py
print("\n\n")
print("                                 #####################################")
print("                                    AI Power Industry Transformation")
print("                                            ALPHA  PROJECT ")
print("                                 #####################################")






import time
start_time = time.perf_counter()
print("\n\n...  start imports ...", flush=True)
import os, sys, time, json, threading, hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Literal

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"          library imports ⏱️ {elapsed:.4f} seconds")

# =============================
# Env
# =============================
t0 = time.perf_counter()

from dotenv import load_dotenv, find_dotenv
_env = find_dotenv("runpod.env", usecwd=True) or "runpod.env"
load_dotenv(_env, override=True)
print(f"ENV loaded from: {_env}", flush=True)

print(f"          Env import ⏱️  {(time.perf_counter()-t0):.4f} seconds")

# =============================
# FastAPI
# =============================
t0 = time.perf_counter()

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import httpx

print(f"          fastAPI import ⏱️  {(time.perf_counter()-t0):.4f} seconds")

# =============================
# LlamaIndex core
# =============================
t0 = time.perf_counter()

from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
    get_response_synthesizer,
)
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.response_synthesizers import ResponseMode

# === Smart chunking imports  ===
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SemanticSplitterNodeParser,
    SentenceWindowNodeParser,
)

from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    LongContextReorder,
)

# Optional reranker 
try:
    from llama_index.core.postprocessor import SentenceTransformerRerank
    HAVE_ST_RERANK = True
except Exception:
    HAVE_ST_RERANK = False

# Optional BM25
try:
    from llama_index.core.retrievers import BM25Retriever
    HAVE_BM25_NATIVE = True
except Exception:
    HAVE_BM25_NATIVE = False

# Lightweight doc/kv stores (if needed)
from llama_index.core.storage.docstore import SimpleDocumentStore

print(f"          llama index import ⏱️  {(time.perf_counter()-t0):.4f} seconds")

# =============================
# FAISS + Torch
# =============================
t0 = time.perf_counter()

import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
import torch

print(f"          FAISS import ⏱️  {(time.perf_counter()-t0):.4f} seconds")

# ======= Cache control switches =======
import gc
CLEAN_CACHE_ON_START = os.getenv("CLEAN_CACHE_ON_START", "0") == "1"
CLEAN_CACHE_AFTER_BUILD = os.getenv("CLEAN_CACHE_AFTER_BUILD", "0") == "1"

def _clean_caches(label: str = ""):
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    if label:
        print(f"[CACHE] cleaned {label}", flush=True)



if CLEAN_CACHE_ON_START:
    _clean_caches("before init")

# =============================
# LLM & Embeddings
# =============================
t0 = time.perf_counter()

#-------------------------------------
from llama_index.core.llms import CustomLLM, LLMMetadata, CompletionResponse
import httpx
from pydantic import Field

class YtlaiLLM(CustomLLM):
    api_key: str = Field(...)
    model: str = Field(default="ilmu-trial")
    base_url: str = Field(default="https://api.ytlailabs.tech/preview/v1/chat/completions")

    # required by LlamaIndex
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            max_input_tokens=8192,
            max_output_tokens=2048,
        )

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            r = httpx.post(self.base_url, json=payload, headers=headers, timeout=90.0)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[YtlaiLLM] API error: {e}", flush=True)
            text, data = f"[YtlaiLLM error: {e}]", {}

        return CompletionResponse(text=text, raw=data)

    def stream_complete(self, prompt: str, **kwargs):
        yield self.complete(prompt, **kwargs)

    def chat(self, messages, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "messages": messages}
        try:
            r = httpx.post(self.base_url, json=payload, headers=headers, timeout=90.0)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[YtlaiLLM chat error: {e}]"





#-------------------------------------

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print(f"          huggingface import ⏱️ {(time.perf_counter()-t0):.4f} seconds")

# ========== Timing helpers ==========
from contextlib import contextmanager
def _now_s() -> float:
    return time.perf_counter() * 1.0

@contextmanager
def stage(label: str):
    t0 = _now_s()
    print(f"[TIMER] ▶ {label} ...", flush=True)
    try:
        yield
    finally:
        dt = _now_s() - t0
        print(f"[TIMER] ✅ {label} ⏱ {dt:.1f} s", flush=True)
        
# =============================
# Config
# =============================
INDEX: Optional[VectorStoreIndex] = None
INDEX_LOCK = threading.Lock()
INDEX_BUILD_BUSY = threading.Event()

BASE_DIR = Path().resolve()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
UNSTRUCTURED_DIR = DATA_DIR / "unstructured"      
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "storage"))  

API_KEY = os.getenv("API_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

ALLOWED_EXTS = {".pdf", ".txt", ".md", ".docx", ".csv"}

MAX_UPLOAD_MB = 50


UNSTRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


Bucket = Literal["structured", "unstructured"]
STRUCTURED_DIR = DATA_DIR / "structured"
STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)

DISABLE_HIER = os.getenv("DISABLE_HIER", "0") == "1"   
DISABLE_SEM  = os.getenv("DISABLE_SEM",  "0") == "1"   

MANIFEST = STORAGE_DIR / "ingest_manifest.json"

print(f"[Paths] DATA_DIR={DATA_DIR} | UNSTRUCTURED_DIR={UNSTRUCTURED_DIR} | STORAGE_DIR={STORAGE_DIR}")




# =============================
# RAG Settings (LLM + Embeddings)
# =============================
t0 = time.perf_counter()

Settings.llm = Ollama(model=DEFAULT_MODEL, base_url=OLLAMA_URL, request_timeout=120.0)
# Settings.llm = YtlaiLLM(
#     api_key=os.getenv("SERVICE_API_KEY"),
#     model="ilmu-trial"
# )


EMB_DEVICE = "cuda"
EMB_BATCH  = int(os.getenv("EMB_BATCH", "32"))
model_ref = os.getenv("EMBED_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5").strip()
EMB_MAX_LEN = int(os.getenv("EMB_MAX_LEN", "1024"))  

_DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}
EMB_DTYPE = os.getenv("EMB_DTYPE", "float16").lower()
TORCH_DTYPE = _DTYPE_MAP.get(EMB_DTYPE, torch.float16)
# if Path(model_ref).exists() or model_ref.startswith(("/", "./", "../")):
#     raise ValueError(f"Local paths are disabled for embeddings: {model_ref}")

print(f"[Embeddings] hub_model={model_ref} device={EMB_DEVICE} batch={EMB_BATCH}")

with stage("Embeddings: load (HuggingFaceEmbedding)"):
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=model_ref,
        device=EMB_DEVICE,
        embed_batch_size=EMB_BATCH,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": TORCH_DTYPE},
        tokenizer_kwargs={
            "truncation": True,
            "max_length": EMB_MAX_LEN,
            "padding": "longest"  
        },
    )

with stage("Embeddings: warmup get_query_embedding"):
    _ = Settings.embed_model.get_query_embedding("warmup for latency")
    print("[Embeddings] warmup ok")

print(f"          (SETTING) ⏱️  {(time.perf_counter()-t0):.4f} seconds ")



# =============================
# Auth
# =============================
bearer = HTTPBearer(auto_error=False)

def _check_key(creds: Optional[HTTPAuthorizationCredentials], required_key: Optional[str], name: str):
    if not required_key:
        return True
    if not creds or creds.scheme.lower() != "bearer" or creds.credentials != required_key:
        raise HTTPException(status_code=401, detail=f"{name} Unauthorized")
    return True

def require_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
    return _check_key(creds, API_KEY, "User")

def require_admin(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
    return _check_key(creds, ADMIN_KEY, "Admin")



# =============================
# Live Watcher (data/)
# =============================
WATCH_DATA      = os.getenv("WATCH_DATA", "0") == "1"
WATCH_INTERVAL  = float(os.getenv("WATCH_INTERVAL", "2"))
WATCH_DEBOUNCE  = float(os.getenv("WATCH_DEBOUNCE", "3"))
WATCH_MODE      = os.getenv("WATCH_MODE", "full").lower()  # "full" | "incremental"

WATCH_PATH = DATA_DIR  # نراقب كامل data/ (structured + unstructured)

def _tree_signature(root: Path) -> str:
    h = hashlib.sha256()
    rp = root.resolve()
    for p in sorted(rp.rglob("*")):
        if p.is_file():
            # ✅ ignore hidden files & sync metadata
            if p.name.startswith("."):
                continue
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            rel = str(p.resolve().relative_to(rp))
            h.update(rel.encode("utf-8", errors="ignore"))
            h.update(str(st.st_size).encode())
            h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()


def _watch_loop():
    """
    حلقة مراقبة بسيطة بالاستقصاء. لا تستخدم CPU كثيراً، وتنتظر WATCH_INTERVAL بين الدورات.
    تستعمل debounce لتجميع طفرات سريعة (نسخ دفعة ملفات مثلاً).
    """
    print(f"[WATCH] Started. path={WATCH_PATH} interval={WATCH_INTERVAL}s debounce={WATCH_DEBOUNCE}s mode={WATCH_MODE}", flush=True)
    last_sig = None
    last_change_ts = 0.0
    pending = False

    # بصمة ابتدائية
    try:
        last_sig = _tree_signature(WATCH_PATH)
    except Exception as e:
        print(f"[WATCH] initial signature failed: {e}", flush=True)
        last_sig = None

    while True:
        time.sleep(WATCH_INTERVAL)
        try:
            sig = _tree_signature(WATCH_PATH)
        except Exception as e:
            print(f"[WATCH] signature failed: {e}", flush=True)
            continue

        if sig != last_sig:
            # حدث تغيير: فعّل حالة انتظار ودوّن آخر وقت تغيير
            last_sig = sig
            last_change_ts = time.time()
            pending = True
            print("[WATCH] change detected -> pending reindex...", flush=True)

        if pending and (time.time() - last_change_ts) >= WATCH_DEBOUNCE:
            # انتهت نافذة الـ debounce => نفّذ إعادة الفهرسة
            try:
                # ✅ (5) serialize builds via busy flag
                if WATCH_MODE == "incremental":
                    _run_build("watch:incremental", build_index_incremental)
                else:
                    _run_build("watch:full", build_index_full)

            except Exception as e:
                print(f"[WATCH] reindex failed: {e}", flush=True)
            finally:
                pending = False




# =============================
# Smart Chunking 
# =============================
TARGET_CHUNK = int(os.getenv("TARGET_CHUNK", "800"))
MIN_CHUNK    = int(os.getenv("MIN_CHUNK", "250"))
MAX_CHUNK    = int(os.getenv("MAX_CHUNK", "1024"))

WINDOW_SIZE  = int(os.getenv("WINDOW_SIZE", "2"))   
FETCH_K          = int(os.getenv("FETCH_K", "40"))    
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "12"))
MMR_LAMBDA       = float(os.getenv("MMR_LAMBDA", "0.5"))
WEIGHT_DENSE     = float(os.getenv("WEIGHT_DENSE", "1.0"))
WEIGHT_BM25      = float(os.getenv("WEIGHT_BM25", "1.0"))
RERANK_TOP_N     = int(os.getenv("RERANK_TOP_N", "6"))

def _make_semantic_splitter() -> SemanticSplitterNodeParser:
    base_kwargs = {
        "buffer_size": 50,
        "breakpoint_percentile_threshold": 95,
        "embed_model": Settings.embed_model,
    }
    try:
        return SemanticSplitterNodeParser.from_defaults(**base_kwargs)
    except TypeError:
        base_kwargs.pop("embed_model", None)
        return SemanticSplitterNodeParser.from_defaults(**base_kwargs)

def build_node_views(docs: List[Document]) -> List[TextNode]:
    all_nodes: List[TextNode] = []

    sem_base_docs: List[Document]

    if not DISABLE_HIER:
        try:
            hier_parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=[MAX_CHUNK, TARGET_CHUNK, max(MIN_CHUNK, 128)]
            )
        except TypeError:
            hier_parser = HierarchicalNodeParser.from_defaults()

        hier_nodes = hier_parser.get_nodes_from_documents(docs)
        for n in hier_nodes:
            n.metadata = n.metadata or {}
            n.metadata["view"] = "hier"

        sem_base_docs = [
            Document(text=hn.get_text(), metadata={**(hn.metadata or {})})
            for hn in hier_nodes
        ]
    else:
    
        sem_base_docs = docs

    if not DISABLE_SEM:
        sem_parser = _make_semantic_splitter()
        for sn in sem_parser.get_nodes_from_documents(sem_base_docs):
            sn.metadata = sn.metadata or {}
            sn.metadata["view"] = "sem"
            all_nodes.append(sn)


    sentwin = SentenceWindowNodeParser.from_defaults(window_size=WINDOW_SIZE)
    sent_nodes = sentwin.get_nodes_from_documents(docs)
    for wn in sent_nodes:
        wn.metadata = wn.metadata or {}
        wn.metadata["view"] = "sentwin"
        all_nodes.append(wn)

    return all_nodes





# =========================================
#  indexing engine
# =========================================


def scan_docs() -> List[Document]:
    """
    Scan and normalize all allowed unstructured files in data/unstructured.
    Returns a list of Document objects.
    """
    if not UNSTRUCTURED_DIR.exists():
        return []

    files = [
        p for p in UNSTRUCTURED_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
    ]
    if not files:
        return []

    reader = SimpleDirectoryReader(
        input_files=[str(p) for p in files],
        filename_as_id=True,
    )
    docs_raw = reader.load_data()
    docs: List[Document] = []

    for d in docs_raw:
        md = dict(d.metadata or {})
        file_path = md.get("file_path") or md.get("source") or d.id_
        p = Path(file_path)

        # Fix source location if needed
        if not p.exists():
            candidates = list(UNSTRUCTURED_DIR.rglob(Path(file_path).name))
            if candidates:
                p = candidates[0]

        p = p.resolve()
        md["source_path"] = str(p)
        md["source"] = p.name
        md.setdefault("type", "unstructured_doc")

        doc_id = "file://" + str(p)
        docs.append(Document(text=d.text, metadata=md, doc_id=doc_id))

    return docs


def build_index_full() -> VectorStoreIndex:
    """
    Always full rebuild of FAISS + manifest.
    """
    docs = scan_docs()
    if not docs:
        raise RuntimeError("No documents available for indexing.")

    print(f"[Index-Full] Rebuilding from scratch: {len(docs)} docs...", flush=True)
    return _rebuild_all(docs, STORAGE_DIR)



def _rebuild_all(docs: List[Document], persist_dir: Path) -> VectorStoreIndex:
    with stage(f"Index: full rebuild from {len}(docs) docs (multi-view + embed + faiss)"):
        storage_context = _fresh_storage_context()
        nodes = build_node_views(docs)
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True,
        )

    # manifest
    with stage("Index: write manifest"):
        manifest: Dict[str, dict] = {}
        for doc in docs:
            sp = _resolve_source_to_path(doc.metadata or {})
            if sp and sp.exists():
                manifest[str(sp.resolve())] = {
                    "doc_id": doc.doc_id,
                    "fingerprint": file_fingerprint(sp),
                    "last_indexed": int(time.time()),
                }
        save_manifest(manifest)

    with stage("Index: persist to storage"):
        index.storage_context.persist(persist_dir=str(persist_dir))
    
    if CLEAN_CACHE_AFTER_BUILD:
        _clean_caches("after full rebuild")
    
        # ✅ Logging stats after full rebuild
    try:
        vector_count = storage_context.vector_store._faiss_index.ntotal if hasattr(storage_context.vector_store, "_faiss_index") else 0
        doc_count = len(docs)
        node_count = len(nodes)
        print(f"[LOG] Full rebuild summary -> Files: {doc_count} | Nodes: {node_count} | Vectors: {vector_count}", flush=True)
    except Exception as e:
        print(f"[LOG] Stats logging failed: {e}", flush=True)

    
    return index




def build_index_incremental() -> VectorStoreIndex:
    """
    Smart incremental update:
    - Load old index + manifest
    - Detect removed / changed docs
    - Refresh updated docs only
    - Fallback to full rebuild if needed
    """
    with stage("Docs: scan"):
        docs = scan_docs()
    if not docs:
        raise RuntimeError("No documents available.")

    with stage("Load index if exists"):
        index = _load_index(STORAGE_DIR)
        if index is None:
            return build_index_full()

    manifest = load_manifest()
    if not manifest:
        print("[Index-Inc] Manifest missing -> full rebuild.")
        return build_index_full()

    # Detect removed
    removed = [k for k in manifest if k not in [d.metadata["source_path"] for d in docs]]
    if removed:
        print(f"[Index-Inc] Removing {len(removed)} deleted files...")
        removed_ids = [manifest[p]["doc_id"] for p in removed if "doc_id" in manifest[p]]
        try:
            index.delete_ref_docs(ref_doc_ids=removed_ids, verbose=True)
        except Exception:
            print("[Index-Inc] Cannot delete -> full rebuild")
            return build_index_full()

    # Detect changed/new
    changed = []
    for d in docs:
        key = d.metadata["source_path"]
        fp = file_fingerprint(Path(key))
        if key not in manifest or manifest[key]["fingerprint"] != fp:
            changed.append(d)

    print(f"[Index-Inc] changed={len(changed)}/{len(docs)}")

    # Apply changes
    if changed:
        try:
            nodes = build_node_views(changed)
            index.refresh_ref_docs(documents=changed, nodes=nodes, verbose=True)
        except Exception:
            print("[Index-Inc] Cannot refresh -> full rebuild")
            return build_index_full()

    # Save new manifest
    new_manifest = {
        d.metadata["source_path"]: {
            "doc_id": d.doc_id,
            "fingerprint": file_fingerprint(Path(d.metadata["source_path"])),
            "last_indexed": int(time.time()),
        }
        for d in docs
    }
    save_manifest(new_manifest)
    # ✅ (3) Reload manifest into memory after refresh/write
    manifest = load_manifest()
    print(f"[Index-Inc] manifest reloaded; {len(manifest)} entries.", flush=True)

    index.storage_context.persist(persist_dir=str(STORAGE_DIR))

        # ✅ Logging stats after incremental update
    try:
        vector_count = index.vector_store._faiss_index.ntotal if hasattr(index.vector_store, "_faiss_index") else 0
        doc_count = len(docs)
        node_count = len(index.docstore.get_nodes()) if hasattr(index.docstore, "get_nodes") else len(index.docstore.get_keys())
        print(f"[LOG] Incremental summary -> Files: {doc_count} | Nodes: {node_count} | Vectors: {vector_count}", flush=True)
    except Exception as e:
        print(f"[LOG] Stats logging failed: {e}", flush=True)


    print("[Index-Inc] ✅ Done")
    return index


def _run_build(kind: str, fn):
    """Serialize builds across watcher/API/startup."""
    if INDEX_BUILD_BUSY.is_set():
        print(f"[INDEX] Busy; skip request: {kind}", flush=True)
        return

    print(f"[INDEX] ▶ start: {kind}", flush=True)
    INDEX_BUILD_BUSY.set()
    try:
        global INDEX
        with INDEX_LOCK:
            INDEX = fn()
        print(f"[INDEX] ✅ done: {kind}", flush=True)
    finally:
        INDEX_BUILD_BUSY.clear()




# =============================
# Manifest helpers
# =============================


def file_fingerprint(p: Path) -> str:
    h = hashlib.sha256()
    st = p.stat()
    h.update(str(p.resolve()).encode())
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()

def load_manifest() -> Dict[str, dict]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}

def save_manifest(m: Dict[str, dict]) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

def _resolve_source_to_path(md: dict) -> Optional[Path]:
    for k in ("source_path", "file_path", "filename", "file_name", "id"):
        v = md.get(k)
        if v:
            p = Path(v)
            if p.exists():
                return p
    name = md.get("source") or md.get("file_name") or md.get("filename")
    if name:
        for cand in UNSTRUCTURED_DIR.rglob(name):
            if cand.is_file():
                return cand
    return None



# =============================
# Index I/O (FAISS)
# =============================
def _fresh_storage_context() -> StorageContext:
    # Probe dim
    probe_vec = Settings.embed_model.get_query_embedding("dimension probe")
    dim = len(probe_vec)
    # HNSW index
    m = 32
    hnsw = faiss.IndexHNSWFlat(dim, m)
    hnsw.hnsw.efConstruction = 200
    hnsw.hnsw.efSearch = 64
    vs = FaissVectorStore(faiss_index=hnsw)
    docstore = SimpleDocumentStore()
    return StorageContext.from_defaults(vector_store=vs, docstore=docstore)

def _load_index(persist_dir: Path) -> Optional[VectorStoreIndex]:
    pd = Path(persist_dir)
    if not pd.exists():
        return None
    try:
        vs = FaissVectorStore.from_persist_dir(str(pd))
        sc = StorageContext.from_defaults(persist_dir=str(pd), vector_store=vs)
        return load_index_from_storage(sc)
    except Exception as e:
        print(f"[Load] failed to load persisted index: {e}", flush=True)
        return None

def load_index_on_startup(persist_dir: Path) -> Optional[VectorStoreIndex]:
    try:
        pd = Path(persist_dir)
        if not pd.exists():
            return None
        vs = FaissVectorStore.from_persist_dir(str(pd))
        sc = StorageContext.from_defaults(persist_dir=str(pd), vector_store=vs)
        return load_index_from_storage(sc)
    except Exception as e:
        print(f"[WARN] Index not loaded at startup: {e}", flush=True)
        return None


# =============================
# Pre-start build (optional) + VRAM cleanup
# =============================
try:
    if WATCH_DATA:
        # ✅ (4) Skip pre-start indexing when watcher is enabled
        with INDEX_LOCK:
            INDEX = _load_index(STORAGE_DIR)
        print("[BOOT] Watcher enabled; skipping pre-start build. Loaded persisted index if present.", flush=True)
    else:
        # Pre-start build only when watcher is OFF
        with INDEX_LOCK:
            try:
                INDEX = build_index_incremental()
                print("[BOOT] Pre-start indexing done (persisted).", flush=True)
            except RuntimeError as e:
                if "No documents available for indexing" in str(e):
                    INDEX = _load_index(STORAGE_DIR)
                    print("[BOOT] No docs; loaded existing index if present.", flush=True)
                else:
                    print(f"[BOOT] Pre-start indexing failed: {e}", flush=True)
                    INDEX = _load_index(STORAGE_DIR)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception:
    pass



# =============================
# Hybrid Retrieval + Query Pipeline 
# =============================
def reciprocal_rank_fusion(
    dense_nodes: List[NodeWithScore],
    bm25_nodes: List[NodeWithScore],
    k: int = SIMILARITY_TOP_K,
    weight_dense: float = WEIGHT_DENSE,
    weight_bm25: float = WEIGHT_BM25,
    rrf_k: int = 60,
) -> List[NodeWithScore]:
    dense_ranks = {n.node.node_id: i for i, n in enumerate(dense_nodes)}
    bm25_ranks  = {n.node.node_id: i for i, n in enumerate(bm25_nodes)}
    ids = set(list(dense_ranks.keys()) + list(bm25_ranks.keys()))

    node_lookup: Dict[str, NodeWithScore] = {}
    for n in dense_nodes:
        node_lookup[n.node.node_id] = n
    for n in bm25_nodes:
        node_lookup.setdefault(n.node.node_id, n)

    fused: List[Tuple[str, float, NodeWithScore]] = []
    for _id in ids:
        rd = dense_ranks.get(_id)
        rb = bm25_ranks.get(_id)
        score = 0.0
        if rd is not None: score += weight_dense * (1.0 / (rrf_k + rd + 1))
        if rb is not None: score += weight_bm25 * (1.0 / (rrf_k + rb + 1))
        fused.append((_id, score, node_lookup[_id]))

    fused.sort(key=lambda x: x[1], reverse=True)
    return [trip[2] for trip in fused[:k]]

class HybridRetriever:
    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self.have_bm25 = HAVE_BM25_NATIVE
        self._bm25: Optional[BM25Retriever] = None
        if self.have_bm25:
            try:
                nodes: List[TextNode] = []
                for nid in self.index.docstore.get_keys():  # type: ignore
                    node = self.index.docstore.get_node(nid)  # type: ignore
                    if isinstance(node, TextNode):
                        nodes.append(node)
                self._bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=SIMILARITY_TOP_K)
                print("[HYBRID] Using BM25Retriever.", flush=True)
            except Exception as e:
                print(f"[HYBRID] BM25 init failed; continuing dense-only: {e}", flush=True)
                self.have_bm25 = False

    def retrieve(self, query: str) -> List[NodeWithScore]:
        dense = self.index.as_retriever(
            similarity_top_k=SIMILARITY_TOP_K,
            vector_store_query_mode="mmr",
            vector_store_kwargs={"mmr_threshold": MMR_LAMBDA, "fetch_k": FETCH_K},
        ).retrieve(query)

        bm25_nodes: List[NodeWithScore] = []
        if self._bm25 is not None:
            bm25_nodes = self._bm25.retrieve(query)

        if bm25_nodes:
            return reciprocal_rank_fusion(dense, bm25_nodes, k=SIMILARITY_TOP_K,
                                          weight_dense=WEIGHT_DENSE, weight_bm25=WEIGHT_BM25, rrf_k=60)
        return dense

def build_query_pipeline():
    postprocessors = []
    # if HAVE_ST_RERANK:
    #     try:
    #         postprocessors.append(SentenceTransformerRerank(top_n=RERANK_TOP_N))
    #         print(f"[QUERY] SentenceTransformerRerank enabled (top_n={RERANK_TOP_N}).", flush=True)
    #     except Exception as e:
    #         print(f"[QUERY] Rerank init failed: {e}", flush=True)

    try:
        postprocessors.append(LongContextReorder())
    except Exception:
        pass

    postprocessors.append(MetadataReplacementPostProcessor(target_metadata_key="window"))
    synth = get_response_synthesizer(response_mode=ResponseMode.COMPACT)
    return postprocessors, synth

def run_query(index: VectorStoreIndex, query: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    postprocessors, synth = build_query_pipeline()

    retriever = HybridRetriever(index)
    t_retr0 = time.perf_counter()
    nodes = retriever.retrieve(query)
    t_retr = (time.perf_counter() - t_retr0) * 1000.0

    t_post0 = time.perf_counter()
    for pp in postprocessors:
        try:
            nodes = pp.postprocess_nodes(nodes, query_str=query)
        except TypeError:
            nodes = pp.postprocess_nodes(nodes)
    t_post = (time.perf_counter() - t_post0) * 1000.0

    t_llm0 = time.perf_counter()
    response = synth.synthesize(query, nodes)
    t_llm = (time.perf_counter() - t_llm0) * 1000.0

    total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "answer": str(response),
        "nodes": nodes, 
        "metrics": {
            "retrieval_ms": round(t_retr, 2),
            "postprocess_ms": round(t_post, 2),
            "llm_ms": round(t_llm, 2),
            "total_ms": round(total_ms, 2),
            "top_k": SIMILARITY_TOP_K,
            "fetch_k": FETCH_K,
            "mmr_lambda": MMR_LAMBDA,
            "fusion_weights": {"dense": WEIGHT_DENSE, "bm25": WEIGHT_BM25},
        },
    }



# =============================
# FastAPI App
# =============================
app = FastAPI(title="RAG API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Schemas ----------
class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = SIMILARITY_TOP_K
    chat_history: Optional[List[ChatTurn]] = None

class Source(BaseModel):
    doc_id: str
    score: float
    metadata: Dict[str, Any] = {}
    preview: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    latency_ms: int

class ModelIn(BaseModel):
    model: str

def to_sources(nodes: List[NodeWithScore]) -> List[Source]:
    out: List[Source] = []
    for n in nodes or []:
        md = dict(n.node.metadata or {})
        preview = (n.node.get_content() or "")[:300]
        out.append(Source(
            doc_id=str(md.get("doc_id") or md.get("file_name") or md.get("filename") or n.node.node_id),
            score=float(n.score or 0.0),
            metadata=md,
            preview=preview,
        ))
    return out

# ---------- Public ----------
@app.get("/health")
def health():
    return {"status": "ok", "model": DEFAULT_MODEL, "embed_device": EMB_DEVICE}

@app.get("/v1/health")
def v1_health():
    return {"ok": True, "index_ready": INDEX is not None}



@app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(require_user)])
def query(req: QueryRequest):
    start = time.time()
    if INDEX is None:
        raise HTTPException(status_code=503, detail="Index not ready")
    with INDEX_LOCK:
        result = run_query(INDEX, req.question)
        nodes = result["nodes"]
        answer = result["answer"]
    elapsed = int((time.time() - start) * 1000)
    return QueryResponse(
        answer=answer,
        sources=to_sources(nodes),
        latency_ms=elapsed
    )




# @app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(require_user)])
# def query(req: QueryRequest):
#     start = time.time()
#     print(f"[QUERY] User query: {req.question}", flush=True)

#     if INDEX is None:
#         raise HTTPException(status_code=503, detail="Index not ready")

#     with INDEX_LOCK:
#         result = run_query(INDEX, req.question)
#         nodes = result["nodes"]
#         answer = result["answer"]

#     elapsed = int((time.time() - start) * 1000)

#     print(f"[RESULT] {answer[:200]}...", flush=True)  # طباعة أول 200 حرف فقط لتجنب طول النص

#     return QueryResponse(
#         answer=answer,
#         sources=to_sources(nodes),
#         latency_ms=elapsed
#     )

# ---------- Files / Admin ----------
def _bucket_dir(bucket: Bucket) -> Path:
    return STRUCTURED_DIR if bucket == "structured" else UNSTRUCTURED_DIR

def _safe_join(base: Path, name: str) -> Path:
    dest = (base / os.path.basename(name)).resolve()
    if not str(dest).startswith(str(base.resolve()) + os.sep):
        raise HTTPException(status_code=400, detail="Invalid path")
    return dest

def _ext_ok(filename: str) -> bool:
    return Path(filename.lower()).suffix in ALLOWED_EXTS


@app.post("/v1/reindex", dependencies=[Depends(require_admin)])
def reindex(full: bool = False):
    # ✅ (5) serialize builds via busy flag
    if full:
        _run_build("api:full", build_index_full)
    else:
        _run_build("api:incremental", build_index_incremental)
    return {"ok": True, "full": full, "busy": INDEX_BUILD_BUSY.is_set()}

@app.get("/v1/models", dependencies=[Depends(require_user)])
async def list_models():
    url = f"{OLLAMA_URL.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            r.raise_for_status()
            tags = [t.get("name") for t in r.json().get("models", []) if t.get("name")]
    except Exception:
        tags = [DEFAULT_MODEL]
    return {"current": DEFAULT_MODEL, "available": tags}


# =============================
# Startup
# =============================


@app.on_event("startup")
def _startup():
    if INDEX:
        print("[BOOT] Index preloaded; startup does nothing.", flush=True)
    else:
        print("[BOOT] No index loaded before startup. API will 503 until you build manually.", flush=True)

    if WATCH_DATA:
        t = threading.Thread(target=_watch_loop, name="data-watcher", daemon=True)
        t.start()



# =============================
# Uvicorn
# =============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8003")))
 