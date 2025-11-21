"""Single-file FastAPI app exposing feature reception pipeline APIs."""
from __future__ import annotations

import hashlib
import logging
import json
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Sequence
from uuid import UUID, uuid4

import asyncpg
import httpx
import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

ALLOWED_CLUSTER_TYPES = [
    "Adoption",
    "Integrations",
    "Analytics",
    "UI/UX",
    "Configuration",
    "Access Control",
    "Competitor Reference",
]
METRIC_KEY = "feature_reception"


class Settings(BaseSettings):
    """Environment configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str | None = None

    # Embedding model settings
    # Model is downloaded once to ~/.cache/torch/sentence_transformers/ and reused
    # Set to empty string "" to disable model and use deterministic fallback
    embed_model_name: str = "BAAI/bge-large-en-v1.5"
    embedding_dimension: int = 1024

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_endpoint: str = "https://generativelanguage.googleapis.com/v1beta"

    merge_similarity_threshold: float = 0.90
    recluster_distance_threshold: float = 0.30
    recluster_min_points: int = 2


settings = Settings()

pool: asyncpg.Pool | None = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("arali_api")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Manage the asyncpg pool lifecycle."""
    logger.info("Starting up application and initializing database pool.")
    global pool
    if settings.database_url:
        try:
            pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=10)
            logger.info("Database pool created successfully.")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
    yield
    if pool:
        await pool.close()
        pool = None
        logger.info("Database pool closed.")


async def get_connection() -> AsyncIterator[asyncpg.Connection]:
    if pool is None:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    async with pool.acquire() as connection:
        yield connection


def normalize(vector: Sequence[float]) -> list[float]:
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    return arr.tolist() if norm == 0 else (arr / norm).tolist()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def normalized_mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    arr = np.asarray(vectors, dtype=float)
    if arr.size == 0:
        return []
    return normalize(arr.mean(axis=0))


def incremental_centroid(
    centroid: Sequence[float],
    size: int,
    new_vectors: Sequence[Sequence[float]],
) -> tuple[list[float], int]:
    if not new_vectors:
        return list(centroid), size
    arr_old = np.asarray(centroid, dtype=float) * size
    arr_new = np.asarray(new_vectors, dtype=float).sum(axis=0)
    total = arr_old + arr_new
    new_size = size + len(new_vectors)
    return normalize(total / new_size), new_size


class EmbeddingClient:
    """Text embedding client; uses local model or deterministic fallback."""

    def __init__(self) -> None:
        self.dimension = settings.embedding_dimension
        self.model_name = settings.embed_model_name
        self.model = None
        self._keep_loaded = False  # Flag to keep model loaded across multiple calls
        self._lock = asyncio.Lock()  # Protect model loading/unloading
        self._active_users = 0  # Track concurrent users

    async def _load_model(self) -> None:
        """Load the model into memory (thread-safe)."""
        async with self._lock:
            if self.model is not None:
                return  # Already loaded
            if self.model_name:
                try:
                    logger.info(f"Loading embedding model: {self.model_name}")
                    self.model = SentenceTransformer(self.model_name)
                    logger.info("Embedding model loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load local model {self.model_name}: {e}")
                    raise

    async def _unload_model(self) -> None:
        """Unload model from memory to free RAM (thread-safe)."""
        async with self._lock:
            # Only unload if no one is using it
            if self._active_users > 0:
                logger.debug(f"Not unloading model: {self._active_users} active users")
                return
            if self.model is not None:
                logger.info("Unloading embedding model to free memory")
                del self.model
                self.model = None
                # Force garbage collection to immediately release memory
                import gc
                gc.collect()
                logger.info("Embedding model unloaded")

    def keep_loaded(self):
        """Context manager to keep model loaded across multiple embed calls."""
        class KeepLoadedContext:
            def __init__(self, client: EmbeddingClient):
                self.client = client
                
            def __enter__(self):
                # Note: This is a synchronous context manager used by synchronous code
                # The _keep_loaded flag is read atomically in embed() under lock
                self.client._keep_loaded = True
                return self.client
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.client._keep_loaded = False
                # Schedule unload asynchronously
                # By the time this runs, all embed() calls will have captured the old flag value
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create a task to unload after a small delay to let pending operations complete
                        async def delayed_unload():
                            await asyncio.sleep(0.1)  # Let pending operations finish
                            await self.client._unload_model()
                        asyncio.create_task(delayed_unload())
                    else:
                        loop.run_until_complete(self.client._unload_model())
                except Exception as e:
                    logger.error(f"Error unloading model: {e}")
                
        return KeepLoadedContext(self)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Embed texts. By default, loads model, processes, then unloads to free memory.
        Use keep_loaded() context manager to keep model loaded across multiple calls.
        Thread-safe: supports concurrent requests.
        """
        if self.model_name:
            # Load model and increment active users atomically
            await self._load_model()
            
            async with self._lock:
                self._active_users += 1
                keep_loaded_snapshot = self._keep_loaded  # Capture flag under lock
            
            try:
                # Process embeddings (no lock needed - model.encode is thread-safe)
                logger.debug(f"Generating embeddings for {len(texts)} texts")
                vectors = self.model.encode(list(texts), normalize_embeddings=True)
                result = [vec.tolist() for vec in vectors]
                
                # Decrement active users after using the model
                async with self._lock:
                    self._active_users -= 1
                
                # Unload only if not in keep_loaded context and no active users
                if not keep_loaded_snapshot:
                    await self._unload_model()
                
                return result
            except Exception as e:
                logger.error(f"Error during embedding: {e}")
                # Ensure we decrement counter even on error
                async with self._lock:
                    self._active_users = max(0, self._active_users - 1)
                # Try to unload if not in keep_loaded context
                if not keep_loaded_snapshot:
                    await self._unload_model()
                raise
        
        # Fallback to deterministic vectors if no model configured
        logger.debug(f"Using deterministic embeddings for {len(texts)} texts")
        return [self._deterministic_vector(text) for text in texts]

    def _deterministic_vector(self, text: str) -> list[float]:
        seed_bytes = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
        seed = int.from_bytes(seed_bytes, "big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension)
        return normalize(vec.tolist())


class ClusterLabeler:
    """LLM-backed or rule-based cluster labeling."""

    def __init__(self) -> None:
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_model
        self.endpoint = settings.gemini_endpoint.rstrip("/")

    async def label(self, payload: dict) -> dict:
        cluster_id = payload.get("cluster_id")
        logger.info(f"Generating label for cluster {cluster_id}")
        insights = payload["insights"]
        if not self.api_key:
            logger.warning("Gemini API key not found, falling back to rule-based labeling.")
            text = " ".join(f"{item['title']} {item['summary']}" for item in insights).lower()
            detected = self._detect_type(text)
            description = " ".join(item["summary"] for item in insights if item["summary"]).strip()
            return {
                "name": insights[0]["title"][:60] if insights else "Cluster",
                "description": description or "Cluster generated from feature feedback.",
                "type": detected,
            }

        prompt_lines = [
            "Cluster context:",
            f"enterprise_id: {payload['enterprise_id']}",
            f"product_id: {payload['product_id']}",
            f"metric_key: {payload['metric_key']}",
            "",
            "Insights (title + summary):",
        ]
        for idx, insight in enumerate(insights, start=1):
            prompt_lines.append(f"{idx}) Title: \"{insight['title']}\"")
            prompt_lines.append(f"   Summary: \"{insight['summary']}\"")
            prompt_lines.append("")
        prompt_lines.append(
            "Return strict JSON with keys name, description, type "
            f"(type in {', '.join(ALLOWED_CLUSTER_TYPES)})."
        )
        prompt = "\n".join(prompt_lines)
        body = {
            "system_instruction": {
                "parts": [
                    {
                        "text": (
                            "You are a product manager working at a B2B company. "
                            "Summarize the dominant theme, craft a differentiated 8-10 word name."
                            "with crisp product-style wording (avoid generic titles like 'Feature Updates'), "
                            "concise description, and assign a valid type."
                            "Title should be descriptive, specific, and easy to scan, anyone reading should get the meaning in a single glance."
                            "Description should be concise and cover the main theme and should be around 30-50 words."
                        )
                    }
                ]
            },
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        url = f"{self.endpoint}/models/{self.model}:generateContent?key={self.api_key}"
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=body)
            if resp.status_code >= 400:
                logger.error(f"Gemini API error: {resp.status_code} - {resp.text}")
                raise HTTPException(status_code=502, detail=resp.text)
        data = resp.json()
        candidates = data.get("candidates") or []
        parts = (candidates[0].get("content", {}).get("parts") if candidates else None) or []
        text = parts[0].get("text", "") if parts else ""
        if not text:
            raise HTTPException(status_code=502, detail="Empty Gemini response")
        # Clean up potential markdown formatting
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        try:
            parsed = json.loads(cleaned_text)
        except json.JSONDecodeError as exc:
            logger.error(f"Gemini JSON parse error. Text was: {text}")
            raise HTTPException(status_code=502, detail="Gemini JSON parse error") from exc

        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], dict):
                parsed = parsed[0]
            else:
                logger.error(f"Gemini returned a list, expected dict. Parsed: {parsed}")
                raise HTTPException(status_code=502, detail="Gemini returned unexpected JSON structure")

        if not isinstance(parsed, dict):
            logger.error(f"Gemini returned {type(parsed)}, expected dict. Parsed: {parsed}")
            raise HTTPException(status_code=502, detail="Gemini returned unexpected JSON structure")

        return {
            "name": str(parsed.get("name", "")).strip(),
            "description": str(parsed.get("description", "")).strip(),
            "type": str(parsed.get("type", "")).strip() or None,
        }

    def _detect_type(self, text: str) -> str | None:
        keyword_map = {
            "Adoption": ["adoption", "usage", "rollout"],
            "Integrations": ["integration", "api", "crm"],
            "Analytics": ["dashboard", "report", "analytics"],
            "UI/UX": ["ui", "ux", "interface", "design"],
            "Configuration": ["config", "setting", "workflow"],
            "Access Control": ["permission", "role", "access"],
            "Competitor Reference": ["competitor", "alternative", "vs"],
        }
        for type_name, keywords in keyword_map.items():
            if any(keyword in text for keyword in keywords):
                return type_name
        return None


embedding_client = EmbeddingClient()
labeler = ClusterLabeler()


def build_feature_text(details: dict | str | None) -> str:
    """Create a concise feature text from details.

    * ``details`` may be a dict (as stored in the DB), a JSON string, or ``None``.
    * If a string is provided we attempt to ``json.loads`` it; on failure we fall back to an empty dict.
    """
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = {}
    # Ensure we have a dict (or empty dict) from here on
    title = str((details or {}).get("feature_title", "")).strip()
    summary = str((details or {}).get("feature_summary", "")).strip()
    if title and summary:
        return f"{title} — {summary}"
    return title or summary


def should_assign_cluster(size: int, sim: float) -> bool:
    # Lowered thresholds to help initial cluster formation
    return (size >= 5 and sim >= 0.80) or (size < 5 and sim >= 0.82)


def _vector_from_db(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip("[]")
        if not cleaned:
            return []
        return [float(v) for v in cleaned.split(",")]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if isinstance(value, memoryview):
        data = value.tolist() if hasattr(value, "tolist") else value
        return [float(v) for v in data]
    return list(value)


def _vector_to_db(vector: Sequence[float]) -> str:
    return str(list(vector))


@dataclass
class InsightCluster:
    id: str
    enterprise_id: UUID
    product_id: str | None
    metric_key: str
    centroid: list[float]
    size: int
    created_at: Any


class FeatureReceptionPipeline:
    """Encapsulates all SQL and orchestration for APIs 1-5."""

    def __init__(self, connection: asyncpg.Connection) -> None:
        self.conn = connection

    async def run_online(self, insight_id: UUID) -> str | None:
        logger.info(f"Processing online insight: {insight_id}")
        insight = await self._fetch_insight(insight_id)
        if not insight:
            logger.warning(f"Insight {insight_id} not found.")
            raise HTTPException(status_code=404, detail="Insight not found")
        if insight["cluster_id"]:
            logger.info(f"Insight {insight_id} already assigned to cluster {insight['cluster_id']}")
            return insight["cluster_id"]
        if insight["metric_key"] != METRIC_KEY:
            logger.warning(f"Unsupported metric_key for insight {insight_id}: {insight['metric_key']}")
            raise HTTPException(status_code=400, detail="Unsupported metric_key")
        embedding = await self._ensure_embedding(insight)
        clusters = await self._fetch_clusters(insight["enterprise_id"], insight["product_id"])
        
        best = None
        if clusters:
            best = self._select_cluster(embedding, clusters)
            
        if not best:
            logger.info(f"No suitable cluster found for insight {insight_id}. It will remain unclustered until re-clustering.")
            return None
            
        logger.info(f"Assigning insight {insight_id} to cluster {best.id}")
        await self.conn.execute(
            "UPDATE meeting_insights SET cluster_id = $2 WHERE id = $1",
            insight_id,
            best.id,
        )
        new_centroid, new_size = incremental_centroid(best.centroid, best.size, [embedding])
        await self._update_cluster_vector(best.id, new_centroid, new_size)
        return best.id

    async def run_daily(self, batch_size: int) -> dict:
        logger.info(f"Starting daily batch processing with batch_size={batch_size}")
        
        # Keep model loaded for entire daily job to avoid repeated load/unload
        with embedding_client.keep_loaded():
            embedded = await self._backfill_embeddings(batch_size)
            logger.info(f"Backfilled embeddings for {embedded} insights.")
            assigned, clusters_updated = await self._assign_clusters(batch_size)
            logger.info(f"Assigned {assigned} insights to clusters, updated {clusters_updated} clusters.")
        
        logger.info("Daily batch processing complete.")
        return {"embedded": embedded, "assigned": assigned, "clusters_updated": clusters_updated}

    async def run_recluster(self, target_enterprise_id: UUID | None = None) -> list[dict]:
        logger.info("Starting weekly re-clustering process.")
        results: list[dict] = []
        pairs = await self._list_pairs(target_enterprise_id)
        for enterprise_id, product_id in pairs:
            logger.info(f"Re-clustering for enterprise {enterprise_id}, product {product_id}")
            embeddings = await self._fetch_embeddings_for_pair(enterprise_id, product_id)
            if len(embeddings) < settings.recluster_min_points:
                logger.info(f"Skipping {enterprise_id}/{product_id}: not enough data points ({len(embeddings)})")
                continue
            insight_ids, vectors = zip(*embeddings)
            arr = np.asarray(vectors, dtype=float)
            clustering = AgglomerativeClustering(
                metric="cosine",
                linkage="average",
                distance_threshold=settings.recluster_distance_threshold,
                n_clusters=None,
            )
            labels = clustering.fit_predict(arr)
            clusters: list[tuple[str, list[float], int, list[UUID]]] = []
            for label in set(labels):
                idxs = [idx for idx, lbl in enumerate(labels) if lbl == label]
                members = [insight_ids[idx] for idx in idxs]
                member_vectors = [vectors[idx] for idx in idxs]
                centroid = normalized_mean(member_vectors)
                clusters.append((str(uuid4()), centroid, len(members), members))
            logger.info(f"Created {len(clusters)} clusters for {enterprise_id}/{product_id}")
            await self._replace_clusters(enterprise_id, product_id, clusters)
            results.append(
                {
                    "enterprise_id": enterprise_id,
                    "product_id": product_id,
                    "clusters_created": len(clusters),
                }
            )
        return results

    async def run_merge(self, target_enterprise_id: UUID | None = None) -> list[dict]:
        logger.info("Starting weekly cluster merge process.")
        results: list[dict] = []
        pairs = await self._list_cluster_pairs(target_enterprise_id)
        for enterprise_id, product_id in pairs:
            logger.info(f"Checking merges for enterprise {enterprise_id}, product {product_id}")
            merges: list[tuple[str, str]] = []
            merge_iteration = 0
            
            # Keep merging until no more merges are found
            while True:
                merge_iteration += 1
                clusters = await self._fetch_clusters(enterprise_id, product_id)
                found_merge = False
                
                for idx, cluster in enumerate(clusters):
                    if found_merge:
                        break
                    for other in clusters[idx + 1 :]:
                        sim = cosine_similarity(cluster.centroid, other.centroid)
                        if sim < settings.merge_similarity_threshold:
                            continue
                        winner, loser = self._choose_winner(cluster, other)
                        new_centroid, new_size = self._merge_centroids(winner, loser)
                        await self.conn.execute(
                            "UPDATE meeting_insights SET cluster_id = $1 WHERE cluster_id = $2",
                            winner.id,
                            loser.id,
                        )
                        await self.conn.execute("DELETE FROM insight_clusters WHERE id = $1", loser.id)
                        await self._update_cluster_vector(winner.id, new_centroid, new_size)
                        merges.append((winner.id, loser.id))
                        logger.info(f"Merged cluster {loser.id} into {winner.id} (similarity: {sim:.3f})")
                        found_merge = True
                        break
                
                if not found_merge:
                    break
                    
            if merges:
                logger.info(f"Performed {len(merges)} merges for {enterprise_id}/{product_id} over {merge_iteration} iterations")
                results.append(
                    {
                        "enterprise_id": enterprise_id,
                        "product_id": product_id,
                        "merges": merges,
                    }
                )
        return results

    async def run_labeling(self, target_enterprise_id: UUID | None = None) -> int:
        logger.info("Starting weekly cluster labeling process.")
        total = 0
        pairs = await self._list_cluster_pairs(target_enterprise_id)
        for enterprise_id, product_id in pairs:
            clusters = await self._fetch_clusters(enterprise_id, product_id)
            for cluster in clusters:
                insights = await self._fetch_cluster_insights(cluster.id)
                if not insights:
                    continue
                payload = {
                    "cluster_id": cluster.id,
                    "enterprise_id": str(enterprise_id),
                    "product_id": cluster.product_id,
                    "metric_key": cluster.metric_key,
                    "insights": [
                        self._insight_prompt_record(row["details_json"])
                        for row in insights
                    ],
                }
                label = await labeler.label(payload)
                label_type = self._validate_type(label.get("type"))
                await self.conn.execute(
                    """
                    UPDATE insight_clusters
                       SET name = $2,
                           description = $3,
                           type = $4,
                           updated_at = NOW()
                     WHERE id = $1
                    """,
                    cluster.id,
                    label.get("name"),
                    label.get("description"),
                    label_type,
                )
                total += 1
        return total

    async def _fetch_insight(self, insight_id: UUID) -> dict | None:
        query = """
            SELECT id, enterprise_id, product_id, metric_key, details_json, embedding, cluster_id
            FROM meeting_insights
            WHERE id = $1
        """
        row = await self.conn.fetchrow(query, insight_id)
        if not row:
            return None
        return {
            "id": row["id"],
            "enterprise_id": row["enterprise_id"],
            "product_id": row["product_id"],
            "metric_key": row["metric_key"],
            "details_json": row["details_json"],
            "embedding": _vector_from_db(row["embedding"]),
            "cluster_id": row["cluster_id"],
        }

    async def _ensure_embedding(self, insight: dict) -> list[float]:
        if insight["embedding"]:
            return insight["embedding"]
        text = build_feature_text(insight["details_json"])
        vector = (await embedding_client.embed([text]))[0]
        await self.conn.execute(
            "UPDATE meeting_insights SET embedding = $2 WHERE id = $1",
            insight["id"],
            _vector_to_db(vector),
        )
        return vector

    async def _fetch_clusters(self, enterprise_id: UUID, product_id: str | None) -> list[InsightCluster]:
        query = """
            SELECT id, enterprise_id, product_id, metric_key, centroid, size, created_at
            FROM insight_clusters
            WHERE enterprise_id = $1
              AND metric_key = $2
              AND (product_id IS NOT DISTINCT FROM $3)
        """
        rows = await self.conn.fetch(query, enterprise_id, METRIC_KEY, product_id)
        return [
            InsightCluster(
                id=row["id"],
                enterprise_id=row["enterprise_id"],
                product_id=row["product_id"],
                metric_key=row["metric_key"],
                centroid=_vector_from_db(row["centroid"]) or [],
                size=row["size"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def _select_cluster(self, embedding: Sequence[float], clusters: Sequence[InsightCluster]) -> InsightCluster | None:
        best: InsightCluster | None = None
        best_sim = -1.0
        for cluster in clusters:
            if not cluster.centroid:
                continue
            sim = cosine_similarity(embedding, cluster.centroid)
            if sim > best_sim:
                best_sim = sim
                best = cluster
        if best and should_assign_cluster(best.size, best_sim):
            return best
        return None

    async def _backfill_embeddings(self, batch_size: int) -> int:
        total = 0
        batch_num = 0
        while True:
            query = """
                SELECT id, details_json
                FROM meeting_insights
                WHERE metric_key = $1
                  AND embedding IS NULL
                ORDER BY created_at
                LIMIT $2
            """
            rows = await self.conn.fetch(query, METRIC_KEY, batch_size)
            if not rows:
                break
            batch_num += 1
            logger.info(f"Backfilling embeddings: batch {batch_num}, processing {len(rows)} insights")
            texts = [build_feature_text(row["details_json"]) for row in rows]
            vectors = await embedding_client.embed(texts)
            for row, vector in zip(rows, vectors, strict=True):
                await self.conn.execute(
                    "UPDATE meeting_insights SET embedding = $2 WHERE id = $1",
                    row["id"],
                    _vector_to_db(vector),
                )
                total += 1
            # Clear references to help garbage collection
            del texts, vectors
        logger.info(f"Backfilling complete: processed {total} insights in {batch_num} batches")
        return total

    async def _assign_clusters(self, batch_size: int) -> tuple[int, int]:
        assigned_total = 0
        updated_clusters = 0
        cluster_cache: dict[tuple[UUID, str | None], list[InsightCluster]] = {}
        cluster_lookup: dict[str, InsightCluster] = {}
        batch_num = 0

        while True:
            query = """
                SELECT id, enterprise_id, product_id, embedding
                FROM meeting_insights
                WHERE metric_key = $1
                  AND cluster_id IS NULL
                  AND embedding IS NOT NULL
                ORDER BY created_at
                LIMIT $2
            """
            rows = await self.conn.fetch(query, METRIC_KEY, batch_size)
            if not rows:
                break
            batch_num += 1
            logger.info(f"Assigning clusters: batch {batch_num}, processing {len(rows)} insights")
            assignments: list[tuple[UUID, str]] = []
            vectors_by_cluster: dict[str, list[list[float]]] = {}
            for row in rows:
                key = (row["enterprise_id"], row["product_id"])
                clusters = cluster_cache.get(key)
                if clusters is None:
                    clusters = await self._fetch_clusters(*key)
                    cluster_cache[key] = clusters
                    for cluster in clusters:
                        cluster_lookup[cluster.id] = cluster
                if not clusters:
                    continue
                embedding = _vector_from_db(row["embedding"])
                if not embedding:
                    continue
                best = self._select_cluster(embedding, clusters)
                if not best:
                    continue
                assignments.append((row["id"], best.id))
                vectors_by_cluster.setdefault(best.id, []).append(embedding)
            if assignments:
                await self.conn.executemany(
                    "UPDATE meeting_insights SET cluster_id = $2 WHERE id = $1",
                    assignments,
                )
                assigned_total += len(assignments)
                for cluster_id, vectors in vectors_by_cluster.items():
                    cluster = cluster_lookup.get(cluster_id)
                    if not cluster:
                        continue
                    new_centroid, new_size = incremental_centroid(cluster.centroid, cluster.size, vectors)
                    await self._update_cluster_vector(cluster_id, new_centroid, new_size)
                    cluster.centroid = new_centroid
                    cluster.size = new_size
                    updated_clusters += 1
            else:
                # No assignments made in this batch - stop processing to avoid infinite loop
                # (insights remain unclustered because no suitable clusters exist or similarity is too low)
                logger.info(f"No assignments made in batch {batch_num}, stopping cluster assignment")
                break

        logger.info(f"Cluster assignment complete: assigned {assigned_total} insights, updated {updated_clusters} clusters in {batch_num} batches")
        return assigned_total, updated_clusters

    async def _update_cluster_vector(self, cluster_id: str, centroid: Sequence[float], size: int) -> None:
        await self.conn.execute(
            """
            UPDATE insight_clusters
               SET centroid = $2,
                   size = $3,
                   updated_at = NOW()
             WHERE id = $1
            """,
            cluster_id,
            _vector_to_db(centroid),
            size,
        )

    async def _list_pairs(self, enterprise_id: UUID | None = None) -> list[tuple[UUID, str | None]]:
        query = """
            SELECT DISTINCT enterprise_id, product_id
            FROM meeting_insights
            WHERE metric_key = $1
              AND embedding IS NOT NULL
        """
        params: list[Any] = [METRIC_KEY]
        if enterprise_id is not None:
            query += " AND enterprise_id = $2"
            params.append(enterprise_id)
        rows = await self.conn.fetch(query, *params)
        return [(row["enterprise_id"], row["product_id"]) for row in rows]

    async def _fetch_embeddings_for_pair(
        self,
        enterprise_id: UUID,
        product_id: str | None,
    ) -> list[tuple[UUID, list[float]]]:
        rows = await self.conn.fetch(
            """
            SELECT id, embedding
            FROM meeting_insights
            WHERE enterprise_id = $1
              AND metric_key = $2
              AND (product_id IS NOT DISTINCT FROM $3)
              AND embedding IS NOT NULL
            """,
            enterprise_id,
            METRIC_KEY,
            product_id,
        )
        return [(row["id"], _vector_from_db(row["embedding"]) or []) for row in rows]

    async def _replace_clusters(
        self,
        enterprise_id: UUID,
        product_id: str | None,
        clusters: Sequence[tuple[str, list[float], int, Sequence[UUID]]],
    ) -> None:
        async with self.conn.transaction():
            await self.conn.execute(
                """
                DELETE FROM insight_clusters
                WHERE enterprise_id = $1
                  AND metric_key = $2
                  AND (product_id IS NOT DISTINCT FROM $3)
                """,
                enterprise_id,
                METRIC_KEY,
                product_id,
            )
            for cluster_id, centroid, size, _ in clusters:
                await self.conn.execute(
                    """
                    INSERT INTO insight_clusters (id, enterprise_id, metric_key, product_id, centroid, size, name, description, type)
                    VALUES ($1, $2, $3, $4, $5, $6, NULL, NULL, NULL)
                    """,
                    cluster_id,
                    enterprise_id,
                    METRIC_KEY,
                    product_id,
                    _vector_to_db(centroid),
                    size,
                )
            assignments = [
                (insight_id, cluster_id)
                for cluster_id, _, _, insight_ids in clusters
                for insight_id in insight_ids
            ]
            await self.conn.executemany(
                "UPDATE meeting_insights SET cluster_id = $2 WHERE id = $1",
                assignments,
            )

    async def _list_cluster_pairs(self, enterprise_id: UUID | None = None) -> list[tuple[UUID, str | None]]:
        query = """
            SELECT DISTINCT enterprise_id, product_id
            FROM insight_clusters
            WHERE metric_key = $1
        """
        params: list[Any] = [METRIC_KEY]
        if enterprise_id is not None:
            query += " AND enterprise_id = $2"
            params.append(enterprise_id)
        rows = await self.conn.fetch(query, *params)
        return [(row["enterprise_id"], row["product_id"]) for row in rows]

    def _choose_winner(self, a: InsightCluster, b: InsightCluster) -> tuple[InsightCluster, InsightCluster]:
        if a.size > b.size:
            return a, b
        if b.size > a.size:
            return b, a
        return (a, b) if a.created_at <= b.created_at else (b, a)

    def _merge_centroids(self, winner: InsightCluster, loser: InsightCluster) -> tuple[list[float], int]:
        arr = (np.asarray(winner.centroid, dtype=float) * winner.size) + (
            np.asarray(loser.centroid, dtype=float) * loser.size
        )
        total = winner.size + loser.size
        return normalize(arr / total), total

    async def _fetch_cluster_insights(self, cluster_id: str) -> list[dict]:
        rows = await self.conn.fetch(
            """
            SELECT id, details_json
            FROM meeting_insights
            WHERE cluster_id = $1
              AND metric_key = $2
            """,
            cluster_id,
            METRIC_KEY,
        )
        return [dict(row) for row in rows]

    def _insight_prompt_record(self, details: dict | str | None) -> dict:
        payload = details or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}
        
        title = str(payload.get("feature_title") or "").strip() or "Untitled Insight"
        summary = str(payload.get("feature_summary") or "").strip()
        return {"title": title, "summary": summary}

    def _validate_type(self, type_value: str | None) -> str | None:
        if not type_value:
            return None
        normalized = type_value.strip().lower()
        for allowed in ALLOWED_CLUSTER_TYPES:
            if normalized == allowed.lower():
                return allowed
        return None


class OnlineTaskRequest(BaseModel):
    insight_id: UUID


class OnlineTaskResponse(BaseModel):
    cluster_id: str | None


class DailyTaskRequest(BaseModel):
    batch_size: int = Field(200, ge=1, le=2000)


class DailyTaskResponse(BaseModel):
    message: str
    embedded: int
    assigned: int
    clusters_updated: int


class WeeklyTaskRequest(BaseModel):
    enterprise_id: UUID | None = None


class MergeResponse(BaseModel):
    enterprise_id: UUID
    product_id: str | None
    merges: list[tuple[str, str]]


class ReclusterResponse(BaseModel):
    enterprise_id: UUID
    product_id: str | None
    clusters_created: int


class LabelingResponse(BaseModel):
    message: str | None = None
    labeled_clusters: int


def create_app() -> FastAPI:
    app = FastAPI(title="Arali API", version="0.1.0", docs_url="/docs", lifespan=lifespan)

    @app.get("/health", tags=["Health"])
    async def read_health() -> dict[str, str]:
        return {"status": "ok"}

    def get_pipeline(connection=Depends(get_connection)) -> FeatureReceptionPipeline:
        return FeatureReceptionPipeline(connection)

    @app.post("/feature-reception/online", response_model=OnlineTaskResponse)
    async def trigger_online(
        payload: OnlineTaskRequest,
        pipeline: FeatureReceptionPipeline = Depends(get_pipeline),
    ) -> OnlineTaskResponse:
        logger.info(f"Received online trigger for insight {payload.insight_id}")
        cluster_id = await pipeline.run_online(payload.insight_id)
        return OnlineTaskResponse(cluster_id=cluster_id)

    @app.post("/feature-reception/daily", response_model=DailyTaskResponse)
    async def trigger_daily(
        payload: DailyTaskRequest,
        background_tasks: BackgroundTasks,
    ) -> DailyTaskResponse:
        async def _run_daily_task(batch_size: int) -> None:
            if pool is None:
                logger.error("DATABASE_URL not configured; cannot run daily task.")
                return
            async with pool.acquire() as connection:
                pipeline = FeatureReceptionPipeline(connection)
                try:
                    logger.info(f"Starting daily task with batch_size={batch_size}")
                    result = await pipeline.run_daily(batch_size)
                    logger.info(f"Daily task completed successfully: {result}")
                except Exception as exc:
                    logger.exception("Daily task failed", exc_info=exc)
                    raise

        background_tasks.add_task(_run_daily_task, payload.batch_size)
        return DailyTaskResponse(
            message="Daily task scheduled. Check logs for progress.",
            embedded=0,
            assigned=0,
            clusters_updated=0,
        )

    @app.post("/feature-reception/weekly/recluster", response_model=list[ReclusterResponse])
    async def trigger_recluster(
        payload: WeeklyTaskRequest | None = None,
        pipeline: FeatureReceptionPipeline = Depends(get_pipeline),
    ) -> list[ReclusterResponse]:
        enterprise_id = payload.enterprise_id if payload else None
        results = await pipeline.run_recluster(enterprise_id)
        return [ReclusterResponse(**item) for item in results]

    @app.post("/feature-reception/weekly/merge", response_model=list[MergeResponse])
    async def trigger_merge(
        payload: WeeklyTaskRequest | None = None,
        pipeline: FeatureReceptionPipeline = Depends(get_pipeline),
    ) -> list[MergeResponse]:
        enterprise_id = payload.enterprise_id if payload else None
        results = await pipeline.run_merge(enterprise_id)
        return [MergeResponse(**item) for item in results]

    @app.post("/feature-reception/weekly/label", response_model=LabelingResponse)
    async def trigger_labeling(
        background_tasks: BackgroundTasks,
        payload: WeeklyTaskRequest | None = None,
    ) -> LabelingResponse:
        enterprise_id = payload.enterprise_id if payload else None

        async def _run_labeling_task(target_enterprise_id: UUID | None) -> None:
            if pool is None:
                logger.error("DATABASE_URL not configured; cannot run labeling task.")
                return
            async with pool.acquire() as connection:
                pipeline = FeatureReceptionPipeline(connection)
                try:
                    logger.info(f"Starting weekly labeling task for enterprise_id={target_enterprise_id}")
                    labeled = await pipeline.run_labeling(target_enterprise_id)
                    logger.info(f"Weekly labeling complete for enterprise filter {target_enterprise_id}: {labeled} clusters labeled")
                except Exception as exc:
                    logger.exception("Weekly labeling task failed", exc_info=exc)
                    raise

        background_tasks.add_task(_run_labeling_task, enterprise_id)
        return LabelingResponse(
            message="Weekly labeling scheduled. Check logs for completion.",
            labeled_clusters=0,
        )

    return app


app = create_app()
