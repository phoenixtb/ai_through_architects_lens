# SISO Production Implementation Guide

**SISO** = Semantic Index for Serving Optimization

Improves on basic semantic caching with:
- Centroid-based clustering for O(k + m) lookup vs O(n)
- Locality-aware eviction (per-cluster LRU)
- Dynamic thresholds per cluster

## Architecture

```
Query → Embed → Find Cluster → Search Within → Hit/Miss
           │         │              │
           ▼         ▼              ▼
   ┌──────────────┐ ┌────────┐ ┌─────────────────┐
   │ Embeddings   │ │Centroid│ │ Cluster Entries │
   │ (local model)│ │ Index  │ │   (per-cluster) │
   └──────────────┘ └────────┘ └─────────────────┘
```

## Component Mapping

| Component | Dev/Notebook | Production |
|-----------|--------------|------------|
| Embeddings | sentence-transformers | Same, or OpenAI/Cohere |
| Centroid Index | Python dict | Redis Sorted Set / Qdrant |
| Cluster Storage | Python dict | Redis Hash / PostgreSQL |
| Vector Search | NumPy cosine | FAISS / Qdrant / pgvector |
| Eviction | In-memory LRU | Redis TTL + custom |

---

## Option 1: Qdrant (Recommended)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(url="http://localhost:6333")  # or Qdrant Cloud

# Create collection
client.create_collection(
    collection_name="siso_cache",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Cache set with cluster_id for locality-aware eviction
def cache_set(query: str, response: str, cluster_id: int):
    embedding = encoder.encode(query)
    client.upsert(
        collection_name="siso_cache",
        points=[PointStruct(
            id=hash(query),
            vector=embedding.tolist(),
            payload={
                "query": query,
                "response": response,
                "cluster_id": cluster_id,
                "accessed_at": time.time()
            }
        )]
    )

# Search with cluster filtering
def cache_get(query: str, cluster_id: int = None):
    embedding = encoder.encode(query)
    filter_condition = {"cluster_id": cluster_id} if cluster_id else None
    
    results = client.search(
        collection_name="siso_cache",
        query_vector=embedding.tolist(),
        query_filter=filter_condition,
        limit=1,
        score_threshold=0.80
    )
    return results[0].payload["response"] if results else None
```

---

## Option 2: Redis + FAISS

```python
import redis
import faiss
import numpy as np

redis_client = redis.Redis()
dimension = 384
centroid_index = faiss.IndexFlatIP(dimension)

def find_cluster(query_embedding):
    """O(k) where k = number of clusters"""
    if centroid_index.ntotal == 0:
        return None
    D, I = centroid_index.search(query_embedding.reshape(1, -1), 1)
    if D[0][0] > 0.7:
        return I[0][0]
    return None

def cache_get(query: str):
    embedding = encoder.encode(query)
    cluster_id = find_cluster(embedding)
    
    if cluster_id is None:
        return None
    
    # Search within cluster (stored in Redis)
    cluster_key = f"siso:cluster:{cluster_id}"
    entries = redis_client.hgetall(cluster_key)
    
    best_score, best_response = 0, None
    for entry_id, data in entries.items():
        entry = json.loads(data)
        score = cosine_similarity(embedding, np.array(entry["embedding"]))
        if score > best_score and score > entry.get("threshold", 0.80):
            best_score = score
            best_response = entry["response"]
    
    return best_response
```

---

## Option 3: PostgreSQL + pgvector

```sql
-- Schema with cluster support
CREATE TABLE siso_cache (
    id SERIAL PRIMARY KEY,
    query TEXT,
    response TEXT,
    embedding vector(384),
    cluster_id INTEGER,
    accessed_at TIMESTAMP DEFAULT NOW(),
    access_count INTEGER DEFAULT 0
);

CREATE INDEX ON siso_cache USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON siso_cache (cluster_id);

-- Cluster centroids
CREATE TABLE siso_clusters (
    id SERIAL PRIMARY KEY,
    centroid vector(384),
    threshold FLOAT DEFAULT 0.80,
    entry_count INTEGER DEFAULT 0
);
```

```python
def cache_get(query: str):
    embedding = encoder.encode(query)
    
    # Find cluster
    cursor.execute("""
        SELECT id FROM siso_clusters 
        ORDER BY centroid <=> %s::vector 
        LIMIT 1
    """, (embedding.tolist(),))
    cluster = cursor.fetchone()
    
    if not cluster:
        return None
    
    # Search within cluster
    cursor.execute("""
        SELECT response, 1 - (embedding <=> %s::vector) as similarity
        FROM siso_cache 
        WHERE cluster_id = %s AND 1 - (embedding <=> %s::vector) > 0.80
        ORDER BY similarity DESC
        LIMIT 1
    """, (embedding.tolist(), cluster[0], embedding.tolist()))
    
    result = cursor.fetchone()
    return result[0] if result else None
```

---

## Clustering Strategy

```python
from sklearn.cluster import MiniBatchKMeans

class SISOClusterManager:
    def __init__(self, n_clusters=20):
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        self.fitted = False
    
    def update_clusters(self, embeddings: np.ndarray):
        """Periodically retrain clusters (cron job)"""
        self.kmeans.partial_fit(embeddings)
        self.fitted = True
        return self.kmeans.cluster_centers_
    
    def assign_cluster(self, embedding: np.ndarray) -> int:
        if not self.fitted:
            return 0
        return self.kmeans.predict(embedding.reshape(1, -1))[0]
```

---

## Summary

| Approach | Complexity | Best For |
|----------|------------|----------|
| Qdrant | Low | Startups, fast setup |
| Redis + FAISS | Medium | High throughput, existing Redis |
| PostgreSQL + pgvector | Medium | Existing Postgres, ACID needs |
| Custom (notebook demo) | Educational | Understanding concepts |

**Recommendation**: Qdrant with `:memory:` for dev, Qdrant Cloud/server for prod.

