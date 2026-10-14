"""Embedding diagnostics API.

Serves lightweight 2D projections of generated ML embedding artifacts for the
React dashboard.  The endpoint is read-only and intentionally recomputes the
projection from the stored CSV artifact so it can be used after every embedding
rebuild without adding a database table.
"""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


router = APIRouter(prefix="/embeddings", tags=["embeddings"])

# umap-learn uses numba under the hood.  The default numba workqueue threading
# layer is not safe when several FastAPI worker threads call UMAP concurrently;
# it can terminate the whole API process with "Concurrent access has been
# detected".  Serialize only the projection fit; cached responses still return
# immediately.
_UMAP_LOCK = threading.Lock()

DEFAULT_ARTIFACT_DIR = Path("/app/betting_app/models/ml/champion_role_embeddings/exp-056")
LOCAL_ARTIFACT_DIR = Path("betting_app/models/ml/champion_role_embeddings/exp-056")

ProjectionPreset = Literal["local", "balanced", "global"]

PROJECTION_PRESETS: dict[str, dict[str, object]] = {
    "local": {
        "label": "Local",
        "description": "Emphasize nearest-neighbour micro-clusters.",
        "umap_n_neighbors": 10,
        "umap_min_dist": 0.02,
        "umap_metric": "cosine",
        "tsne_perplexity": 10,
    },
    "balanced": {
        "label": "Balanced",
        "description": "Default diagnostic view balancing local and global structure.",
        "umap_n_neighbors": 30,
        "umap_min_dist": 0.08,
        "umap_metric": "cosine",
        "tsne_perplexity": 30,
    },
    "global": {
        "label": "Global",
        "description": "Prefer broader role/champion archetype layout over tiny clusters.",
        "umap_n_neighbors": 80,
        "umap_min_dist": 0.35,
        "umap_metric": "cosine",
        "tsne_perplexity": 50,
    },
}


def _preset_config(preset: str) -> dict[str, object]:
    return PROJECTION_PRESETS.get(preset, PROJECTION_PRESETS["balanced"])


def _bounded_neighbor_count(value: object, n_points: int) -> int:
    return max(2, min(int(value), max(2, n_points - 1)))


def _bounded_perplexity(value: object, n_points: int) -> int:
    # sklearn requires perplexity < n_samples. Keep it conservative for small filters.
    return max(2, min(int(value), max(2, n_points - 1)))


def _auto_cluster_count(n_points: int) -> int:
    if n_points < 8:
        return 0
    return max(2, min(8, int(round(np.sqrt(n_points / 2.0)))))


def _artifact_dir() -> Path:
    if DEFAULT_ARTIFACT_DIR.exists():
        return DEFAULT_ARTIFACT_DIR
    return LOCAL_ARTIFACT_DIR


def _load_manifest(artifact_dir: Path) -> dict:
    manifest_path = artifact_dir / "walk_forward_manifest.json"
    if not manifest_path.exists():
        return {"snapshots": []}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"snapshots": []}


def _available_snapshot_ids(artifact_dir: Path) -> list[str]:
    manifest = _load_manifest(artifact_dir)
    snapshots = [str(item.get("snapshot")) for item in manifest.get("snapshots", []) if item.get("snapshot")]
    return sorted(set(snapshots))


def _resolve_snapshot_dir(artifact_dir: Path, snapshot: str) -> tuple[Path, str]:
    snapshot_norm = snapshot.strip() or "latest"
    if snapshot_norm in {"latest", "current"}:
        snapshots = _available_snapshot_ids(artifact_dir)
        if snapshots:
            latest = snapshots[-1]
            return artifact_dir / "snapshots" / latest, latest
        return artifact_dir, "current"
    if not snapshot_norm.replace("-", "").isdigit():
        raise HTTPException(status_code=400, detail="Snapshot must be 'latest' or YYYY-MM-DD.")
    snapshot_dir = artifact_dir / "snapshots" / snapshot_norm
    if not snapshot_dir.exists():
        raise HTTPException(status_code=404, detail=f"Champion embedding snapshot not found: {snapshot_norm}")
    return snapshot_dir, snapshot_norm


def _artifact_paths(snapshot: str = "latest") -> tuple[Path, Path, str, list[str]]:
    artifact_dir = _artifact_dir()
    resolved_dir, resolved_snapshot = _resolve_snapshot_dir(artifact_dir, snapshot)
    return (
        resolved_dir / "champion_role_embeddings.csv",
        resolved_dir / "metadata.json",
        resolved_snapshot,
        _available_snapshot_ids(artifact_dir),
    )


def _json_safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _json_safe_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return str(value)


def _mtime_key(snapshot: str) -> tuple[str, float, str, float, str, tuple[str, ...]]:
    csv_path, metadata_path, resolved_snapshot, available_snapshots = _artifact_paths(snapshot)
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Champion embedding artifact not found. Run "
                "betting_app.scripts.build_champion_role_embeddings first."
            ),
        )
    metadata_mtime = metadata_path.stat().st_mtime if metadata_path.exists() else 0.0
    return str(csv_path), csv_path.stat().st_mtime, str(metadata_path), metadata_mtime, resolved_snapshot, tuple(available_snapshots)


@lru_cache(maxsize=32)
def _project_champion_embeddings(
    csv_path_str: str,
    csv_mtime: float,
    metadata_path_str: str,
    metadata_mtime: float,
    resolved_snapshot: str,
    available_snapshots: tuple[str, ...],
    method: str,
    preset: str,
    role: str,
    min_games: int,
    max_points: int,
) -> dict:
    del csv_mtime, metadata_mtime  # cache-busting keys; values are not used directly

    csv_path = Path(csv_path_str)
    metadata_path = Path(metadata_path_str)
    df = pd.read_csv(csv_path)
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}

    vector_cols = [c for c in df.columns if c.startswith("emb_")]
    if len(vector_cols) < 2:
        raise HTTPException(status_code=422, detail="Embedding artifact does not contain emb_* vector columns.")

    available_roles = sorted(str(r) for r in df["role"].dropna().unique()) if "role" in df.columns else []
    filtered = df.copy()
    if role != "ALL":
        filtered = filtered[filtered["role"].astype(str) == role]
    min_games_column = "recent_games" if "recent_games" in filtered.columns else "n_games"
    if min_games_column in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered[min_games_column], errors="coerce").fillna(0) >= min_games]

    if filtered.empty:
        raise HTTPException(status_code=404, detail="No champion-role embeddings match the selected filters.")

    filtered = filtered.sort_values(["role", "champion_name", "champion_id"], na_position="last").head(max_points)
    matrix = filtered[vector_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)

    cluster_count = _auto_cluster_count(len(filtered)) if role != "ALL" else 0
    cluster_labels: list[int | None]
    cluster_counts: dict[str, int]
    if cluster_count > 0:
        labels = KMeans(n_clusters=cluster_count, n_init=20, random_state=42).fit_predict(matrix)
        cluster_labels = [int(v) for v in labels]
        cluster_counts = {str(i): int(np.sum(labels == i)) for i in range(cluster_count)}
    else:
        cluster_labels = [None] * len(filtered)
        cluster_counts = {}

    actual_method = method
    preset_cfg = _preset_config(preset)
    if method in {"umap", "tsne"} and len(filtered) < 4:
        actual_method = "pca"

    projection_warning = None

    if actual_method == "umap":
        try:
            import umap  # type: ignore[import-untyped]

            with _UMAP_LOCK:
                projection = umap.UMAP(
                    n_components=2,
                    n_neighbors=_bounded_neighbor_count(preset_cfg["umap_n_neighbors"], len(filtered)),
                    min_dist=float(preset_cfg["umap_min_dist"]),
                    metric=str(preset_cfg["umap_metric"]),
                    random_state=42,
                ).fit_transform(matrix)
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="UMAP is not installed in the API container. Rebuild after installing umap-learn.",
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive API fallback
            projection_warning = f"UMAP failed ({type(exc).__name__}); returned PCA fallback."
            actual_method = "pca"
            projection = PCA(n_components=2, random_state=42).fit_transform(matrix)
    elif actual_method == "pca":
        projection = PCA(n_components=2, random_state=42).fit_transform(matrix)
    elif actual_method == "tsne":
        perplexity = _bounded_perplexity(preset_cfg["tsne_perplexity"], len(filtered))
        try:
            projection = TSNE(
                n_components=2,
                init="pca",
                learning_rate="auto",
                perplexity=perplexity,
                random_state=42,
                max_iter=1200,
            ).fit_transform(matrix)
        except Exception as exc:  # pragma: no cover - defensive API fallback
            projection_warning = f"t-SNE failed ({type(exc).__name__}); returned PCA fallback."
            actual_method = "pca"
            projection = PCA(n_components=2, random_state=42).fit_transform(matrix)
    else:
        raise HTTPException(status_code=400, detail="Unsupported projection method.")

    x = projection[:, 0]
    y = projection[:, 1]
    span_x = float(np.ptp(x)) or 1.0
    span_y = float(np.ptp(y)) or 1.0
    points = []
    for idx, row in enumerate(filtered.to_dict(orient="records")):
        points.append(
            {
                "champion_id": str(row.get("champion_id", "")),
                "champion_name": row.get("champion_name") or str(row.get("champion_id", "UNKNOWN")),
                "role": row.get("role"),
                "x": float(x[idx]),
                "y": float(y[idx]),
                "x_norm": float((x[idx] - x.min()) / span_x),
                "y_norm": float((y[idx] - y.min()) / span_y),
                "n_games": _json_safe_float(row.get("n_games")),
                "recent_games": _json_safe_float(row.get("recent_games")),
                "recent_window_days": _json_safe_float(row.get("recent_window_days")),
                "recent_date_max": _json_safe_str(row.get("recent_date_max")),
                "win_rate": _json_safe_float(row.get("win_rate")),
                "fallback_level": row.get("fallback_level"),
                "window_days": _json_safe_float(row.get("window_days")),
                "shrinkage_weight_observed": _json_safe_float(row.get("shrinkage_weight_observed")),
                "age_days_mean": _json_safe_float(row.get("age_days_mean")),
                "kda": _json_safe_float(row.get("mean_kda")),
                "damage_share": _json_safe_float(row.get("mean_damage_share")),
                "gold_share": _json_safe_float(row.get("mean_gold_share")),
                "kill_participation": _json_safe_float(row.get("mean_kill_participation")),
                "cluster_id": cluster_labels[idx],
                "cluster_label": f"Cluster {cluster_labels[idx] + 1}" if cluster_labels[idx] is not None else None,
            }
        )

    return {
        "metadata": {
            "artifact_path": str(csv_path),
            "method": actual_method,
            "requested_method": method,
            "preset": preset,
            "preset_config": {
                "label": preset_cfg["label"],
                "description": preset_cfg["description"],
                "umap_n_neighbors": _bounded_neighbor_count(preset_cfg["umap_n_neighbors"], len(filtered)),
                "umap_min_dist": preset_cfg["umap_min_dist"],
                "umap_metric": preset_cfg["umap_metric"],
                "tsne_perplexity": _bounded_perplexity(preset_cfg["tsne_perplexity"], len(filtered)),
            },
            "projection_warning": projection_warning,
            "snapshot": resolved_snapshot,
            "available_snapshots": list(available_snapshots),
            "role": role,
            "min_games": min_games,
            "min_games_column": min_games_column,
            "recent_window_days": metadata.get("recent_window_days"),
            "total_points": int(len(points)),
            "cluster_count": cluster_count,
            "cluster_counts": cluster_counts,
            "available_roles": available_roles,
            "source_rows": metadata.get("source_rows"),
            "reference_date": metadata.get("reference_date"),
            "embedding_dim": metadata.get("embedding_dim", len(vector_cols)),
            "model_name": metadata.get("model_name", "ChampionRoleEmbeddings"),
            "model_version": metadata.get("model_version", "exp-056"),
            "fallback_counts": metadata.get("fallback_counts", {}),
        },
        "points": points,
    }


@router.get("/champions")
def champion_embedding_projection(
    method: Literal["umap", "tsne", "pca"] = Query("umap", description="2D projection method."),
    preset: ProjectionPreset = Query("balanced", description="Projection preset: local, balanced, or global."),
    snapshot: str = Query("latest", description="Walk-forward snapshot: latest/current or YYYY-MM-DD."),
    role: str = Query("ALL", description="Role filter: ALL, TOP, JUNGLE, MID, ADC, SUPPORT."),
    min_games: int = Query(0, ge=0, le=1000),
    max_points: int = Query(800, ge=10, le=2000),
):
    """Return a 2D projection of champion-role embeddings.

    UMAP is the default nonlinear view; t-SNE and PCA are available for
    comparison/debugging.
    """
    key = _mtime_key(snapshot)
    role_norm = role.upper()
    return _project_champion_embeddings(*key, method, preset, role_norm, int(min_games), int(max_points))
