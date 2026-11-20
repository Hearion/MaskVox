from __future__ import annotations

from pathlib import Path

import numpy as np

_BYTE_LIMIT = 2 * 1024**3


class VoiceIndex:
    def __init__(self, path: Path, dimension: int) -> None:
        import faiss

        # These process-wide limits are set here on the sole inference worker
        # before loading an index. The offline CLI has its own process.
        faiss.omp_set_num_threads(1)
        faiss.set_deserialization_loop_limit(1_000_000)
        faiss.set_deserialization_vector_byte_limit(_BYTE_LIMIT)
        faiss.set_deserialization_lattice_r2_limit(4096)
        if path.stat().st_size > _BYTE_LIMIT:
            raise ValueError("音色索引文件超过 2 GB，已关闭音色贴合")
        with path.open("rb") as source:
            # A Python reader handles Unicode paths. With ONDISK_SAME_DIR,
            # Faiss rejects external inverted lists without a native file path.
            reader = faiss.PyCallbackIOReader(source.read, 1024 * 1024)
            flags = faiss.IO_FLAG_READ_ONLY | faiss.IO_FLAG_ONDISK_SAME_DIR
            self.database = faiss.read_index(reader, flags)
        if self.database.d != dimension:
            raise ValueError("索引维度与音色不匹配，已关闭音色贴合")
        if self.database.metric_type != faiss.METRIC_L2:
            raise ValueError("音色索引需要使用 L2 距离，已关闭音色贴合")
        total = self.database.ntotal
        if total < 1 or not self.database.is_trained:
            raise ValueError("音色索引为空，请选用 added_ 开头的索引")
        if total * dimension * 4 > _BYTE_LIMIT:
            raise ValueError("音色索引超过 2 GB 的向量上限，已关闭音色贴合")
        self.vectors = self.database.reconstruct_n(0, total)
        for offset in range(0, total, 4096):
            if not np.isfinite(self.vectors[offset : offset + 4096]).all():
                raise ValueError("音色索引含有无效向量，已关闭音色贴合")
        self.neighbors = min(total, 8)

    def blend(self, features: np.ndarray, amount: float) -> np.ndarray:
        scores, ids = self.database.search(np.ascontiguousarray(features), self.neighbors)
        valid = (ids >= 0) & (ids < len(self.vectors)) & np.isfinite(scores) & (scores >= 0)
        hits = valid & (scores == 0)
        inverse_square = np.square(1 / np.maximum(scores, 1e-8))
        weights = np.where(hits.any(axis=1, keepdims=True), hits, np.where(valid, inverse_square, 0))
        weight_sum = weights.sum(axis=1, keepdims=True)
        weights /= np.where(weight_sum > 0, weight_sum, 1)
        retrieved = np.sum(self.vectors[ids.clip(0, len(self.vectors) - 1)] * weights[:, :, None], axis=1)
        mixture = amount * retrieved + (1 - amount) * features
        return np.where(weight_sum > 0, mixture, features).astype(np.float32, copy=False)
