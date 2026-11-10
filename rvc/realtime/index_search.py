"""Exact nearest-neighbour retrieval for the realtime index blend, on the model's device.

The FAISS path searches an IVF index on the CPU: the features leave the GPU, only the
`nprobe` nearest cells are looked at (so a closer vector in a neighbouring cell is
missed), and the result goes back to the GPU. An index here is small - 36 minutes of
training audio is about 110,000 vectors - so comparing every query against every vector
is one matrix product on the GPU: nothing is missed, and nothing leaves the device.

The result is what a FAISS search that visits every cell (`nprobe == nlist`) returns,
blended with the same weights as the FAISS path: `(1 / squared distance) ** 2`,
normalised over the `top_k` neighbours.
"""

import torch

# Upper bound on queries x vectors compared at once (fp32, so 4 bytes each): 64M is
# 256 MB of scores, and a realtime window is a few hundred queries at most.
MAX_SCORES_PER_CHUNK = 64_000_000

# A query that sits exactly on an index vector has distance 0. FAISS's weights turn
# into NaN there; here that neighbour simply takes (almost) all of the weight.
MIN_SQUARED_DISTANCE = 1e-12


class ExactIndex:
    def __init__(self, vectors, device, top_k: int = 8):
        """vectors: the index contents, (N, D) - what `index.reconstruct_n` returns."""
        self.vectors = torch.as_tensor(vectors, dtype=torch.float32).to(device)
        if self.vectors.dim() != 2 or self.vectors.shape[0] == 0:
            raise ValueError(
                f"The index holds no usable vectors (shape {tuple(self.vectors.shape)})."
            )
        self.squared_norms = (self.vectors * self.vectors).sum(dim=1)
        self.top_k = max(1, min(int(top_k), self.vectors.shape[0]))
        self.chunk = max(1, MAX_SCORES_PER_CHUNK // self.vectors.shape[0])

    @property
    def dim(self):
        return self.vectors.shape[1]

    def search(self, queries):
        """Squared L2 distances and indices of the top_k nearest vectors, nearest first."""
        distances, indices = [], []
        for start in range(0, queries.shape[0], self.chunk):
            q = queries[start : start + self.chunk].to(self.vectors.dtype)
            scores = (
                (q * q).sum(dim=1, keepdim=True)
                - 2.0 * (q @ self.vectors.T)
                + self.squared_norms
            )
            d, ix = torch.topk(scores, self.top_k, dim=1, largest=False)
            distances.append(d)
            indices.append(ix)
        return torch.cat(distances), torch.cat(indices)

    def retrieve(self, queries):
        """For each query row, the weighted mean of its top_k nearest index vectors."""
        distances, indices = self.search(queries)
        weight = torch.square(1.0 / distances.clamp_min(MIN_SQUARED_DISTANCE))
        weight = weight / weight.sum(dim=1, keepdim=True)
        neighbours = self.vectors[indices]  # (T, k, D)
        return (neighbours * weight.unsqueeze(-1)).sum(dim=1).to(queries.dtype)
