"""
Unit tests for animal_id.identification — embedder, clusterer, and metrics.
"""

import numpy as np
import pytest

from animal_id.identification import (
    cluster,
    cluster_quality,
    embed_gallery,
    normalize_embeddings,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(dim: int, angle_radians: float) -> np.ndarray:
    """Return a 2-D unit vector at *angle_radians* padded with zeros to *dim*."""
    v = np.zeros(dim, dtype=np.float32)
    v[0] = np.cos(angle_radians)
    v[1] = np.sin(angle_radians)
    return v


# ---------------------------------------------------------------------------
# clusterer tests
# ---------------------------------------------------------------------------


class TestNormalizeEmbeddings:
    def test_unit_vectors_unchanged(self):
        """Already-normalized rows stay unchanged."""
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((8, 16)).astype(np.float32)
        normed = normalize_embeddings(raw)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, np.ones(8), atol=1e-6)

    def test_zero_row_safe(self):
        """A zero-norm row must not produce NaN."""
        emb = np.zeros((3, 4), dtype=np.float32)
        emb[1] = [1.0, 0.0, 0.0, 0.0]
        result = normalize_embeddings(emb)
        assert not np.any(np.isnan(result))
        # Non-zero row is still normalized.
        np.testing.assert_allclose(np.linalg.norm(result[1]), 1.0, atol=1e-6)


class TestCluster:
    """cluster() must recover two well-separated groups as distinct clusters."""

    def _make_blobs(self, n_per_group: int = 10, dim: int = 64, noise: float = 0.02):
        """Two tight groups around near-orthogonal directions in R^dim."""
        rng = np.random.default_rng(42)
        # Group A: centred near e_0 (angle ≈ 0)
        center_a = _unit_vec(dim, 0.0)
        # Group B: centred near e_1 (angle = π/2)
        center_b = _unit_vec(dim, np.pi / 2)

        group_a = center_a + rng.normal(0, noise, (n_per_group, dim)).astype(np.float32)
        group_b = center_b + rng.normal(0, noise, (n_per_group, dim)).astype(np.float32)

        embeddings = np.vstack([group_a, group_b])
        true_groups = [0] * n_per_group + [1] * n_per_group
        return embeddings, true_groups

    def test_two_clusters_found(self):
        embeddings, _ = self._make_blobs()
        labels = cluster(embeddings, eps=0.1, min_samples=2)
        unique_non_noise = set(labels.tolist()) - {-1}
        assert len(unique_non_noise) == 2

    def test_intra_group_same_label(self):
        """All points in the same true group share a predicted cluster label."""
        n = 10
        embeddings, _ = self._make_blobs(n_per_group=n)
        labels = cluster(embeddings, eps=0.1, min_samples=2)
        group_a_labels = set(labels[:n].tolist()) - {-1}
        group_b_labels = set(labels[n:].tolist()) - {-1}
        # Each group's non-noise points should map to a single cluster.
        assert len(group_a_labels) == 1
        assert len(group_b_labels) == 1
        # The two groups map to different clusters.
        assert group_a_labels.isdisjoint(group_b_labels)

    def test_returns_ndarray(self):
        embeddings, _ = self._make_blobs()
        labels = cluster(embeddings)
        assert isinstance(labels, np.ndarray)


# ---------------------------------------------------------------------------
# metrics tests
# ---------------------------------------------------------------------------


class TestClusterQuality:
    def test_perfect_prediction_scores(self):
        """Perfect clustering: pred == true → top scores and matching counts."""
        true = [0, 0, 1, 1, 2, 2]
        pred = np.array([0, 0, 1, 1, 2, 2])

        result = cluster_quality(true, pred)

        assert result["v_measure"] == pytest.approx(1.0, abs=1e-9)
        assert result["adjusted_rand_index"] == pytest.approx(1.0, abs=1e-9)
        assert result["purity"] == pytest.approx(1.0, abs=1e-9)
        assert result["num_clusters"] == 3
        assert result["num_true_identities"] == 3
        assert result["noise_rate"] == pytest.approx(0.0)

    def test_all_noise_returns_zero_purity(self):
        """When every point is noise, purity should be 0."""
        true = [0, 1, 2]
        pred = np.array([-1, -1, -1])
        result = cluster_quality(true, pred)
        assert result["purity"] == pytest.approx(0.0)
        assert result["noise_rate"] == pytest.approx(1.0)
        assert result["num_clusters"] == 0

    def test_mixed_prediction_keys_present(self):
        """All expected keys are present in the returned dict."""
        true = [0, 0, 1, 1, 0, 1]
        pred = np.array([0, 0, 0, 1, 1, -1])
        result = cluster_quality(true, pred)
        expected_keys = {
            "homogeneity",
            "completeness",
            "v_measure",
            "adjusted_rand_index",
            "num_clusters",
            "num_true_identities",
            "noise_rate",
            "purity",
        }
        assert set(result.keys()) == expected_keys

    def test_noise_rate_calculation(self):
        """noise_rate equals fraction of -1 labels."""
        true = [0, 0, 1, 1]
        pred = np.array([0, -1, 1, -1])
        result = cluster_quality(true, pred)
        assert result["noise_rate"] == pytest.approx(0.5)

    def test_purity_computed_over_non_noise(self):
        """Purity excludes noise points from the denominator."""
        # 4 non-noise points: cluster 0 has [A, A, B] → majority 2
        #                     cluster 1 has [B]        → majority 1
        # purity = (2 + 1) / 4 = 0.75
        true = ["A", "A", "B", "B", "C"]
        pred = np.array([0, 0, 0, 1, -1])
        result = cluster_quality(true, pred)
        assert result["purity"] == pytest.approx(0.75, abs=1e-9)


# ---------------------------------------------------------------------------
# embedder tests
# ---------------------------------------------------------------------------


class TestEmbedGallery:
    """embed_gallery must skip None embeddings and keep arrays aligned."""

    def _fake_pipeline(self, return_map: dict):
        """Return a duck-typed pipeline stub.

        Args:
            return_map: ``{image_path: embedding_or_None}``.
        """

        class _FakePipeline:
            def generate_embedding(self, image_path: str):
                return return_map.get(image_path)

        return _FakePipeline()

    def test_alignment_and_none_dropping(self):
        """Nones are dropped; embeddings, labels, paths stay aligned."""
        vec_a = np.array([1.0, 0.0], dtype=np.float32)
        vec_c = np.array([0.0, 1.0], dtype=np.float32)

        items = [
            {"image_path": "a.jpg", "identity_label": "dog_1"},
            {"image_path": "b.jpg", "identity_label": "dog_2"},  # None
            {"image_path": "c.jpg", "identity_label": "dog_3"},
        ]
        pipeline = self._fake_pipeline({"a.jpg": vec_a, "b.jpg": None, "c.jpg": vec_c})

        embeddings, labels, paths = embed_gallery(pipeline, items, show_progress=False)

        assert embeddings.shape == (2, 2)
        assert labels == ["dog_1", "dog_3"]
        assert paths == ["a.jpg", "c.jpg"]
        np.testing.assert_array_equal(embeddings[0], vec_a)
        np.testing.assert_array_equal(embeddings[1], vec_c)

    def test_all_none_returns_empty(self):
        """All Nones → empty embeddings array and empty lists."""
        items = [
            {"image_path": "x.jpg", "identity_label": 0},
            {"image_path": "y.jpg", "identity_label": 1},
        ]
        pipeline = self._fake_pipeline({"x.jpg": None, "y.jpg": None})

        embeddings, labels, paths = embed_gallery(pipeline, items, show_progress=False)

        assert len(labels) == 0
        assert len(paths) == 0
        assert embeddings.shape[0] == 0

    def test_output_dtype_float32(self):
        """Returned embeddings are float32 regardless of pipeline output dtype."""
        items = [{"image_path": "z.jpg", "identity_label": 42}]
        pipeline = self._fake_pipeline(
            {"z.jpg": np.array([1.0, 2.0], dtype=np.float64)}
        )
        embeddings, _, _ = embed_gallery(pipeline, items, show_progress=False)
        assert embeddings.dtype == np.float32

    def test_labels_aligned_with_integer_identity(self):
        """Integer identity labels are preserved without coercion."""
        items = [
            {"image_path": "p.jpg", "identity_label": 7},
            {"image_path": "q.jpg", "identity_label": 8},
        ]
        pipeline = self._fake_pipeline(
            {
                "p.jpg": np.ones(4, dtype=np.float32),
                "q.jpg": np.zeros(4, dtype=np.float32),
            }
        )
        _, labels, _ = embed_gallery(pipeline, items, show_progress=False)
        assert labels == [7, 8]
