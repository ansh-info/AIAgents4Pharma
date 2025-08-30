"""
Test cases for tools/utils/extractions/milvus_multimodal_pcst.py
"""

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pymilvus
import pytest

from ..utils.extractions.milvus_multimodal_pcst import (
    DynamicLibraryLoader,
    MultimodalPCSTPruning,
    SystemDetector,
)


class FakeMilvusCollection:
    """fake pymilvus.Collection with minimal methods for testing"""

    def __init__(self, name):
        """test_system_detector_init_and_methods"""
        self.name = name
        # Default sizes; tests can monkeypatch attributes
        self.num_entities = 6
        self._search_data = []  # set by tests
        self._query_batches = {}  # dict: (start,end)->list of dict rows

    def load(self):  # no-op
        """load collection"""
        return None

    def search(self, data, anns_field, param, limit, output_fields):
        """search method"""

        # Return a list [hits], where hits is an iterable of objects with .id and .score
        # We'll synthesize predictable hits: ids = range(limit) with descending scores
        class Hit:
            """hit object"""

            def __init__(self, i, s):
                """init hit"""
                self.id, self.score = i, s

        hits = [Hit(i, float(limit - i)) for i in range(limit)]
        return [hits]

    def query(self, expr, output_fields):
        """query method"""
        # Expect expr like: triplet_index >= a and triplet_index < b
        # We'll extract a,b and yield rows accordingly
        if "triplet_index" in expr:
            parts = expr.replace(" ", "").split("triplet_index>=")[1]
            start = int(parts.split("andtriplet_index<")[0])
            end = int(parts.split("andtriplet_index<")[1])
            rows = []
            for i in range(start, end):
                rows.append({"head_index": i, "tail_index": i + 1})
            return rows
        # raise AssertionError(f"Unexpected expr: {expr}")


class FakeAsyncConnMgr:
    """Minimal async connection manager for *_async methods."""

    def __init__(self, num_nodes=10, num_edges=8):
        """init"""
        self._num_nodes = num_nodes
        self._num_edges = num_edges

    async def async_get_collection_stats(self, collection_name):
        """async get_collection_stats"""
        if collection_name.endswith("_edges"):
            return {"num_entities": self._num_edges}
        return {"num_entities": self._num_nodes}

    async def async_search(
        self, collection_name, data, anns_field, param, limit, output_fields
    ):
        """async search"""
        # Return list of dicts with 'id' and 'distance'
        return [[{"id": i, "distance": float(limit - i)} for i in range(limit)]]


@pytest.fixture
def patch_milvus_collection(monkeypatch):
    """patch pymilvus.Collection with FakeMilvusCollection"""
    # Patch pymilvus.Collection inside the module under test

    mod = importlib.import_module(
        "..utils.extractions.milvus_multimodal_pcst", package=__package__
    )
    monkeypatch.setattr(mod, "Collection", FakeMilvusCollection, raising=True)
    return mod


@pytest.fixture
def fake_detector_cpu(monkeypatch):
    """force CPU-only environment (macOS + no NVIDIA)"""
    # Make sure detector reports CPU (no GPU)
    det = SystemDetector.__new__(SystemDetector)
    det.os_type = "darwin"
    det.architecture = "arm64"
    det.has_nvidia_gpu = False
    det.use_gpu = False
    return det


@pytest.fixture
def fake_detector_gpu(monkeypatch):
    """force GPU-capable environment (Linux + NVIDIA)"""
    # Force GPU-capable environment (Linux + NVIDIA)
    det = SystemDetector.__new__(SystemDetector)
    det.os_type = "linux"
    det.architecture = "x86_64"
    det.has_nvidia_gpu = True
    det.use_gpu = True
    return det


@pytest.fixture
def patch_cupy_cudf(monkeypatch):
    """Provide minimal cupy/cudf-like objects for GPU branch."""

    class FakeCP:
        """fake cupy with minimal methods"""

        float32 = np.float32

        @staticmethod
        def asarray(x):
            """static asarray method"""
            return np.asarray(x)

        class linalg:
            """linalg submodule"""

            @staticmethod
            def norm(x, axis=None, keepdims=False):
                """norm method"""
                return np.linalg.norm(x, axis=axis, keepdims=keepdims)

    class FakeCuDF:
        """fake cudf with minimal methods"""

        DataFrame = pd.DataFrame
        concat = staticmethod(pd.concat)

    mod = importlib.import_module(
        "..utils.extractions.milvus_multimodal_pcst", package=__package__
    )
    monkeypatch.setattr(mod, "cp", FakeCP, raising=True)
    monkeypatch.setattr(mod, "cudf", FakeCuDF, raising=True)
    monkeypatch.setattr(mod, "CUDF_AVAILABLE", True, raising=True)
    return SimpleNamespace(FakeCP=FakeCP, FakeCuDF=FakeCuDF)


def test_dynamic_library_loader_cpu_path(fake_detector_cpu):
    """test DynamicLibraryLoader in CPU mode"""
    loader = DynamicLibraryLoader(fake_detector_cpu)
    assert loader.use_gpu is False
    assert loader.metric_type == "COSINE"
    assert loader.normalize_vectors is False
    # normalize_matrix should be pass-through on CPU
    m = np.array([[3.0, 4.0]])
    out = loader.normalize_matrix(m, axis=1)
    assert np.allclose(out, m)
    # to_list works for numpy arrays
    assert loader.to_list(np.array([1, 2, 3])) == [1, 2, 3]


def test_dynamic_library_loader_gpu_path(fake_detector_gpu, patch_cupy_cudf):
    """dynamic loader in GPU mode"""
    loader = DynamicLibraryLoader(fake_detector_gpu)
    assert loader.use_gpu is True
    assert loader.metric_type == "IP"
    assert loader.normalize_vectors is True
    # normalization should change the norm to 1 along axis=1
    m = np.array([[3.0, 4.0]], dtype=np.float32)
    out = loader.normalize_matrix(m, axis=1)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_prepare_collections_creates_expected_collections(
    monkeypatch, patch_milvus_collection, fake_detector_cpu
):
    """prepare_collections creates expected collections based on modality"""
    loader = DynamicLibraryLoader(fake_detector_cpu)
    pcst = MultimodalPCSTPruning(loader=loader)

    cfg = SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg"))

    # modality != "prompt" => nodes, nodes_type, edges
    colls = pcst.prepare_collections(cfg, modality="gene/protein")
    assert set(colls.keys()) == {"nodes", "nodes_type", "edges"}
    assert "nodes_gene_protein" in colls["nodes_type"].name

    # modality == "prompt" => no nodes_type
    colls2 = pcst.prepare_collections(cfg, modality="prompt")
    assert set(colls2.keys()) == {"nodes", "edges"}


@pytest.mark.asyncio
async def test__load_edge_index_from_milvus_async_batches(
    monkeypatch, patch_milvus_collection, fake_detector_cpu
):
    """load_edge_index_from_milvus_async handles batching correctly"""
    loader = DynamicLibraryLoader(fake_detector_cpu)
    pcst = MultimodalPCSTPruning(loader=loader)
    cfg = SimpleNamespace(
        milvus_db=SimpleNamespace(database_name="primekg", query_batch_size=3)
    )

    class CountingCollection(FakeMilvusCollection):
        """collection that forces specific num_entities for batching"""

        def __init__(self, name):
            """init"""
            super().__init__(name)
            self.num_entities = 7  # forces batches: 0-3, 3-6, 6-7

    # Patch the symbol inside the module under test
    mod = importlib.import_module(
        "..utils.extractions.milvus_multimodal_pcst", package=__package__
    )
    monkeypatch.setattr(mod, "Collection", CountingCollection, raising=True)

    # ALSO patch the direct import used inside load_edges_sync(): "from pymilvus import Collection"

    monkeypatch.setattr(pymilvus, "Collection", CountingCollection, raising=True)

    edge_index = await pcst.load_edge_index_async(cfg, _connection_manager=None)

    assert edge_index.shape[0] == 2
    heads, tails = edge_index
    assert np.all(tails - heads == 1)
    assert heads[0] == 0 and heads[-1] == 6


def test__compute_node_prizes_search_branches(
    monkeypatch, patch_milvus_collection, fake_detector_cpu
):
    """compute_node_prizes uses correct collection based on use_description"""
    loader = DynamicLibraryLoader(fake_detector_cpu)
    pcst_desc = MultimodalPCSTPruning(loader=loader, use_description=True, topk=4)
    pcst_feat = MultimodalPCSTPruning(loader=loader, use_description=False, topk=3)

    cfg = SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg"))

    # Build collections using prepare_collections (will create nodes and nodes_type)
    colls = pcst_feat.prepare_collections(cfg, modality="gene/protein")

    # use_description=True should search colls["nodes"]
    prizes_desc = pcst_desc._compute_node_prizes([0.1, 0.2], colls)
    # top 4 get positive values from arange(4..1)
    assert np.count_nonzero(prizes_desc) == 4

    # use_description=False should search colls["nodes_type"]
    prizes_feat = pcst_feat._compute_node_prizes([0.1, 0.2], colls)
    assert np.count_nonzero(prizes_feat) == 3


@pytest.mark.asyncio
async def test__compute_node_prizes_async_uses_manager(fake_detector_cpu):
    """compute_node_prizes_async uses connection manager and topk correctly"""
    loader = DynamicLibraryLoader(fake_detector_cpu)
    pcst = MultimodalPCSTPruning(loader=loader, topk=3, metric_type="COSINE")

    manager = FakeAsyncConnMgr(num_nodes=5)
    prizes = await pcst._compute_node_prizes_async(
        query_emb=[0.1, 0.2],
        collection_name="primekg_nodes_gene_protein",
        connection_manager=manager,
        use_description=False,
    )
    assert np.count_nonzero(prizes) == 3


def test__compute_edge_prizes_and_scaling(
    monkeypatch, patch_milvus_collection, fake_detector_cpu
):
    """compute_edge_prizes uses correct collection and scaling"""
    loader = DynamicLibraryLoader(fake_detector_cpu)
    pcst = MultimodalPCSTPruning(loader=loader, topk_e=4, c_const=0.2)
    cfg = SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg"))
    colls = pcst.prepare_collections(cfg, modality="gene/protein")

    prizes = pcst._compute_edge_prizes([0.3, 0.1], colls)
    # Should have nonzero values, at least topk_e many unique-based-scaled entries
    assert np.count_nonzero(prizes) >= 1
    # ensure size matches num_entities of edges collection (Fake uses 6)
    assert prizes.shape[0] == colls["edges"].num_entities


@pytest.mark.asyncio
async def test__compute_edge_prizes_async_and_scaling(fake_detector_cpu):
    """compute_edge_prizes_async uses connection manager and scaling"""
    loader = DynamicLibraryLoader(fake_detector_cpu)
    pcst = MultimodalPCSTPruning(loader=loader, topk_e=3, c_const=0.1)

    manager = FakeAsyncConnMgr(num_edges=7)
    prizes = await pcst._compute_edge_prizes_async(
        text_emb=[0.2, 0.4],
        collection_name="primekg_edges",
        connection_manager=manager,
    )
    assert np.count_nonzero(prizes) >= 1
    assert prizes.shape[0] == 7


def test_compute_prizes_calls_node_and_edge_paths(
    monkeypatch, patch_milvus_collection, fake_detector_cpu
):
    """compute_prizes calls the node and edge prize methods and combines results"""
    loader = DynamicLibraryLoader(fake_detector_cpu)
    pcst = MultimodalPCSTPruning(loader=loader, topk=2, topk_e=2, use_description=False)
    cfg = SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg"))
    colls = pcst.prepare_collections(cfg, modality="gene/protein")

    out = pcst.compute_prizes(text_emb=[0.1, 0.2], query_emb=[0.1, 0.2], colls=colls)
    assert "nodes" in out and "edges" in out
    assert out["nodes"].shape[0] == colls["nodes"].num_entities
    assert out["edges"].shape[0] == colls["edges"].num_entities


@pytest.mark.asyncio
async def test_compute_prizes_async_uses_thread(
    fake_detector_cpu, patch_milvus_collection
):
    """compute_prizes_async uses connection manager and returns combined prizes"""
    loader = DynamicLibraryLoader(fake_detector_cpu)
    pcst = MultimodalPCSTPruning(loader=loader, topk=2, topk_e=2)
    cfg = SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg"))
    manager = FakeAsyncConnMgr()
    out = await pcst.compute_prizes_async(
        text_emb=[0.1, 0.2],
        query_emb=[0.1, 0.2],
        cfg=cfg,
        modality="gene/protein",
    )
    assert "nodes" in out and "edges" in out


def test_compute_subgraph_costs_and_mappings(fake_detector_cpu):
    """compute_subgraph_costs creates expected outputs and mappings"""
    loader = DynamicLibraryLoader(fake_detector_cpu)
    pcst = MultimodalPCSTPruning(
        loader=loader, topk=2, topk_e=2, c_const=0.1, cost_e=0.5
    )

    # prizes with some nonzero edge prizes to create real/virtual splits
    prizes = {
        "nodes": np.array([0, 0, 0, 0, 0], dtype=np.float32),
        "edges": np.array([0.1, 0.4, 0.9, 0.0], dtype=np.float32),  # mix of low/high
    }
    # simple edge_index: 2 x 4
    edge_index = np.array(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ],
        dtype=np.int64,
    )
    edges_dict, final_prizes, costs, mapping = pcst.compute_subgraph_costs(
        edge_index=edge_index, num_nodes=5, prizes=prizes
    )
    # Edges dict should expose combined edges and count of real edges
    assert "edges" in edges_dict and "num_prior_edges" in edges_dict
    assert final_prizes.shape[0] >= prizes["nodes"].shape[0]
    assert isinstance(mapping["edges"], dict) and isinstance(mapping["nodes"], dict)


def test_get_subgraph_nodes_edges_maps_virtuals(fake_detector_cpu):
    """subgraph extraction maps virtuals and includes real edges/nodes"""
    loader = DynamicLibraryLoader(fake_detector_cpu)
    pcst = MultimodalPCSTPruning(loader=loader)
    num_nodes = 5
    vertices = np.array([0, 2, 5, 6])  # includes virtuals 5,6

    # Edges here are indices (0..3). First two are "real".
    edges_indices = np.array([0, 1, 2, 3])
    edge_index = np.array(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ]
    )
    edge_bundle = {
        "edges": edges_indices,
        "num_prior_edges": 2,  # only indices <2 are treated as real
        "edge_index": edge_index,
    }

    # Map real edge indices 0,1 to existing columns (keep them in-range)
    # Map virtual vertices (>= num_nodes) to existing columns 2,3
    mapping = {"edges": {0: 0, 1: 1}, "nodes": {5: 2, 6: 3}}

    sub = pcst.get_subgraph_nodes_edges(num_nodes, vertices, edge_bundle, mapping)

    # Edges should include mapped real edges (0,1) plus mapped virtuals (2,3)
    assert set(sub["edges"].tolist()) == {0, 1, 2, 3}
    # Nodes should include unique set from real vertices + edge_index columns involved
    assert set(sub["nodes"].tolist()).issuperset({0, 1, 2, 3})


def test_extract_subgraph_pipeline(
    monkeypatch, fake_detector_cpu, patch_milvus_collection
):
    """End-to-end skeleton of extract_subgraph with its heavy deps mocked."""
    loader = DynamicLibraryLoader(fake_detector_cpu)
    pcst = MultimodalPCSTPruning(
        loader=loader, topk=2, topk_e=2, root=-1, num_clusters=1, pruning="strong"
    )

    # Mock prepare_collections to return predictable sizes
    colls = {
        "nodes": SimpleNamespace(num_entities=5),
        "edges": SimpleNamespace(num_entities=4),
    }

    def fake_prepare(cfg, modality):
        return colls

    monkeypatch.setattr(
        MultimodalPCSTPruning,
        "prepare_collections",
        staticmethod(fake_prepare),
        raising=True,
    )

    # Let load_edge_index run the real implementation for coverage
    # The test mocks Collection to handle Milvus calls
    pass

    # Mock compute_prizes → return consistent arrays
    def fake_compute_prizes(text_emb, query_emb, c):
        """compute_prizes mock"""
        return {
            "nodes": np.zeros(colls["nodes"].num_entities, dtype=np.float32),
            "edges": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        }

    monkeypatch.setattr(
        MultimodalPCSTPruning,
        "compute_prizes",
        staticmethod(fake_compute_prizes),
        raising=True,
    )

    # Mock compute_subgraph_costs → return edges_dict, prizes, costs, mapping
    # Keep mapping within the 0..3 columns of edge_index to avoid OOB
    def fake_costs(edge_index, num_nodes, prizes):
        """fake costs"""
        edges_dict = {"edges": np.array([0, 1]), "num_prior_edges": 2}
        final_prizes = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        costs = np.array([0.1, 0.2], dtype=np.float32)
        mapping = {"edges": {0: 0, 1: 1}, "nodes": {}}
        return edges_dict, final_prizes, costs, mapping

    monkeypatch.setattr(
        MultimodalPCSTPruning,
        "compute_subgraph_costs",
        staticmethod(fake_costs),
        raising=True,
    )

    # Patch pcst_fast.pcst_fast
    def fake_pcst(edges, prizes, costs, root, num_clusters, pruning, verbosity):
        """pcst_fast mock"""
        # Return vertices (some real) and edge indices [0,1]
        return [0, 1, 3], [0, 1]

    mod = importlib.import_module(
        "..utils.extractions.milvus_multimodal_pcst", package=__package__
    )
    monkeypatch.setattr(
        mod, "pcst_fast", SimpleNamespace(pcst_fast=fake_pcst), raising=True
    )

    out = pcst.extract_subgraph(
        text_emb=[0.1, 0.2],
        query_emb=[0.1, 0.2],
        modality="gene/protein",
        cfg=SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg")),
    )
    assert set(out.keys()) == {"nodes", "edges"}
    assert isinstance(out["nodes"], np.ndarray)


def test_module_import_gpu_try_block(monkeypatch):
    """
    Force the top-level `try: import cudf, cupy` to succeed by temporarily
    injecting fakes into sys.modules, then reload the module to execute those lines.
    Finally, restore to the original state by removing the fakes and reloading again.
    """

    # Inject fakes so import succeeds
    class _FakeCP:
        """fake cupy"""

        float32 = np.float32

        # @staticmethod
        # def asarray(x):
        #     return np.asarray(x)

        # class linalg:
        #     @staticmethod
        #     def norm(x, axis=None, keepdims=False):
        #         return np.linalg.norm(x, axis=axis, keepdims=keepdims)

    class _FakeCuDF:
        """fake cudf"""

        DataFrame = pd.DataFrame
        concat = staticmethod(pd.concat)

    monkeypatch.setitem(sys.modules, "cupy", _FakeCP)
    monkeypatch.setitem(sys.modules, "cudf", _FakeCuDF)

    mod = importlib.import_module(
        "..utils.extractions.milvus_multimodal_pcst", package=__package__
    )
    mod = importlib.reload(mod)  # executes lines 18–20

    assert getattr(mod, "CUDF_AVAILABLE", False) is True
    assert mod.cp is _FakeCP
    assert mod.cudf is _FakeCuDF

    # Clean up: remove fakes and reload once more to restore original state for other tests
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.delitem(sys.modules, "cudf", raising=False)
    mod2 = importlib.reload(mod)
    # After cleanup, CUDF_AVAILABLE may be False (depending on env). We don't assert it.


def test_system_detector_init_and_methods(monkeypatch):
    """successful detection of Linux + NVIDIA GPU environment"""

    mod = importlib.import_module(
        "..utils.extractions.milvus_multimodal_pcst", package=__package__
    )

    # Mock platform and subprocess to simulate a Linux + NVIDIA environment
    monkeypatch.setattr(mod.platform, "system", lambda: "Linux", raising=True)
    monkeypatch.setattr(mod.platform, "machine", lambda: "x86_64", raising=True)

    class _Ret:
        """return object"""

        def __init__(self, rc):
            """init"""
            self.returncode = rc

    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **k: _Ret(0), raising=True
    )  # nvidia-smi present

    det = mod.SystemDetector()  # executes lines 35–46 + _detect_nvidia_gpu try path
    info = det.get_system_info()  # line 65
    assert info["os_type"] == "linux"
    assert info["architecture"] == "x86_64"
    assert info["has_nvidia_gpu"] is True
    assert info["use_gpu"] is True

    # line 74
    assert det.is_gpu_compatible() is True


def test_system_detector_detect_gpu_exception_path(monkeypatch):
    """system detector handles exception in subprocess.run gracefully"""

    mod = importlib.import_module(
        "..utils.extractions.milvus_multimodal_pcst", package=__package__
    )

    # Force macOS + exception in subprocess.run → has_nvidia_gpu False; use_gpu False (no CUDA on macOS)
    monkeypatch.setattr(mod.platform, "system", lambda: "Darwin", raising=True)
    monkeypatch.setattr(mod.platform, "machine", lambda: "arm64", raising=True)

    def _boom(*a, **k):
        """crash"""
        raise FileNotFoundError("no nvidia-smi")

    monkeypatch.setattr(mod.subprocess, "run", _boom, raising=True)

    det = (
        mod.SystemDetector()
    )  # executes __init__ + exception branch in _detect_nvidia_gpu
    assert det.has_nvidia_gpu is False
    assert det.use_gpu is False
    # Also verify the helper methods
    assert det.is_gpu_compatible() is False
    info = det.get_system_info()
    assert info["use_gpu"] is False


def test_dynamic_loader_gpu_fallback_when_no_cudf(monkeypatch):
    """dynamic loader falls back to CPU mode when CUDF is not available"""
    # Build a detector that *thinks* GPU is available
    det = SimpleNamespace(
        os_type="linux", architecture="x86_64", has_nvidia_gpu=True, use_gpu=True
    )

    # Ensure CUDF_AVAILABLE is False in the module to trigger the fallback branch

    mod = importlib.import_module(
        "..utils.extractions.milvus_multimodal_pcst", package=__package__
    )
    monkeypatch.setattr(mod, "CUDF_AVAILABLE", False, raising=True)

    loader = mod.DynamicLibraryLoader(det)  # should hit lines 119–122
    # After fallback, loader should be in CPU mode
    assert loader.use_gpu is False
    assert loader.metric_type == "COSINE"
    assert loader.normalize_vectors is False


def test_normalize_matrix_bottom_return_path(fake_detector_cpu):
    """normalize_matrix takes the bottom return path when use_gpu is False"""
    # Start in CPU mode (use_gpu False), but force normalize_vectors True to skip the early return
    loader = DynamicLibraryLoader(fake_detector_cpu)
    loader.normalize_vectors = True  # override to enter the GPU-path check
    loader.use_gpu = False  # ensure we take the final `return matrix` at line 145

    m = np.array([[1.0, 2.0, 2.0]], dtype=np.float32)
    out = loader.normalize_matrix(m, axis=1)
    # Should be unchanged because use_gpu is False → bottom return path
    assert np.allclose(out, m)


def test_to_list_to_arrow_and_default_paths(fake_detector_cpu):
    """library loader to_list handles to_arrow and default paths"""
    loader = DynamicLibraryLoader(fake_detector_cpu)

    class _ArrowObj:
        """arrow-like object"""

        def __init__(self, data):
            """init"""
            self._data = data

        def to_pylist(self):
            """pylist method"""
            return list(self._data)

    class _HasToArrow:
        """has to_arrow method"""

        def __init__(self, data):
            """init"""
            self._arrow = _ArrowObj(data)

        def to_arrow(self):
            """arrow method"""
            return self._arrow

    # `to_arrow` path
    obj = _HasToArrow((1, 2, 3))
    assert loader.to_list(obj) == [1, 2, 3]

    # generic fallback to list()
    assert loader.to_list((4, 5)) == [4, 5]
