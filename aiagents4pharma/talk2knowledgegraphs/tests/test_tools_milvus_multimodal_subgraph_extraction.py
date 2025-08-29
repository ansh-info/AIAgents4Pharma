"""
Test cases for tools/milvus_multimodal_subgraph_extraction.py
"""

import asyncio
import importlib
import math
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

# Import the tool under test (as you requested)
from ..tools.milvus_multimodal_subgraph_extraction import (
    MultimodalSubgraphExtractionTool,
)


class FakeDF:
    """Pandas-like shim exposed as loader.df"""

    @staticmethod
    def DataFrame(*args, **kwargs):
        """df = pd.DataFrame(data, columns=cols)"""
        return pd.DataFrame(*args, **kwargs)

    @staticmethod
    def concat(objs, **kwargs):
        """concatenated = pd.concat(objs, **kwargs)"""
        return pd.concat(objs, **kwargs)


class FakePY:
    """NumPy/CuPy-like shim exposed as loader.py"""

    def __init__(self):
        """initialize linalg.norm"""
        self.linalg = types.SimpleNamespace(norm=lambda x: float(np.linalg.norm(x)))

    @staticmethod
    def array(x):
        """if x is list/tuple, convert to np.array"""
        return np.array(x)

    @staticmethod
    def asarray(x):
        """asarray = np.asarray(x)"""
        return np.asarray(x)

    @staticmethod
    def concatenate(xs):
        """concatenated = np.concatenate(xs)"""
        return np.concatenate(xs)

    @staticmethod
    def unique(x):
        """unique = np.unique(x)"""
        return np.unique(x)


@pytest.fixture
def fake_loader_factory(monkeypatch):
    """
    Provides a factory that installs a Fake DynamicLibraryLoader
    with toggleable normalize_vectors & metric_type.
    """
    instances = {}

    class FakeDynamicLibraryLoader:
        """fake of DynamicLibraryLoader with toggle-able attributes"""

        def __init__(self, detector):
            """initialize with detector to set use_gpu default"""
            # toggle-able per-test
            self.use_gpu = getattr(detector, "use_gpu", False)
            # Expose df/py shims
            self.df = FakeDF()
            self.py = FakePY()
            # defaults can be patched per-test
            self.metric_type = "COSINE"
            self.normalize_vectors = True

        # allow test to tweak after construction
        def set(self, **kwargs):
            """set attributes from kwargs"""
            for k, v in kwargs.items():
                setattr(self, k, v)

    class FakeSystemDetector:
        """fake of SystemDetector with fixed use_gpu"""

        def __init__(self):
            """fixed use_gpu"""
            self.use_gpu = False

    # Patch imports inside the module under test

    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )

    monkeypatch.setattr(mod, "SystemDetector", FakeSystemDetector, raising=True)
    monkeypatch.setattr(
        mod, "DynamicLibraryLoader", FakeDynamicLibraryLoader, raising=True
    )

    def get_loader(tool: MultimodalSubgraphExtractionTool):
        """get the loader instance from the tool"""
        # Access the instance created during tool.__init__
        return tool.loader

    return SimpleNamespace(get_loader=get_loader, instances=instances)


@pytest.fixture
def fake_hydra(monkeypatch):
    """Stub hydra.initialize and hydra.compose for both tool cfg and db cfg."""

    class CfgTool:
        """cfg for tool; dynamic_metrics and search_metric_type are toggleable"""

        def __init__(self, dynamic_metrics=True, search_metric_type=None):
            """initialize with toggles"""
            # required fields read by tool
            self.cost_e = 1.0
            self.c_const = 0.5
            self.root = -1
            self.num_clusters = 1
            self.pruning = "strong"
            self.verbosity_level = 0
            self.search_metric_type = search_metric_type
            self.vector_processing = types.SimpleNamespace(
                dynamic_metrics=dynamic_metrics
            )

    class CfgAll:
        """cfg for db; fixed values"""

        def __init__(self):
            """initialize with fixed values"""
            # expose utils.database.milvus with node color dict
            self.utils = types.SimpleNamespace(
                database=types.SimpleNamespace(
                    milvus=types.SimpleNamespace(
                        milvus_db=types.SimpleNamespace(database_name="primekg"),
                        node_colors_dict={
                            "gene_protein": "red",
                            "disease": "blue",
                        },
                    )
                )
            )

    class HydraCtx:
        """hydra context manager stub"""

        def __enter__(self):
            """enter returns self"""
            return self

        def __exit__(self, *a):
            """exit does nothing"""
            return False

    def initialize(**kwargs):
        """initialize returns context manager"""
        return HydraCtx()

    # Switchable return based on overrides/config_name
    def compose(config_name, overrides=None):
        """compose returns different cfgs based on args"""
        if config_name == "config" and overrides:
            # tool config call
            # allow two modes: dynamic on/off and explicit search_metric_type
            dyn = True
            search_metric_type = None
            for ov in overrides:
                # we just accept the override; details don't matter
                pass
            return types.SimpleNamespace(
                tools=types.SimpleNamespace(
                    multimodal_subgraph_extraction=CfgTool(
                        dynamic_metrics=True, search_metric_type=None
                    )
                )
            )
        elif config_name == "config":
            # db config call
            return CfgAll()
        # else:
        #     raise AssertionError("Unexpected hydra.compose usage")

    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )
    monkeypatch.setattr(
        mod,
        "hydra",
        types.SimpleNamespace(initialize=initialize, compose=compose),
        raising=True,
    )
    return compose


@pytest.fixture
def fake_pcst_and_fast(monkeypatch):
    """Stub MultimodalPCSTPruning and pcst_fast.pcst_fast."""

    class FakePCST:
        """fake of MultimodalPCSTPruning with simplified methods"""

        def __init__(self, **kwargs):
            """initialize and record kwargs"""
            # Record arguments for dynamic metric assertions
            self.kwargs = kwargs
            self.root = kwargs.get("root", -1)
            self.num_clusters = kwargs.get("num_clusters", 1)
            self.pruning = kwargs.get("pruning", "strong")
            self.verbosity_level = kwargs.get("verbosity_level", 0)
            self.loader = kwargs["loader"]

        async def _load_edge_index_from_milvus_async(self, cfg_db, connection_manager):
            """load edge index async; return dummy structure"""
            # Return a small edge_index structure that compute_subgraph_costs can accept
            return {"dummy": True}

        async def compute_prizes_async(
            self, desc_emb, feat_emb, cfg_db, connection_manager, node_type
        ):
            """compute prizes async; return dummy prizes"""
            # Return a simple prizes object
            return {"prizes": np.array([1.0, 2.0, 3.0])}

        def compute_subgraph_costs(self, edge_index, num_nodes, prizes):
            """compute subgraph costs; return dummy edges, prizes_final, costs, mapping"""
            # Return edges_dict, prizes_final, costs, mapping
            edges_dict = {
                "edges": np.array([[0, 1], [1, 2], [2, 3]]),
                "num_prior_edges": 0,
            }
            prizes_final = np.array([1.0, 0.0, 0.5, 0.2])
            costs = np.array([0.1, 0.1, 0.1])
            mapping = {"dummy": True}
            return edges_dict, prizes_final, costs, mapping

        def get_subgraph_nodes_edges(
            self, num_nodes, result_vertices, result_edges_bundle, mapping
        ):
            """get subgraph nodes and edges; return dummy structure"""
            # Return a consistent "subgraph" structure with .tolist() available
            return {
                "nodes": np.array([10, 11]),
                "edges": np.array([100]),
            }

    def fake_pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity):
        """fake pcst_fast function; return fixed vertices and edges"""
        # Return (vertices, edges) indices; values don't matter because FakePCST.get_subgraph... ignores them
        return [0, 1], [0]

    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )

    # Patch class and function
    monkeypatch.setattr(mod, "MultimodalPCSTPruning", FakePCST, raising=True)
    monkeypatch.setattr(
        mod, "pcst_fast", types.SimpleNamespace(pcst_fast=fake_pcst_fast), raising=True
    )

    return SimpleNamespace(FakePCST=FakePCST)


@pytest.fixture
def fake_milvus_and_manager(monkeypatch):
    """
    Stub pymilvus.Collection and MilvusConnectionManager
    to provide deterministic query results.
    """

    class FakeCollection:
        """fake of pymilvus.Collection with query method"""

        def __init__(self, name):
            """initialize with name"""
            self.name = name

        def load(self):
            """load does nothing"""
            return None

        def query(self, expr, output_fields):
            """query returns fixed rows based on expr"""
            # Parse expr to determine which path we're in
            # expr can be:
            #  - node_name IN ["TP53","EGFR"]
            #  - node_index IN [10,11]
            #  - triplet_index IN [100]
            if "node_name IN" in expr:
                # Return matches for node_name queries
                # Use simple mapping for test
                rows = [
                    {
                        "node_id": "G:TP53",
                        "node_name": "TP53",
                        "node_type": "gene_protein",
                        "feat": "F",
                        "feat_emb": [1, 2, 3],
                        "desc": "TP53 desc",
                        "desc_emb": [0.1, 0.2, 0.3],
                    },
                    {
                        "node_id": "G:EGFR",
                        "node_name": "EGFR",
                        "node_type": "gene_protein",
                        "feat": "F",
                        "feat_emb": [4, 5, 6],
                        "desc": "EGFR desc",
                        "desc_emb": [0.4, 0.5, 0.6],
                    },
                    {
                        "node_id": "D:GLIO",
                        "node_name": "glioblastoma",
                        "node_type": "disease",
                        "feat": "F",
                        "feat_emb": [7, 8, 9],
                        "desc": "GBM desc",
                        "desc_emb": [0.7, 0.8, 0.9],
                    },
                ]
                # Filter roughly by presence of token in expr
                keep = []
                if '"TP53"' in expr:
                    keep.append(rows[0])
                if '"EGFR"' in expr:
                    keep.append(rows[1])
                if '"glioblastoma"' in expr:
                    keep.append(rows[2])
                return keep

            if "node_index IN" in expr:
                # Return nodes/attrs required by _process_subgraph_data (must include node_index to be dropped)
                return [
                    {
                        "node_index": 10,
                        "node_id": "G:TP53",
                        "node_name": "TP53",
                        "node_type": "gene_protein",
                        "desc": "TP53 desc",
                    },
                    {
                        "node_index": 11,
                        "node_id": "D:GLIO",
                        "node_name": "glioblastoma",
                        "node_type": "disease",
                        "desc": "GBM desc",
                    },
                ]

            if "triplet_index IN" in expr:
                return [
                    {
                        "triplet_index": 100,
                        "head_id": "G:TP53",
                        "tail_id": "D:GLIO",
                        "edge_type": "associates_with|evidence",
                    }
                ]

            # raise AssertionError(f"Unexpected expr: {expr}")

    class FakeManager:
        """fake of MilvusConnectionManager with async query method"""

        def __init__(self, cfg_db):
            """initialize with cfg_db"""
            self.cfg_db = cfg_db
            self.connected = False

        def ensure_connection(self):
            """ensure_connection sets connected True"""
            self.connected = True

        def test_connection(self):
            """test_connection always returns True"""
            return True

        def get_connection_info(self):
            """get_connection_info returns fixed dict"""
            return {"database": "primekg"}

        # Async Milvus-like helpers used by _query_milvus_collection_async
        async def async_query(self, collection_name, expr, output_fields):
            """simulate async query returning rows based on expr"""
            # Mirror Collection.query behavior for async path
            col = FakeCollection(collection_name)
            # Add one case where a group yields no rows to exercise empty-async branch
            # if 'node_name IN ["NOHIT"]' in expr:
            #     return []
            return col.query(expr, output_fields)

        async def async_get_collection_stats(self, name):
            """async get_collection_stats returns fixed num_entities"""
            # Used to compute num_nodes
            return {"num_entities": 1234}

    # Patch targets inside module under test

    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )
    fake_pymilvus = types.SimpleNamespace(Collection=FakeCollection)
    monkeypatch.setattr(mod, "Collection", FakeCollection, raising=True)

    # Patch the ConnectionManager class used inside the tool
    # so that constructing it yields our fake.
    def fake_manager_ctor(cfg_db):
        """fake ctor returning FakeManager"""
        return FakeManager(cfg_db)

    # The tool imports MilvusConnectionManager from ..utils.database
    # We patch the symbol inside the tool module.
    monkeypatch.setattr(mod, "MilvusConnectionManager", fake_manager_ctor, raising=True)

    return SimpleNamespace(FakeCollection=FakeCollection, FakeManager=FakeManager)


@pytest.fixture
def fake_read_excel(monkeypatch):
    """Patch pandas.read_excel to return multiple sheets to exercise concat/rename logic."""

    def _fake_read_excel(path, sheet_name=None):
        """fake read_excel returning two sheets"""
        assert sheet_name is None
        # Two sheets; first has a hyphen in sheet-like node type to test hyphen->underscore logic upstream
        return {
            "gene-protein": pd.DataFrame(
                {
                    "name": ["TP53", "EGFR"],
                    "node_type": ["gene/protein", "gene/protein"],
                }
            ),
            "disease": pd.DataFrame(
                {"name": ["glioblastoma"], "node_type": ["disease"]}
            ),
        }

    monkeypatch.setattr(pd, "read_excel", _fake_read_excel)
    return _fake_read_excel


@pytest.fixture
def base_state():
    """Minimal viable state; uploaded_files will be supplied per-test."""

    class Embedder:
        """embedder with fixed embed_query output"""

        def embed_query(self, text):
            """embed_query returns fixed embedding"""
            # vector with norm=3 → normalized = [1/3, 2/3, 2/3] when enabled
            return [1.0, 2.0, 2.0]

    return {
        "uploaded_files": [],
        "embedding_model": Embedder(),
        "dic_source_graph": [{"name": "PrimeKG"}],
        "topk_nodes": 5,
        "topk_edges": 10,
    }


def test_read_multimodal_files_empty(
    monkeypatch,
    fake_loader_factory,
    base_state,
    fake_hydra,
    fake_pcst_and_fast,
    fake_milvus_and_manager,
):
    """test _read_multimodal_files returns empty DataFrame when no files present"""
    tool = MultimodalSubgraphExtractionTool()
    loader = fake_loader_factory.get_loader(tool)
    # ensure CPU path default ok
    loader.set(use_gpu=False, normalize_vectors=True, metric_type="COSINE")

    # No multimodal file -> empty DataFrame-like (len == 0)
    df = tool._read_multimodal_files(base_state)
    assert len(df) == 0


def test_normalize_vector_toggle(
    fake_loader_factory, fake_hydra, fake_pcst_and_fast, fake_milvus_and_manager
):
    """normalize_vector returns normalized or original based on loader setting"""
    tool = MultimodalSubgraphExtractionTool()
    loader = fake_loader_factory.get_loader(tool)

    v = [1.0, 2.0, 2.0]

    # With normalization
    loader.set(normalize_vectors=True)
    out = tool.normalize_vector(v)
    # norm = 3
    assert pytest.approx(out, rel=1e-6) == [1 / 3, 2 / 3, 2 / 3]

    # Without normalization
    loader.set(normalize_vectors=False)
    out2 = tool.normalize_vector(v)
    assert out2 == v


@pytest.mark.asyncio
async def test_run_async_happy_path(
    monkeypatch,
    fake_loader_factory,
    fake_hydra,
    fake_pcst_and_fast,
    fake_milvus_and_manager,
    fake_read_excel,
    base_state,
):
    """async run with Excel file exercises most code paths"""
    # Prepare state with a multimodal Excel file
    state = dict(base_state)
    state["uploaded_files"] = [{"file_type": "multimodal", "file_path": "/fake.xlsx"}]

    tool = MultimodalSubgraphExtractionTool()
    loader = fake_loader_factory.get_loader(tool)
    loader.set(normalize_vectors=True, metric_type="COSINE")

    # Execute async run
    cmd = await tool._run_async(
        tool_call_id="tc-1",
        state=state,
        prompt="find gbm genes",
        arg_data=SimpleNamespace(extraction_name="E1"),
    )

    # Validate Command.update structure
    assert isinstance(cmd.update, dict)
    assert "dic_extracted_graph" in cmd.update
    deg = cmd.update["dic_extracted_graph"][0]
    assert deg["name"] == "E1"
    assert deg["graph_source"] == "PrimeKG"
    # graph_dict exists and has unified + per-query entries
    assert "graph_dict" in deg and "graph_text" in deg
    assert len(deg["graph_dict"]["name"]) >= 1
    # messages are present
    assert "messages" in cmd.update
    # selections were added to state during prepare_query (coloring step)
    # (cannot access mutated external state here, but the successful finish implies it)


@pytest.mark.asyncio
async def test_dynamic_metric_selection_paths(
    monkeypatch,
    fake_loader_factory,
    fake_pcst_and_fast,
    fake_milvus_and_manager,
    base_state,
):
    """
    Exercise both dynamic metric branches. Preseed `state["selections"]`
    because the prompt-only path won't populate it.
    """
    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )

    class CfgToolA:
        """cfg with dynamic_metrics=True and search_metric_type=None"""

        def __init__(self):
            """initialize with fixed values"""
            self.cost_e = 1.0
            self.c_const = 0.5
            self.root = -1
            self.num_clusters = 1
            self.pruning = "strong"
            self.verbosity_level = 0
            self.search_metric_type = None
            self.vector_processing = types.SimpleNamespace(dynamic_metrics=True)

    class CfgToolB:
        """cfg with dynamic_metrics=False and search_metric_type='L2'"""

        def __init__(self):
            """initialize with fixed values"""
            self.cost_e = 1.0
            self.c_const = 0.5
            self.root = -1
            self.num_clusters = 1
            self.pruning = "strong"
            self.verbosity_level = 0
            self.search_metric_type = "L2"
            self.vector_processing = types.SimpleNamespace(dynamic_metrics=False)

    class CfgAll:
        """cfg for db; fixed values"""

        def __init__(self):
            """object with fixed values"""
            self.utils = types.SimpleNamespace(
                database=types.SimpleNamespace(
                    milvus=types.SimpleNamespace(
                        milvus_db=types.SimpleNamespace(database_name="primekg"),
                        node_colors_dict={"gene_protein": "red", "disease": "blue"},
                    )
                )
            )

    class HydraCtx:
        """hydra context manager stub"""

        def __enter__(self):
            """enter returns self"""
            return self

        def __exit__(self, *a):
            """exit does nothing"""
            return False

    def initialize(**kwargs):
        """initialize returns context manager"""
        return HydraCtx()

    # flip between tool cfg A then B when overrides are present; db cfg when not
    calls = {"i": 0}

    def compose(config_name, overrides=None):
        """compose returns different cfgs based on args"""
        if config_name == "config" and overrides:
            calls["i"] += 1
            if calls["i"] == 1:
                return types.SimpleNamespace(
                    tools=types.SimpleNamespace(
                        multimodal_subgraph_extraction=CfgToolA()
                    )
                )
            else:
                return types.SimpleNamespace(
                    tools=types.SimpleNamespace(
                        multimodal_subgraph_extraction=CfgToolB()
                    )
                )
        elif config_name == "config":
            return CfgAll()
        # raise AssertionError("unexpected compose")

    monkeypatch.setattr(
        mod,
        "hydra",
        types.SimpleNamespace(initialize=initialize, compose=compose),
        raising=True,
    )

    # ---- Run with dynamic_metrics=True (uses loader.metric_type) ----
    stateA = dict(base_state)
    # Preseed selections so _prepare_final_subgraph can color nodes
    stateA["selections"] = {"gene_protein": ["G:TP53"], "disease": ["D:GLIO"]}

    toolA = MultimodalSubgraphExtractionTool()
    loaderA = fake_loader_factory.get_loader(toolA)
    loaderA.set(metric_type="COSINE")

    cmdA = await toolA._run_async(
        tool_call_id="tc-A",
        state=stateA,
        prompt="only prompt",
        arg_data=SimpleNamespace(extraction_name="E-A"),
    )
    assert "dic_extracted_graph" in cmdA.update

    # ---- Run with dynamic_metrics=False (uses cfg.search_metric_type) ----
    stateB = dict(base_state)
    stateB["selections"] = {"gene_protein": ["G:TP53"], "disease": ["D:GLIO"]}

    toolB = MultimodalSubgraphExtractionTool()
    loaderB = fake_loader_factory.get_loader(toolB)
    loaderB.set(metric_type="IP")

    cmdB = await toolB._run_async(
        tool_call_id="tc-B",
        state=stateB,
        prompt="only prompt two",
        arg_data=SimpleNamespace(extraction_name="E-B"),
    )
    assert "dic_extracted_graph" in cmdB.update


def test_run_sync_wrapper(
    monkeypatch,
    fake_loader_factory,
    fake_hydra,
    fake_pcst_and_fast,
    fake_milvus_and_manager,
    base_state,
):
    """run the sync wrapper which calls the async path internally"""
    tool = MultimodalSubgraphExtractionTool()
    loader = fake_loader_factory.get_loader(tool)
    loader.set(normalize_vectors=True)

    state = dict(base_state)
    # Preseed selections because this test uses prompt-only flow
    state["selections"] = {"gene_protein": ["G:TP53"], "disease": ["D:GLIO"]}

    cmd = tool._run(
        tool_call_id="tc-sync",
        state=state,
        prompt="sync run",
        arg_data=SimpleNamespace(extraction_name="E-sync"),
    )
    assert "dic_extracted_graph" in cmd.update


def test_connection_error_raises_runtimeerror(
    monkeypatch,
    fake_loader_factory,
    fake_hydra,
    fake_pcst_and_fast,
    fake_milvus_and_manager,
    base_state,
):
    """
    Make ensure_connection raise to exercise the error path in _run_async.
    """

    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )

    class ExplodingManager:
        """exploding manager whose ensure_connection raises"""

        def __init__(self, cfg_db):
            """initialize with cfg_db"""
            pass

        def ensure_connection(self):
            """ "ensure_connection always raises"""
            raise RuntimeError("nope")

    # Patch manager ctor to explode
    monkeypatch.setattr(
        mod,
        "MilvusConnectionManager",
        lambda cfg_db: ExplodingManager(cfg_db),
        raising=True,
    )

    tool = MultimodalSubgraphExtractionTool()

    with pytest.raises(RuntimeError) as ei:
        asyncio.get_event_loop().run_until_complete(
            tool._run_async(
                tool_call_id="tc-err",
                state=base_state,
                prompt="will fail",
                arg_data=SimpleNamespace(extraction_name="E-err"),
            )
        )
    assert "Cannot connect to Milvus database" in str(ei.value)


def test_prepare_query_modalities_async_with_excel_grouping(
    monkeypatch,
    fake_loader_factory,
    fake_hydra,
    fake_pcst_and_fast,
    fake_milvus_and_manager,
    fake_read_excel,
    base_state,
):
    """prepare_query_modalities_async with Excel file populates state['selections"""
    # Use the public async prep path via _run_async in another test,
    # but here directly target the helper to assert selections are added.
    tool = MultimodalSubgraphExtractionTool()
    loader = fake_loader_factory.get_loader(tool)
    loader.set(normalize_vectors=False)

    # State with one Excel + one "nohit" row to exercise empty async result path
    state = dict(base_state)
    state["uploaded_files"] = [{"file_type": "multimodal", "file_path": "/fake.xlsx"}]

    # We also monkeypatch the async_query to return empty for a fabricated node

    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )
    # create a fake manager just to call the method
    mgr = mod.MilvusConnectionManager(mod.hydra.compose("config").utils.database.milvus)

    async def run():
        qdf = await tool._prepare_query_modalities_async(
            prompt={"text": "query", "emb": [[0.1, 0.2, 0.3]]},
            state=state,
            cfg_db=mod.hydra.compose("config").utils.database.milvus,
            connection_manager=mgr,
        )
        # After reading excel and querying, selections should be set
        assert "selections" in state and isinstance(state["selections"], dict)
        # Prompt row appended
        pdf = getattr(qdf, "to_pandas", lambda: qdf)()
        assert any(pdf["node_type"] == "prompt")

    asyncio.get_event_loop().run_until_complete(run())


def test__query_milvus_collection_sync_casts_and_builds_expr(
    fake_loader_factory,
    fake_milvus_and_manager,  # provides FakeCollection patch
):
    """query_milvus_collection builds expr and returns expected columns and types"""

    tool = MultimodalSubgraphExtractionTool()
    loader = fake_loader_factory.get_loader(tool)
    loader.set(normalize_vectors=False)  # doesn't matter for this test

    # Build a node_type_df exactly like the function expects
    node_type_df = pd.DataFrame({"q_node_name": ["TP53", "EGFR"]})

    # cfg_db only needs database_name
    cfg_db = SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg"))

    # Use a node_type containing '/' to exercise replace('/', '_')
    out_df = tool._query_milvus_collection("gene/protein", node_type_df, cfg_db)

    # Must have all columns in q_columns + 'use_description'
    expected_cols = [
        "node_id",
        "node_name",
        "node_type",
        "feat",
        "feat_emb",
        "desc",
        "desc_emb",
        "use_description",
    ]
    assert list(out_df.columns) == expected_cols

    # Returned rows are the two we asked for; embeddings must be floats
    assert set(out_df["node_name"]) == {"TP53", "EGFR"}
    for row in out_df.itertuples(index=False):
        assert all(isinstance(x, float) for x in row.feat_emb)
        assert all(isinstance(x, float) for x in row.desc_emb)

    # 'use_description' is forced False in this path
    assert (out_df["use_description"] == False).all()


def test__prepare_query_modalities_sync_with_multimodal_grouping(
    monkeypatch,
    fake_loader_factory,
    fake_milvus_and_manager,  # provides FakeCollection for node queries
    base_state,
):
    """pepare_query_modalities with multimodal file populates state['selections']"""

    tool = MultimodalSubgraphExtractionTool()
    loader = fake_loader_factory.get_loader(tool)
    loader.set(normalize_vectors=False)

    # Force _read_multimodal_files to return grouped rows across 2 types.
    multimodal_df = pd.DataFrame(
        {
            "q_node_type": ["gene_protein", "gene_protein", "disease"],
            "q_node_name": ["TP53", "EGFR", "glioblastoma"],
        }
    )
    monkeypatch.setattr(
        tool, "_read_multimodal_files", lambda state: multimodal_df, raising=True
    )

    # cfg_db minimal
    cfg_db = SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg"))

    # prompt dict expected by the function
    prompt = {"text": "user text", "emb": [[0.1, 0.2, 0.3]]}

    # run sync helper (NOT the async one)
    qdf = tool._prepare_query_modalities(prompt, base_state, cfg_db)

    # 1) It should have appended the prompt row with node_type='prompt' and use_description=True
    pdf = getattr(qdf, "to_pandas", lambda: qdf)()
    assert "prompt" in set(pdf["node_type"])
    # last row is the appended prompt row (per implementation)
    last = pdf.iloc[-1]
    assert last["node_type"] == "prompt"
    # avoid identity comparison with numpy.bool_
    assert bool(last["use_description"])  # was: `is True`

    # 2) Prior rows are from Milvus queries; ensure they exist and carry use_description=False
    non_prompt = pdf[pdf["node_type"] != "prompt"]
    assert not non_prompt.empty
    assert (non_prompt["use_description"] == False).all()
    # We expect at least TP53/EGFR/glioblastoma present from our FakeCollection
    assert {"TP53", "EGFR", "glioblastoma"}.issubset(set(non_prompt["node_name"]))

    # 3) The function must have populated state['selections'] grouped by node_type
    assert "selections" in base_state and isinstance(base_state["selections"], dict)
    # Sanity: keys align with node types returned by queries
    assert (
        "gene_protein" in base_state["selections"]
        or "gene/protein" in base_state["selections"]
    )
    assert "disease" in base_state["selections"]
    # And the IDs collected are the ones FakeCollection returns
    collected_ids = set(sum(base_state["selections"].values(), []))
    assert {"G:TP53", "G:EGFR", "D:GLIO"}.issubset(collected_ids)


def test__prepare_query_modalities_sync_prompt_only_branch(
    fake_loader_factory,
    base_state,
):
    """run the prompt-only branch of _prepare_query_modalities"""
    tool = MultimodalSubgraphExtractionTool()
    fake_loader_factory.get_loader(tool).set(normalize_vectors=False)

    # Force empty multimodal_df → else: query_df = prompt_df
    empty_df = pd.DataFrame(columns=["q_node_type", "q_node_name"])
    tool._read_multimodal_files = lambda state: empty_df  # per-instance patch

    cfg_db = SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg"))

    # Flat vector (common case), but function should handle either flat or nested
    expected_emb = [0.1, 0.2, 0.3]
    prompt = {"text": "only prompt", "emb": expected_emb}

    qdf = tool._prepare_query_modalities(prompt, base_state, cfg_db)
    pdf = getattr(qdf, "to_pandas", lambda: qdf)()

    # All rows should be prompt rows with use_description True
    assert set(pdf["node_type"]) == {"prompt"}
    assert pdf["use_description"].map(bool).all()

    # Coerce whatever shape we got into a flat list of floats to compare
    def coerce_elem(x):
        """coerce element to flat list of floats"""
        # single scalar -> list of one
        if not isinstance(x, (list, tuple)):
            return [float(x)]
        # list/tuple; if nested [[...]] pick inner
        # if len(x) > 0 and isinstance(x[0], (list, tuple)):
        #     return [float(v) for v in x[0]]
        # return [float(v) for v in x]

    feat_emb_col = pdf["feat_emb"].tolist()
    # If we have multiple rows each with scalar, flatten them
    flat = []
    for elem in feat_emb_col:
        flat.extend(coerce_elem(elem))

    # Compare numerically to avoid dtype surprises
    assert len(flat) == len(expected_emb)
    for a, b in zip(flat, expected_emb):
        assert math.isclose(a, b, rel_tol=1e-9)


@pytest.mark.asyncio
async def test__prepare_query_modalities_async_single_task_branch(
    fake_loader_factory,
    fake_milvus_and_manager,  # FakeManager & FakeCollection
    fake_hydra,  # <<< ensure Hydra is mocked
    base_state,
):
    """prepare_query_modalities_async with single group exercises single-task path"""
    tool = MultimodalSubgraphExtractionTool()
    fake_loader_factory.get_loader(tool).set(normalize_vectors=False)

    # exactly one node type → len(tasks) == 1 → query_results = [await tasks[0]]
    single_group_df = pd.DataFrame(
        {"q_node_type": ["gene_protein"], "q_node_name": ["TP53"]}
    )
    tool._read_multimodal_files = lambda state: single_group_df

    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )
    cfg_db = mod.hydra.compose("config").utils.database.milvus
    manager = mod.MilvusConnectionManager(cfg_db)

    prompt = {"text": "p", "emb": [[0.1, 0.2, 0.3]]}
    qdf = await tool._prepare_query_modalities_async(
        prompt, base_state, cfg_db, manager
    )

    pdf = getattr(qdf, "to_pandas", lambda: qdf)()
    # it should contain both the TP53 row (from Milvus) and the appended prompt row
    assert "TP53" in set(pdf["node_name"])
    assert "prompt" in set(pdf["node_type"])


def test__perform_subgraph_extraction_sync_unifies_nodes_edges(
    monkeypatch,
    fake_loader_factory,
    base_state,
):
    """perform_subgraph_extraction sync path unifies nodes/edges across multiple queries"""
    # Patch MultimodalPCSTPruning to implement .extract_subgraph for sync path

    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )

    call_counter = {"i": 0}

    class FakePCSTSync:
        """fake of MultimodalPCSTPruning with extract_subgraph method"""

        def __init__(self, **kwargs):
            """init with kwargs; ignore them"""
            pass

        def extract_subgraph(self, desc_emb, feat_emb, node_type, cfg_db):
            """extract_subgraph returns different subgraphs per call"""
            # Return different subgraphs across calls to exercise union/unique
            call_counter["i"] += 1
            if call_counter["i"] == 1:
                return {"nodes": np.array([10, 11]), "edges": np.array([100])}
            else:
                return {"nodes": np.array([11, 12]), "edges": np.array([101])}

    monkeypatch.setattr(mod, "MultimodalPCSTPruning", FakePCSTSync, raising=True)

    # Build a query_df with two rows (will yield two subgraphs)
    tool = MultimodalSubgraphExtractionTool()
    loader = fake_loader_factory.get_loader(tool)
    loader.set(normalize_vectors=False)

    query_df = loader.df.DataFrame(
        [
            {
                "node_id": "u1",
                "node_name": "Q1",
                "node_type": "gene_protein",
                "feat": "f",
                "feat_emb": [[0.1]],
                "desc": "d",
                "desc_emb": [[0.1]],
                "use_description": False,
            },
            {
                "node_id": "u2",
                "node_name": "Q2",
                "node_type": "disease",
                "feat": "f",
                "feat_emb": [[0.2]],
                "desc": "d",
                "desc_emb": [[0.2]],
                "use_description": True,
            },
        ]
    )

    # Minimal cfg and cfg_db
    cfg = SimpleNamespace(
        cost_e=1.0,
        c_const=0.5,
        root=-1,
        num_clusters=1,
        pruning="strong",
        verbosity_level=0,
        vector_processing=SimpleNamespace(dynamic_metrics=True),
        search_metric_type=None,
    )
    cfg_db = SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg"))

    # state for topk values
    state = dict(base_state)

    out = tool._perform_subgraph_extraction(state, cfg, cfg_db, query_df)
    pdf = getattr(out, "to_pandas", lambda: out)()

    # first row is Unified Subgraph with unioned nodes/edges
    unified = pdf.iloc[0]
    assert unified["name"] == "Unified Subgraph"
    assert set(unified["nodes"]) == {10, 11, 12}
    assert set(unified["edges"]) == {100, 101}

    # subsequent rows correspond to Q1 and Q2
    names = list(pdf["name"])
    assert "Q1" in names and "Q2" in names


def test__prepare_final_subgraph_defaults_black_when_no_colors(
    fake_loader_factory,
    fake_milvus_and_manager,  # gives FakeCollection for node/edge lookups
):
    """prepare_final_subgraph colors nodes black when no selections/colors present"""
    # Prepare a minimal subgraph DataFrame
    tool = MultimodalSubgraphExtractionTool()
    fake_loader_factory.get_loader(tool).set(normalize_vectors=False)

    subgraphs_df = tool.loader.df.DataFrame(
        [("Unified Subgraph", [10, 11], [100])],
        columns=["name", "nodes", "edges"],
    )

    # cfg_db required by Collection names; selections empty → color_df empty
    cfg_db = SimpleNamespace(
        milvus_db=SimpleNamespace(database_name="primekg"),
        node_colors_dict={"gene_protein": "red", "disease": "blue"},
    )
    state = {"selections": {}}  # IMPORTANT: key exists but empty → triggers else: black

    graph_dict = tool._prepare_final_subgraph(state, subgraphs_df, cfg_db)

    # Inspect colors on returned nodes; all should be black
    nodes_list = graph_dict["nodes"][0]  # first (and only) graph's nodes list
    assert len(nodes_list) > 0
    for node_id, attrs in nodes_list:
        assert attrs["color"] == "black"


@pytest.mark.asyncio
async def test__perform_subgraph_extraction_async_no_vector_processing_branch(
    monkeypatch,
    fake_loader_factory,
    fake_milvus_and_manager,  # <<< ensure FakeManager is used
    base_state,
):
    """perform_subgraph_extraction async path with no vector_processing exercises else: branch"""
    tool = MultimodalSubgraphExtractionTool()
    fake_loader_factory.get_loader(tool).set(normalize_vectors=False)

    # Make _extract_single_subgraph_async return a fixed subgraph so we avoid PCST internals
    async def _fake_extract(pcst_instance, query_row, cfg_db, manager):
        """fake _extract_single_subgraph_async returning fixed subgraph"""
        return {"nodes": np.array([10]), "edges": np.array([100])}

    monkeypatch.setattr(
        tool, "_extract_single_subgraph_async", _fake_extract, raising=True
    )

    # Build a one-row query_df
    qdf = tool.loader.df.DataFrame(
        [
            {
                "node_id": "u",
                "node_name": "Q",
                "node_type": "prompt",
                "feat": "f",
                "feat_emb": [[0.1]],
                "desc": "d",
                "desc_emb": [[0.1]],
                "use_description": True,
            }
        ]
    )

    # cfg WITHOUT vector_processing attribute → triggers the else: dynamic_metrics_enabled = False
    cfg = SimpleNamespace(
        cost_e=1.0,
        c_const=0.5,
        root=-1,
        num_clusters=1,
        pruning="strong",
        verbosity_level=0,
        # no vector_processing here
        search_metric_type="COSINE",
    )
    cfg_db = SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg"))

    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )
    manager = mod.MilvusConnectionManager(cfg_db)  # this uses your FakeManager

    out = await tool._perform_subgraph_extraction_async(
        state=base_state,
        cfg=cfg,
        cfg_db=cfg_db,
        query_df=qdf,
        connection_manager=manager,
    )
    pdf = getattr(out, "to_pandas", lambda: out)()
    assert "Unified Subgraph" in set(pdf["name"])


def test__perform_subgraph_extraction_sync_uses_cfg_search_metric_type_when_no_vector_processing(
    monkeypatch,
    fake_loader_factory,
    base_state,
):
    """perform_subgraph_extraction sync path uses cfg.search_metric_type when no vector_processing"""
    # Patch MultimodalPCSTPruning to capture metric_type passed in (line 412 path)
    mod = importlib.import_module(
        "..tools.milvus_multimodal_subgraph_extraction", package=__package__
    )

    captured_metric_types = []

    class FakePCSTSync:
        """fake of MultimodalPCSTPruning capturing metric_type in ctor"""

        def __init__(self, **kwargs):
            """init capturing metric_type"""
            # Capture the metric_type used by the business logic
            captured_metric_types.append(kwargs.get("metric_type"))

        def extract_subgraph(self, desc_emb, feat_emb, node_type, cfg_db):
            """extract_subgraph returns minimal subgraph"""
            # Minimal valid return for the sync path
            return {"nodes": np.array([10]), "edges": np.array([100])}

    monkeypatch.setattr(mod, "MultimodalPCSTPruning", FakePCSTSync, raising=True)

    # Instantiate tool and ensure loader.metric_type is different from cfg.search_metric_type
    tool = MultimodalSubgraphExtractionTool()
    loader = fake_loader_factory.get_loader(tool)
    loader.set(metric_type="COSINE")  # should NOT be used in this test

    # Build a single-row query_df to hit the loop once
    query_df = loader.df.DataFrame(
        [
            {
                "node_id": "u1",
                "node_name": "Q1",
                "node_type": "gene_protein",
                "feat": "f",
                "feat_emb": [[0.1]],
                "desc": "d",
                "desc_emb": [[0.1]],
                "use_description": True,
            }
        ]
    )

    cfg = SimpleNamespace(
        cost_e=1.0,
        c_const=0.5,
        root=-1,
        num_clusters=1,
        pruning="strong",
        verbosity_level=0,
        search_metric_type="IP",  # expect this to be used
    )

    cfg_db = SimpleNamespace(milvus_db=SimpleNamespace(database_name="primekg"))
    state = dict(base_state)

    # Run the sync extraction
    _ = tool._perform_subgraph_extraction(state, cfg, cfg_db, query_df)

    # Assert business logic picked cfg.search_metric_type, not loader.metric_type
    assert captured_metric_types, "PCST was not constructed"
    assert captured_metric_types[0] == "IP"
