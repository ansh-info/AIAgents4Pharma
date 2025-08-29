# tests/test_utils_database_milvus_connection_manager.py

import asyncio as _asyncio
import importlib
from types import SimpleNamespace

import pytest
from pymilvus.exceptions import MilvusException

from ..utils.database.milvus_connection_manager import MilvusConnectionManager


class FakeConnections:
    def __init__(self):
        self._map = {}
        self._addr = {}

    def has_connection(self, alias):
        return alias in self._map

    def connect(self, alias, host, port, user, password):
        self._map[alias] = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
        }
        self._addr[alias] = (host, port)

    def disconnect(self, alias):
        self._map.pop(alias, None)
        self._addr.pop(alias, None)

    def get_connection_addr(self, alias):
        return self._addr.get(alias, None)


class FakeDB:
    def __init__(self):
        self._using = None

    def using_database(self, name):
        self._using = name


class FakeCollection:
    registry = {}

    def __init__(self, name):
        self.name = name
        # Default num_entities for stats fallback
        self.num_entities = FakeCollection.registry.get(name, {}).get("num_entities", 7)

    def load(self):
        FakeCollection.registry.setdefault(self.name, {}).update({"loaded": True})

    def query(self, expr=None, output_fields=None, limit=None):
        return [{"id": 1}]

    def search(
        self, data=None, anns_field=None, param=None, limit=None, output_fields=None
    ):
        class Hit:
            def __init__(self, idx, score):
                self.id = idx
                self.score = score

        return [[Hit(i, 1.0 - 0.1 * i) for i in range(limit or 1)]]


class FakeSyncClient:
    def __init__(self, uri, token, db_name):
        self.uri = uri
        self.token = token
        self.db_name = db_name


class FakeAsyncClient:
    def __init__(self, uri, token, db_name):
        self.uri = uri
        self.token = token
        self.db_name = db_name
        self._closed = False

    async def load_collection(self, collection_name):
        # mark loaded in registry
        FakeCollection.registry.setdefault(collection_name, {}).update(
            {"loaded_async": True}
        )

    async def search(
        self, collection_name, data, anns_field, search_params, limit, output_fields
    ):
        return [[{"id": i, "distance": 0.1 * i} for i in range(limit)]]

    async def query(self, collection_name, filter, output_fields, limit):
        return [{"ok": True, "filter": filter}]

    async def close(self):
        self._closed = True


@pytest.fixture(autouse=True)
def patch_pymilvus(monkeypatch):
    """
    Patch the pymilvus symbols inside the module-under-test namespace.
    """

    mod = importlib.import_module(
        "..utils.database.milvus_connection_manager", package=__package__
    )

    # fresh fakes per test
    fake_conn = FakeConnections()
    fake_db = FakeDB()

    monkeypatch.setattr(mod, "connections", fake_conn, raising=True)
    monkeypatch.setattr(mod, "db", fake_db, raising=True)
    monkeypatch.setattr(mod, "Collection", FakeCollection, raising=True)
    monkeypatch.setattr(mod, "MilvusClient", FakeSyncClient, raising=True)
    monkeypatch.setattr(mod, "AsyncMilvusClient", FakeAsyncClient, raising=True)

    yield
    # cleanup
    MilvusConnectionManager.clear_instances()
    FakeCollection.registry.clear()


@pytest.fixture
def cfg():
    # minimal cfg namespace with milvus_db sub-keys used by the manager
    return SimpleNamespace(
        milvus_db=SimpleNamespace(
            host="127.0.0.1",
            port=19530,
            user="u",
            password="p",
            database_name="dbX",
            alias="default",
        )
    )


def test_singleton_and_init(cfg):
    # Two instances with same config key should be identical
    a = MilvusConnectionManager(cfg)
    b = MilvusConnectionManager(cfg)
    assert a is b
    # basic attributes initialized once
    assert a.database_name == "dbX"


def test_ensure_connection_creates_and_reuses(cfg):
    mgr = MilvusConnectionManager(cfg)
    # First call creates connection and sets db
    assert mgr.ensure_connection() is True
    # Second call should reuse
    assert mgr.ensure_connection() is True


def test_get_connection_info_connected_and_disconnected(cfg):
    mgr = MilvusConnectionManager(cfg)
    # before ensure, not connected
    info = mgr.get_connection_info()
    assert info["connected"] is False
    # after ensure, connected
    mgr.ensure_connection()
    info2 = mgr.get_connection_info()
    assert info2["connected"] is True
    assert info2["database"] == "dbX"
    assert info2["connection_address"] == ("127.0.0.1", 19530)


def test_get_sync_and_async_client(cfg):
    mgr = MilvusConnectionManager(cfg)
    c1 = mgr.get_sync_client()
    c2 = mgr.get_sync_client()
    assert c1 is c2
    a1 = mgr.get_async_client()
    a2 = mgr.get_async_client()
    assert a1 is a2


def test_test_connection_success(cfg):
    mgr = MilvusConnectionManager(cfg)
    assert mgr.test_connection() is True


def test_get_collection_success(cfg):
    mgr = MilvusConnectionManager(cfg)
    coll = mgr.get_collection("dbX_nodes")
    assert isinstance(coll, FakeCollection)
    # ensure loaded
    assert FakeCollection.registry["dbX_nodes"]["loaded"] is True


def test_get_collection_failure_raises(cfg, monkeypatch):
    mgr = MilvusConnectionManager(cfg)

    class Boom(FakeCollection):
        def load(self):
            raise RuntimeError("load failed")

    mod = importlib.import_module(
        "..utils.database.milvus_connection_manager", package=__package__
    )
    monkeypatch.setattr(mod, "Collection", Boom, raising=True)

    with pytest.raises(MilvusException):
        mgr.get_collection("dbX_nodes")


@pytest.mark.asyncio
async def test_async_search_success(cfg):
    mgr = MilvusConnectionManager(cfg)
    res = await mgr.async_search(
        collection_name="dbX_edges",
        data=[[0.1, 0.2]],
        anns_field="feat_emb",
        param={"metric_type": "COSINE"},
        limit=2,
        output_fields=["id"],
    )
    assert isinstance(res, list)
    assert len(res[0]) == 2


@pytest.mark.asyncio
async def test_async_search_falls_back_to_sync(cfg, monkeypatch):
    mgr = MilvusConnectionManager(cfg)

    # Make Async client creation fail (get_async_client returns None)
    def bad_async_client(*a, **k):
        return None

    mod = importlib.import_module(
        "..utils.database.milvus_connection_manager", package=__package__
    )
    monkeypatch.setattr(mgr, "get_async_client", bad_async_client, raising=True)

    res = await mgr.async_search(
        collection_name="dbX_edges",
        data=[[0.1, 0.2]],
        anns_field="feat_emb",
        param={"metric_type": "COSINE"},
        limit=3,
        output_fields=["id"],
    )
    # Sync fallback should produce hits
    assert len(res[0]) == 3


def test_sync_search_error_raises(cfg, monkeypatch):
    mgr = MilvusConnectionManager(cfg)

    class Boom(FakeCollection):
        def load(self):
            pass

        def search(self, *a, **k):
            raise RuntimeError("sync search fail")

    mod = importlib.import_module(
        "..utils.database.milvus_connection_manager", package=__package__
    )
    monkeypatch.setattr(mod, "Collection", Boom, raising=True)

    with pytest.raises(MilvusException):
        mgr._sync_search(
            "dbX_edges", [[0.1]], "feat_emb", {"metric_type": "COSINE"}, 1, ["id"]
        )


@pytest.mark.asyncio
async def test_async_query_success(cfg):
    mgr = MilvusConnectionManager(cfg)
    res = await mgr.async_query(
        collection_name="dbX_nodes", expr="id > 0", output_fields=["id"], limit=1
    )
    assert isinstance(res, list)
    assert res[0]["ok"] is True


@pytest.mark.asyncio
async def test_async_query_falls_back_to_sync(cfg, monkeypatch):
    mgr = MilvusConnectionManager(cfg)

    def bad_async_client(*a, **k):
        return None

    monkeypatch.setattr(mgr, "get_async_client", bad_async_client, raising=True)

    res = await mgr.async_query(
        collection_name="dbX_nodes", expr="id > 0", output_fields=["id"], limit=1
    )
    assert isinstance(res, list)


def test_sync_query_error_raises(cfg, monkeypatch):
    mgr = MilvusConnectionManager(cfg)

    class Boom(FakeCollection):
        def load(self):
            pass

        def query(self, *a, **k):
            raise RuntimeError("sync query fail")

    mod = importlib.import_module(
        "..utils.database.milvus_connection_manager", package=__package__
    )
    monkeypatch.setattr(mod, "Collection", Boom, raising=True)

    with pytest.raises(MilvusException):
        mgr._sync_query("dbX_nodes", "x > 0", ["id"], 5)


@pytest.mark.asyncio
async def test_async_load_collection_ok(cfg):
    mgr = MilvusConnectionManager(cfg)
    ok = await mgr.async_load_collection("dbX_nodes")
    assert ok is True
    # async loaded mark present
    assert FakeCollection.registry["dbX_nodes"]["loaded_async"] is True


@pytest.mark.asyncio
async def test_async_load_collection_error_raises(cfg, monkeypatch):
    mgr = MilvusConnectionManager(cfg)

    class BadAsync(FakeAsyncClient):
        async def load_collection(self, *a, **k):
            raise RuntimeError("boom")

    mod = importlib.import_module(
        "..utils.database.milvus_connection_manager", package=__package__
    )
    monkeypatch.setattr(mod, "AsyncMilvusClient", BadAsync, raising=True)
    # Force recreation of async client on this mgr
    mgr._async_client = None

    with pytest.raises(MilvusException):
        await mgr.async_load_collection("dbX_nodes")


@pytest.mark.asyncio
async def test_async_get_collection_stats_ok(cfg):
    mgr = MilvusConnectionManager(cfg)
    FakeCollection.registry["dbX_nodes"] = {"num_entities": 42}
    stats = await mgr.async_get_collection_stats("dbX_nodes")
    assert stats == {"num_entities": 42}


@pytest.mark.asyncio
async def test_async_get_collection_stats_error(cfg, monkeypatch):
    mgr = MilvusConnectionManager(cfg)

    class BadCollection(FakeCollection):
        def __init__(self, name):
            # override to avoid FakeCollection.__init__ assigning to self.num_entities
            self.name = name

        @property
        def num_entities(self):
            raise RuntimeError("stats fail")

    mod = importlib.import_module(
        "..utils.database.milvus_connection_manager", package=__package__
    )
    monkeypatch.setattr(mod, "Collection", BadCollection, raising=True)

    # Directly trigger the property so the exact line is covered
    with pytest.raises(RuntimeError):
        _ = BadCollection("dbX_nodes").num_entities

    # And verify the manager wraps it into MilvusException
    with pytest.raises(MilvusException):
        await mgr.async_get_collection_stats("dbX_nodes")


def test_disconnect_closes_both_clients(cfg):
    mgr = MilvusConnectionManager(cfg)
    # create both clients
    mgr.get_sync_client()
    ac = mgr.get_async_client()
    mgr.ensure_connection()
    ok = mgr.disconnect()
    assert ok is True
    # references cleared
    assert mgr._sync_client is None
    assert mgr._async_client is None


def test_from_config_and_get_instance_are_singleton(cfg):
    a = MilvusConnectionManager.from_config(cfg)
    b = MilvusConnectionManager.get_instance(cfg)
    assert a is b


def test_from_hydra_config_success(monkeypatch):
    # Fake hydra returning desired cfg shape
    class HydraCtx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def initialize(**k):
        return HydraCtx()

    def compose(config_name, overrides):
        return SimpleNamespace(
            utils=SimpleNamespace(
                database=SimpleNamespace(
                    milvus=SimpleNamespace(
                        milvus_db=SimpleNamespace(
                            host="127.0.0.1",
                            port=19530,
                            user="u",
                            password="p",
                            database_name="dbY",
                            alias="aliasY",
                        )
                    )
                )
            )
        )

    mod = importlib.import_module(
        "..utils.database.milvus_connection_manager", package=__package__
    )
    monkeypatch.setattr(
        mod,
        "hydra",
        SimpleNamespace(initialize=initialize, compose=compose),
        raising=True,
    )
    mgr = MilvusConnectionManager.from_hydra_config(
        overrides=["utils/database/milvus=default"]
    )
    assert isinstance(mgr, MilvusConnectionManager)


def test_from_hydra_config_failure_raises(monkeypatch):
    class HydraCtx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def initialize(**k):
        return HydraCtx()

    def compose(*a, **k):
        raise RuntimeError("compose fail")

    mod = importlib.import_module(
        "..utils.database.milvus_connection_manager", package=__package__
    )
    monkeypatch.setattr(
        mod,
        "hydra",
        SimpleNamespace(initialize=initialize, compose=compose),
        raising=True,
    )
    with pytest.raises(MilvusException):
        MilvusConnectionManager.from_hydra_config()


def test_get_async_client_init_exception_returns_none(cfg, monkeypatch):

    mod = importlib.import_module(
        "..utils.database.milvus_connection_manager", package=__package__
    )

    class BadAsyncClient:
        def __init__(self, *a, **k):
            raise RuntimeError("cannot init async client")

    monkeypatch.setattr(mod, "AsyncMilvusClient", BadAsyncClient, raising=True)

    mgr = MilvusConnectionManager(cfg)
    assert mgr.get_async_client() is None  # hits the except → log → return None


def test_ensure_connection_milvus_exception_branch(cfg, monkeypatch):

    mod = importlib.import_module(
        "..utils.database.milvus_connection_manager", package=__package__
    )
    mgr = MilvusConnectionManager(cfg)

    # has_connection → False so it tries to connect
    def has_conn(alias):
        return False

    def connect(*a, **k):
        raise MilvusException("boom")  # specific MilvusException

    monkeypatch.setattr(mod.connections, "has_connection", has_conn, raising=True)
    monkeypatch.setattr(mod.connections, "connect", connect, raising=True)

    with pytest.raises(MilvusException):
        mgr.ensure_connection()  # hits 'except MilvusException as e: raise'


def test_ensure_connection_generic_exception_wrapped(cfg, monkeypatch):

    mod = importlib.import_module(
        "..utils.database.milvus_connection_manager", package=__package__
    )
    mgr = MilvusConnectionManager(cfg)

    def has_conn(alias):
        return False

    def connect(*a, **k):
        raise RuntimeError("generic failure")  # generic exception

    monkeypatch.setattr(mod.connections, "has_connection", has_conn, raising=True)
    monkeypatch.setattr(mod.connections, "connect", connect, raising=True)

    with pytest.raises(MilvusException):
        mgr.ensure_connection()  # hits 'except Exception as e: raise MilvusException(...)'


def test_get_connection_info_error_branch(cfg, monkeypatch):
    mod = importlib.import_module(
        "..utils.database.milvus_connection_manager", package=__package__
    )
    mgr = MilvusConnectionManager(cfg)

    # Force an exception when fetching connection info
    def has_conn(alias):
        return True

    def get_addr(alias):
        raise RuntimeError("addr fail")

    monkeypatch.setattr(mod.connections, "has_connection", has_conn, raising=True)
    monkeypatch.setattr(mod.connections, "get_connection_addr", get_addr, raising=True)

    info = mgr.get_connection_info()
    assert info["connected"] is False
    assert "error" in info


def test_test_connection_failure_returns_false(cfg, monkeypatch):
    mgr = MilvusConnectionManager(cfg)
    # Make ensure_connection blow up so test_connection catches and returns False
    monkeypatch.setattr(
        mgr,
        "ensure_connection",
        lambda: (_ for _ in ()).throw(RuntimeError("no conn")),
        raising=True,
    )
    assert mgr.test_connection() is False


@pytest.mark.asyncio
async def test_disconnect_uses_create_task_when_loop_running(cfg):
    mgr = MilvusConnectionManager(cfg)
    # create async client so disconnect tries to close it
    acli = mgr.get_async_client()
    # ensure a sync connection exists to also exercise that branch
    mgr.ensure_connection()

    # We are in an async test → running loop exists → should call loop.create_task(...)
    ok = mgr.disconnect()
    assert ok is True
    assert mgr._async_client is None
    assert mgr._sync_client is None


def test_disconnect_async_close_exception_sets_false(cfg, monkeypatch):
    mgr = MilvusConnectionManager(cfg)

    class BadAsyncClose:
        async def close(self):
            raise RuntimeError("close fail")

    # Inject a "bad" async client
    mgr._async_client = BadAsyncClose()

    # Force the no-running-loop branch so it uses asyncio.run(...) which will raise from close()

    monkeypatch.setattr(
        _asyncio,
        "get_running_loop",
        lambda: (_ for _ in ()).throw(RuntimeError("no loop")),
        raising=True,
    )

    # Stub asyncio.run to directly call the coro and raise
    def fake_run(coro):
        # drive the coroutine to exception
        loop = _asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(_asyncio, "run", fake_run, raising=True)

    # Also make sure no sync connection path crashes
    ok = mgr.disconnect()
    assert ok is False
    assert mgr._async_client is None  # cleared even on failure


def test_disconnect_outer_exception_returns_false(cfg, monkeypatch):
    mgr = MilvusConnectionManager(cfg)
    # Make connections.has_connection itself raise to jump to outer except

    mod = importlib.import_module(
        "..utils.database.milvus_connection_manager", package=__package__
    )
    monkeypatch.setattr(
        mod.connections,
        "has_connection",
        lambda alias: (_ for _ in ()).throw(RuntimeError("outer boom")),
        raising=True,
    )

    assert mgr.disconnect() is False
