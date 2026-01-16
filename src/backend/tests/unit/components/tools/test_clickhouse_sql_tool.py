import pytest

from lfx.components.tools.clickhouse_sql import ClickHouseSQLToolComponent
from tests.base import DID_NOT_EXIST, ComponentTestBaseWithoutClient


class TestClickHouseSQLToolComponent(ComponentTestBaseWithoutClient):
    @pytest.fixture
    def component_class(self):
        return ClickHouseSQLToolComponent

    @pytest.fixture
    def default_kwargs(self):
        # Intentionally omit credentials so the component returns an error Data
        # rather than attempting any real network call during the base test.
        return {"query": "SELECT 1", "select_only": True}

    @pytest.fixture
    def file_names_mapping(self):
        # Component not yet released, mark all versions as non-existent
        return [
            {"version": "1.0.17", "module": "tools", "file_name": DID_NOT_EXIST},
            {"version": "1.0.18", "module": "tools", "file_name": DID_NOT_EXIST},
            {"version": "1.0.19", "module": "tools", "file_name": DID_NOT_EXIST},
            {"version": "1.1.0", "module": "tools", "file_name": DID_NOT_EXIST},
            {"version": "1.1.1", "module": "tools", "file_name": DID_NOT_EXIST},
        ]


def test_missing_configuration_returns_error_data():
    component = ClickHouseSQLToolComponent(query="SELECT 1", select_only=True)
    result = component.run_model()[0]
    assert result.data["success"] is False
    assert "Missing required configuration" in result.data["error"]


def test_rejects_unsafe_sql_when_select_only_enabled():
    component = ClickHouseSQLToolComponent(query="INSERT INTO t VALUES (1)", select_only=True)
    result = component.run_model()[0]
    assert result.data["success"] is False
    assert "Only SELECT queries are allowed" in result.data["error"]


def test_success_parses_csv_with_names(monkeypatch):
    monkeypatch.setenv("PROD_CH_HOST", "example-clickhouse.local")
    monkeypatch.setenv("PROD_CH_USER", "user1")
    monkeypatch.setenv("PROD_CH_PASSWORD", "pass1")
    monkeypatch.setenv("DATABASE", "pcm")

    called = {}

    class FakeResponse:
        def __init__(self, text: str):
            self.text = text
            self.status_code = 200

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, timeout, verify):
            called["timeout"] = timeout
            called["verify"] = verify

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None, headers=None, auth=None):
            called["url"] = url
            called["params"] = params
            called["headers"] = headers
            called["auth"] = auth
            return FakeResponse("a,b\n1,2\n")

    import lfx.components.tools.clickhouse_sql as mod

    monkeypatch.setattr(mod.httpx, "Client", FakeClient)

    component = ClickHouseSQLToolComponent(query="SELECT 1;", environment="prod", select_only=True, max_rows_to_parse=1000)
    result = component.run_model()[0]

    assert result.data["success"] is True
    assert result.data["columns"] == ["a", "b"]
    assert result.data["rows"] == [{"a": "1", "b": "2"}]
    assert result.data["row_count"] == 1

    assert called["url"] == "https://example-clickhouse.local:8443"
    assert called["params"]["query"] == "SELECT 1"
    assert called["headers"]["X-ClickHouse-Database"] == "pcm"
    assert called["headers"]["X-ClickHouse-Format"] == "CSVWithNames"
    assert called["auth"] == ("user1", "pass1")

