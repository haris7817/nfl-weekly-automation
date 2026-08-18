"""Sheet synchronisation logic, exercised against a fake Sheets API.

No live Google access is required (implementation guide 30.1). The fake
records every call so the idempotency rules can be asserted directly.
"""

from __future__ import annotations

import pandas as pd

from src.fair_line import FAIR_LINE_COLUMNS, STATUS_PROJECTED
from src.google_sheets import (
    HISTORY_KEY,
    PREDICTION_HEADERS,
    TAB_HISTORY,
    ensure_tabs,
    replace_tab,
    upsert_prediction_history,
)


class FakeRequest:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class FakeValues:
    def __init__(self, store, calls):
        self._store = store
        self._calls = calls

    @staticmethod
    def _tab(range_):
        return range_.split("!")[0].strip("'")

    def get(self, spreadsheetId, range):
        return FakeRequest({"values": self._store.get(self._tab(range), [])})

    def clear(self, spreadsheetId, range):
        self._calls.append(("clear", self._tab(range)))
        self._store[self._tab(range)] = []
        return FakeRequest({})

    def update(self, spreadsheetId, range, valueInputOption, body):
        self._calls.append(("update", self._tab(range)))
        self._store[self._tab(range)] = [list(r) for r in body["values"]]
        return FakeRequest({})

    def append(self, spreadsheetId, range, valueInputOption, insertDataOption, body):
        self._calls.append(("append", self._tab(range)))
        self._store.setdefault(self._tab(range), []).extend(
            [list(r) for r in body["values"]]
        )
        return FakeRequest({})


class FakeSpreadsheets:
    def __init__(self, store, calls):
        self._store = store
        self._calls = calls

    def get(self, spreadsheetId):
        return FakeRequest(
            {"sheets": [{"properties": {"title": t}} for t in self._store]}
        )

    def values(self):
        return FakeValues(self._store, self._calls)

    def batchUpdate(self, spreadsheetId, body):
        for request in body["requests"]:
            title = request["addSheet"]["properties"]["title"]
            self._store.setdefault(title, [])
            self._calls.append(("addSheet", title))
        return FakeRequest({})


class FakeSheetsService:
    def __init__(self, tabs=None):
        self.store = {tab: [] for tab in (tabs or [])}
        self.calls = []

    def spreadsheets(self):
        return FakeSpreadsheets(self.store, self.calls)


def predictions(rows, season=2025, week=6):
    records = []
    for away, home, spread in rows:
        record = {column: "" for column in FAIR_LINE_COLUMNS}
        record.update(
            {
                "run_timestamp_utc": "2026-01-01T00:00:00Z",
                "season": season,
                "week": week,
                "away_team": away,
                "home_team": home,
                "fair_spread_home": spread,
                "status": STATUS_PROJECTED,
            }
        )
        records.append(record)
    return pd.DataFrame(records, columns=list(FAIR_LINE_COLUMNS))


class TestEnsureTabs:
    def test_creates_only_missing_tabs(self):
        service = FakeSheetsService(tabs=["Latest Predictions"])

        ensure_tabs(service, "sheet-id", ["Latest Predictions", "Model Info"])

        added = [name for kind, name in service.calls if kind == "addSheet"]
        assert added == ["Model Info"]

    def test_no_calls_when_everything_exists(self):
        service = FakeSheetsService(tabs=["A", "B"])

        ensure_tabs(service, "sheet-id", ["A", "B"])

        assert service.calls == []


class TestReplaceTab:
    def test_clears_then_writes(self):
        service = FakeSheetsService(tabs=["EPA Splits"])

        replace_tab(service, "sheet-id", "EPA Splits", ["A", "B"], [[1, 2], [3, 4]])

        assert [kind for kind, _ in service.calls] == ["clear", "update"]
        assert service.store["EPA Splits"] == [["A", "B"], [1, 2], [3, 4]]

    def test_second_replace_does_not_accumulate(self):
        service = FakeSheetsService(tabs=["EPA Splits"])

        replace_tab(service, "sheet-id", "EPA Splits", ["A"], [[1], [2]])
        replace_tab(service, "sheet-id", "EPA Splits", ["A"], [[9]])

        assert service.store["EPA Splits"] == [["A"], [9]]


class TestUpsertPredictionHistory:
    def test_first_write_stores_every_row(self):
        service = FakeSheetsService(tabs=[TAB_HISTORY])

        upsert_prediction_history(
            service, "sheet-id", predictions([("BBB", "AAA", -3.0)])
        )

        assert len(service.store[TAB_HISTORY]) == 2  # header + 1 row

    def test_rerunning_the_same_week_does_not_duplicate(self):
        service = FakeSheetsService(tabs=[TAB_HISTORY])
        df = predictions([("BBB", "AAA", -3.0), ("DDD", "CCC", 1.5)])

        upsert_prediction_history(service, "sheet-id", df)
        upsert_prediction_history(service, "sheet-id", df)

        assert len(service.store[TAB_HISTORY]) == 3  # header + 2 rows

    def test_rerun_refreshes_the_value(self):
        service = FakeSheetsService(tabs=[TAB_HISTORY])

        upsert_prediction_history(
            service, "sheet-id", predictions([("BBB", "AAA", -3.0)])
        )
        result = upsert_prediction_history(
            service, "sheet-id", predictions([("BBB", "AAA", -7.5)])
        )

        spread_column = FAIR_LINE_COLUMNS.index("fair_spread_home")
        assert service.store[TAB_HISTORY][1][spread_column] == -7.5
        assert result["replaced"] == 1

    def test_a_new_week_is_appended_not_replaced(self):
        service = FakeSheetsService(tabs=[TAB_HISTORY])

        upsert_prediction_history(
            service, "sheet-id", predictions([("BBB", "AAA", -3.0)], week=6)
        )
        upsert_prediction_history(
            service, "sheet-id", predictions([("DDD", "CCC", 2.0)], week=7)
        )

        assert len(service.store[TAB_HISTORY]) == 3  # header + 2 weeks

    def test_history_is_sorted_by_season_then_week(self):
        service = FakeSheetsService(tabs=[TAB_HISTORY])

        upsert_prediction_history(
            service, "sheet-id", predictions([("BBB", "AAA", -3.0)], week=9)
        )
        upsert_prediction_history(
            service, "sheet-id", predictions([("DDD", "CCC", 2.0)], week=4)
        )

        body = service.store[TAB_HISTORY][1:]
        weeks = [int(row[FAIR_LINE_COLUMNS.index("week")]) for row in body]
        assert weeks == sorted(weeks)

    def test_existing_rows_from_other_weeks_survive(self):
        service = FakeSheetsService(tabs=[TAB_HISTORY])
        service.store[TAB_HISTORY] = [
            PREDICTION_HEADERS,
            ["2025-01-01T00:00:00Z", 2025, 1, "", "ZZZ", "YYY", "", "", "",
             "", -1.0, 5, 1, "", STATUS_PROJECTED, ""],
        ]

        upsert_prediction_history(
            service, "sheet-id", predictions([("BBB", "AAA", -3.0)], week=6)
        )

        body = service.store[TAB_HISTORY][1:]
        assert any(row[4] == "ZZZ" for row in body)
        assert len(body) == 2

    def test_history_key_columns_exist_in_the_schema(self):
        for column in HISTORY_KEY:
            assert column in FAIR_LINE_COLUMNS
