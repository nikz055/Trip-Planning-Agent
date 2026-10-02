"""The HTTP API and the pages it serves, driven end to end: request -> form ->
plan -> lock -> revise -> accept -> share. Runs on both stores."""

from __future__ import annotations

import httpx
import pytest

from trip_agent.api.app import create_app
from trip_agent.tests.conftest import VIENNA

FORM = {"origin": "London", "start_date": "2026-11-07", "end_date": "2026-11-11", "adults": "2"}


@pytest.fixture
async def client(orchestrator):
    app = create_app(orchestrator)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            yield http


async def test_request_to_accepted_plan(client):
    health = (await client.get("/api/health")).json()
    assert "STAND-IN" in health["planner"]  # the UI shows this label

    waiting = (await client.post("/api/trips", json={"text": "Plan a trip to Austria"})).json()
    trip = waiting["trip_id"]
    assert waiting["state"] == "WAITING_FOR_DETAILS" and waiting["form"]["fields"]

    invalid = (await client.post(f"/api/trips/{trip}/form", json={"answers": {**FORM, "start_date": "2026-09-01"}})).json()
    assert invalid["state"] == "WAITING_FOR_DETAILS"
    assert any(f["error"] for f in invalid["form"]["fields"])

    plan = (await client.post(f"/api/trips/{trip}/form", json={"answers": FORM})).json()
    assert plan["state"] == "PRESENTED" and plan["trip_id"] == trip
    items = [i for d in plan["plan"]["days"] for i in d["items"]]
    assert {i["cost_status"] for i in items} == {"confirmed", "estimated"}  # the badges
    assert all(i["fetched_at"] and i["source"] == "mock" for i in items if i["ref"])  # fetch times
    assert plan["plan"]["budget"]["confirmed_total"] > 0 and plan["plan"]["budget"]["estimated_total"] > 0

    locked = (await client.post(f"/api/trips/{trip}/lock", json={"item_id": "hotel", "locked": True})).json()
    assert locked["locked_item_ids"] == ["hotel"]

    revised = (await client.post(f"/api/trips/{trip}/revise", json={"instruction": "Less walking on day 2"})).json()
    assert revised["state"] == "PRESENTED" and revised["version"] == 2
    assert revised["plan"]["days"][0] == locked["plan"]["days"][0]

    assert (await client.get(f"/api/trips/{trip}")).json()["version"] == 2
    accepted = (await client.post(f"/api/trips/{trip}/accept")).json()
    assert accepted["state"] == "ACCEPTED" and accepted["terminal"] is True


async def test_share_link_is_read_only(client):
    plan = (await client.post("/api/trips", json={"text": VIENNA})).json()
    shared = (await client.get(f"/api/share/{plan['trip_id']}")).json()
    assert shared["read_only"] is True and shared["plan"] == plan["plan"]
    assert set(shared) == {"trip_id", "state", "planner", "version", "request", "plan", "read_only"}
    assert "raw_text" not in shared["request"] and "total_budget" not in shared["request"]
    links = [i["link"] for d in shared["plan"]["days"] for i in d["items"] if i["link"]]
    assert len(links) == 3 and all(link.startswith("https://") for link in links)  # links to providers
    assert (await client.post(f"/api/share/{plan['trip_id']}")).status_code == 405  # nothing to act on

    waiting = (await client.post("/api/trips", json={"text": "Plan a trip to Austria"})).json()
    assert (await client.get(f"/api/share/{waiting['trip_id']}")).status_code == 404  # no plan yet


async def test_idempotency_key_header(client):
    headers = {"Idempotency-Key": "browser-key-0001"}
    first = (await client.post("/api/trips", json={"text": VIENNA}, headers=headers)).json()
    second = (await client.post("/api/trips", json={"text": VIENNA}, headers=headers)).json()
    assert first == second


async def test_errors_are_json_with_the_right_status(client):
    missing = await client.get("/api/trips/00000000-0000-0000-0000-000000000000")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
    plan = (await client.post("/api/trips", json={"text": VIENNA})).json()
    wrong_state = await client.post(f"/api/trips/{plan['trip_id']}/form", json={"answers": FORM})
    assert wrong_state.status_code == 409 and wrong_state.json()["error"]["code"] == "invalid_state"
    assert (await client.post("/api/trips", json={"text": ""})).status_code == 422
    assert (await client.post("/api/trips", json={"text": "x"}, headers={"Idempotency-Key": "short"})).status_code == 400


async def test_pages_and_static_files_are_served(client):
    for path in ("/", "/share/00000000-0000-0000-0000-000000000000"):
        page = await client.get(path)
        assert page.status_code == 200 and '<main id="app"' in page.text
    script = await client.get("/static/app.js")
    assert script.status_code == 200 and "Idempotency-Key" in script.text
    assert "supabase" not in script.text.lower()  # the browser never talks to the database
    assert (await client.get("/static/styles.css")).status_code == 200
