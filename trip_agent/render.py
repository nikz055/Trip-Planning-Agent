"""Plain-text rendering of a trip view for the CLI."""

from __future__ import annotations

from typing import Any

View = dict[str, Any]


def _when(item: dict[str, Any]) -> str:
    if not item["start"]:
        return " " * 11
    return f"{item['start'][11:16]}-{item['end'][11:16] if item['end'] else '     '}"


def _provenance(item: dict[str, Any]) -> str:
    """BR-10: every priced item shows its source and when it was fetched."""
    if not item["source"]:
        return "estimate, no source"
    fetched = item["fetched_at"][:16].replace("T", " ")
    return f"{item['source']}, fetched {fetched} UTC" + (", STALE" if item["stale"] else "")


def _plan_lines(plan: dict[str, Any]) -> list[str]:
    currency = plan["budget_check"]["currency"]
    lines: list[str] = []
    for day in plan["days"]:
        lines.append(f"Day {day['day']}  {day['date']}")
        if not day["items"]:
            lines.append("  (nothing scheduled)")
        for item in day["items"]:
            lock = " [locked]" if item["locked"] else ""
            lines.append(
                f"  {_when(item)}  {item['type']:<15} {item['title'][:44]:<44} "
                f"{currency} {item['cost']:>8.2f}  {item['cost_status']:<9} ({_provenance(item)}){lock}  #{item['id']}"
            )
            if item["link"]:
                lines.append(f"  {'':11}  {'':15} {item['link']}")
        lines.append("")
    budget, check = plan["budget"], plan["budget_check"]
    lines += [
        "Budget",
        f"  flights          {currency} {budget['flights']:>9.2f}",
        f"  lodging          {currency} {budget['lodging']:>9.2f}",
        f"  local transport  {currency} {budget['local_transport']:>9.2f}",
        f"  activities       {currency} {budget['activities']:>9.2f}",
        f"  meals allowance  {currency} {budget['meals_allowance']:>9.2f}",
        f"  other allowance  {currency} {budget['other_allowance']:>9.2f}",
        f"  total            {currency} {budget['total']:>9.2f}",
        f"    confirmed (from tool results)  {currency} {budget['confirmed_total']:>9.2f}",
        f"    estimated                      {currency} {budget['estimated_total']:>9.2f}",
        f"  {check['label']}",
    ]
    if check["note"]:
        lines.append(f"  {check['note']}")
    for title, entries in (("Assumptions", plan["assumptions"]), ("Gaps", plan["gaps"])):
        if entries:
            lines += ["", title] + [f"  - {entry}" for entry in entries]
    if plan["violations"]:
        lines += ["", "Violations"] + [f"  - [{v['rule']}] {v['target']}: {v['message']}" for v in plan["violations"]]
    return lines


def render_view(view: View) -> str:
    request = view["request"]
    lines = [
        f"Planner: {view['planner']}",
        f"Trip {view['trip_id']}  state {view['state']}"
        + (f"  plan version {view['version']}" if view["version"] else ""),
    ]
    if request.get("destination_city"):
        lines.append(
            f"{request.get('origin') or '?'} to {request['destination_city']}, "
            f"{request.get('start_date') or '?'} to {request.get('end_date') or '?'}, "
            f"{request.get('adults') or '?'} adult(s), pace {request['pace']}"
        )
    for notice in view["notices"]:
        lines.append(f"NOTE: {notice}")
    if view["limits"]["stopped_by"]:
        lines.append(f"Planning stopped early: {view['limits']['stopped_by']}.")
    if view["give_up"]:
        lines.append(f"No plan: {view['give_up']['reason']}")
        lines += [f"  - {s}" for s in view["give_up"]["suggestions"]]

    form = view["form"]
    if view["state"] == "WAITING_FOR_DETAILS" and form:
        lines += ["", form["reason"], "Details needed (answer with --set FIELD=VALUE):"]
        for field in form["fields"]:
            current = f" [{field['value']}]" if field["value"] not in (None, "", []) else ""
            flag = "" if field["required"] else " (optional)"
            options = f"  options: {', '.join(field['options'])}" if field["options"] else ""
            error = f"  ERROR: {field['error']}" if field["error"] else ""
            lines.append(f"  {field['name']}: {field['label']}{flag}{current}{options}{error}")
        if form["suggestions"]:
            lines.append("Ideas for this destination: " + ", ".join(form["suggestions"]))

    if view["plan"]:
        if view["state"] == "PARTIAL":
            lines += ["", "PARTIAL PLAN: see Gaps and Violations below."]
        lines += [""] + _plan_lines(view["plan"])
    limits = view["limits"]
    lines.append(
        f"\n{limits['iterations']} planner iteration(s) in the last run, about {limits['tokens_used']} tokens, "
        f"estimated LLM cost USD {limits['estimated_cost_usd']:.4f}"
    )
    return "\n".join(lines)
