// Minimal UI. Talks only to the backend API. All text from the server is put in
// the page with textContent, never as HTML.
(function () {
  "use strict";
  const app = document.getElementById("app");
  const shareMatch = location.pathname.match(/^\/share\/([0-9a-f-]{36})$/);
  const readOnly = Boolean(shareMatch);
  let view = null;
  let busy = false;

  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [name, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (name === "class") el.className = value;
      else if (name.startsWith("on")) el.addEventListener(name.slice(2), value);
      else if (name === "value") el.value = value;
      else el.setAttribute(name, value === true ? "" : value);
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      el.append(child.nodeType ? child : document.createTextNode(String(child)));
    }
    return el;
  }

  async function api(method, path, body) {
    const headers = { "Content-Type": "application/json" };
    if (method !== "GET") headers["Idempotency-Key"] = crypto.randomUUID();
    const response = await fetch(path, { method, headers, body: body ? JSON.stringify(body) : undefined });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = data.error ? data.error.message : (data.detail ? JSON.stringify(data.detail) : response.statusText);
      throw new Error(detail);
    }
    return data;
  }

  async function act(label, call) {
    if (busy) return;
    busy = true;
    render(label);
    try {
      view = await call();
      if (!readOnly && view.trip_id) history.replaceState(null, "", "/?trip=" + view.trip_id);
      render();
    } catch (error) {
      busy = false;
      render(null, error.message);
      return;
    }
    busy = false;
    render();
  }

  const money = (plan, amount) => plan.budget_check.currency + " " + Number(amount).toFixed(2);
  const clock = (iso) => (iso ? iso.slice(11, 16) : "");
  const stamp = (iso) => (iso ? iso.slice(0, 16).replace("T", " ") + " UTC" : "");

  // ---- sections -----------------------------------------------------------

  const EXAMPLES = [
    ["Vienna, art and music", "5 days in Vienna from London, 7-11 Nov 2026, 2 adults, budget 2500 EUR, we like art and music"],
    ["Jaipur long weekend", "4 days in Jaipur from Bengaluru, 12-15 Nov 2026, 2 adults, budget 150000 rupees"],
    ["Goa with a child", "Goa from Chennai, 20-24 Nov 2026, 2 adults and 1 child, relaxed"],
    ["Just an idea", "Plan a trip to Austria"],
  ];
  const DESTINATIONS = [
    ["Vienna", "Austria", "Palaces, coffee houses and concert halls.", "5 days in Vienna from Delhi, 7-11 Nov 2026, 2 adults"],
    ["Lisbon", "Portugal", "Hills, trams, tiles and the Atlantic light.", "5 days in Lisbon from Mumbai, 3-7 Dec 2026, 2 adults"],
    ["Jaipur", "India", "Forts, bazaars and the pink old city.", "4 days in Jaipur from Bengaluru, 12-15 Nov 2026, 2 adults"],
    ["Goa", "India", "Beaches, old churches and night markets.", "Goa from Chennai, 20-24 Nov 2026, 2 adults"],
  ];

  function landing() {
    const text = h("textarea", { id: "request", rows: "3", placeholder: "For example: 5 days in Vienna from London, 7-11 Nov 2026, 2 adults, budget 2500 EUR, we like art and music" });
    const submit = () => text.value.trim() && act("Planning your trip…", () => api("POST", "/api/trips", { text: text.value }));
    const fill = (value) => { text.value = value; text.focus(); text.scrollIntoView({ block: "center", behavior: "smooth" }); };
    const step = (n, title, body) => h("div", { class: "step" }, h("div", { class: "step-n", "aria-hidden": "true" }, n), h("h3", {}, title), h("p", {}, body));
    const feature = (title, body) => h("div", { class: "feature" }, h("h3", {}, title), h("p", {}, body));

    return [
      h("section", { class: "hero" },
        h("div", { class: "hero-copy" },
          h("p", { class: "eyebrow" }, "City trips, 3 to 7 days"),
          h("h1", {}, "A trip plan you can actually follow."),
          h("p", { class: "lede" }, "Describe your trip in a sentence. Get a day-by-day plan with flights, a hotel and realistic time between stops, with every cost marked as confirmed or estimated.")),
        h("div", { class: "hero-form" },
          h("label", { for: "request" }, "Where do you want to go?"),
          text,
          h("div", { class: "row between" },
            h("span", { class: "muted small-text" }, "Dates, travellers and budget help. We ask for anything missing."),
            h("button", { class: "primary big", disabled: busy, onclick: submit }, "Plan my trip")),
          h("div", { class: "chips" }, h("span", { class: "muted small-text" }, "Try:"),
            EXAMPLES.map(([label, value]) => h("button", { class: "chip", type: "button", onclick: () => fill(value) }, label))))),

      h("section", { class: "facts", "aria-label": "At a glance" },
        h("div", {}, h("strong", {}, "9"), h("span", {}, "checks on every plan")),
        h("div", {}, h("strong", {}, "4"), h("span", {}, "sample destinations")),
        h("div", {}, h("strong", {}, "120"), h("span", {}, "departure cities")),
        h("div", {}, h("strong", {}, "1"), h("span", {}, "short form, at most"))),

      h("section", { class: "band" },
        h("h2", {}, "How it works"),
        h("div", { class: "steps" },
          step("1", "Tell us the trip", "Write it the way you would say it. If something is missing, one short form asks for all of it at once."),
          step("2", "We build and check it", "Opening hours, closures, travel time, buffers, hotel nights and the full budget are verified before you see the plan."),
          step("3", "Change what you like", "Lock what you want to keep and ask for a change in plain words. Only the affected days are replanned."))),

      h("section", { class: "band" },
        h("h2", {}, "What you get"),
        h("div", { class: "features" },
          feature("Costs you can trust", "Every price shows where it came from and when it was fetched. Estimates are labelled as estimates and totalled separately."),
          feature("Nothing made up", "Each flight, hotel and attraction traces back to a search result. Anything that does not is rejected before you see it."),
          feature("Honest about limits", "If a search fails or the budget does not stretch, the plan says so plainly instead of hiding it."),
          feature("Easy to share", "Every plan has a read-only link with provider links for booking. This app does not book or take payment."))),

      h("section", { class: "band" },
        h("h2", {}, "Destinations in this demo"),
        h("div", { class: "dest-grid" },
          DESTINATIONS.map(([city, country, blurb, example]) =>
            h("button", { class: "dest", type: "button", onclick: () => fill(example) },
              h("span", { class: "dest-country" }, country), h("span", { class: "dest-city" }, city), h("span", { class: "dest-blurb" }, blurb), h("span", { class: "dest-cta" }, "Plan this trip →"))))),

      h("footer", { class: "foot" },
        h("p", {}, "Demo running on sample data: place names are real, but hours, prices, hotels and flights are illustrative."),
        h("p", {}, "Planning only. No booking, no payment, no account.")),
    ];
  }

  function formSection(form) {
    const inputs = {};
    const fields = form.fields.map((field) => {
      const id = "f-" + field.name;
      let input;
      if (field.kind === "select" || field.options.length) {
        const options = field.options.includes(String(field.value ?? "")) || field.value == null ? field.options : [String(field.value), ...field.options];
        input = h("select", { id }, field.required ? null : h("option", { value: "" }, "—"),
          options.map((o) => h("option", { value: o, selected: String(field.value ?? "") === o }, o)));
      } else if (field.kind === "bool") {
        input = h("select", { id }, h("option", { value: "false", selected: !field.value }, "No"), h("option", { value: "true", selected: Boolean(field.value) }, "Yes"));
      } else {
        const type = { date: "date", int: "number", number: "number" }[field.kind] || "text";
        input = h("input", { id, type, value: field.value ?? "", step: field.kind === "number" ? "any" : null, min: type === "number" ? "0" : null });
      }
      inputs[field.name] = input;
      return h("div", { class: "field" + (field.error ? " invalid" : "") },
        h("label", { for: id }, field.label + (field.required ? "" : " (optional)")),
        input,
        field.error ? h("div", { class: "error", role: "alert" }, field.error) : null);
    });
    const submit = () => {
      const answers = {};
      for (const [name, input] of Object.entries(inputs)) answers[name] = input.value;
      act("Planning your trip…", () => api("POST", `/api/trips/${view.trip_id}/form`, { answers }));
    };
    return h("section", { class: "card" },
      h("h2", {}, "A few details"),
      h("p", {}, form.reason),
      form.suggestions.length ? h("p", { class: "muted" }, "Ideas for this destination: " + form.suggestions.join(", ")) : null,
      h("div", { class: "grid" }, fields),
      h("div", { class: "row" },
        h("button", { class: "primary", disabled: busy, onclick: submit }, "Continue"),
        h("button", { disabled: busy, onclick: () => act("Cancelling…", () => api("POST", `/api/trips/${view.trip_id}/cancel`)) }, "Cancel trip")));
  }

  function itemRow(plan, item, canLock) {
    const priced = Boolean(item.source);
    const meta = [];
    if (priced) meta.push(`Source: ${item.source}, fetched ${stamp(item.fetched_at)}`);
    else meta.push("Estimate, not from a provider");
    if (item.notes) meta.push(item.notes);
    const lockButton = canLock && priced
      ? h("button", { class: "small", disabled: busy, "aria-pressed": String(item.locked),
          onclick: () => act(null, () => api("POST", `/api/trips/${view.trip_id}/lock`, { item_id: item.id, locked: !item.locked })) },
          item.locked ? "Unlock" : "Lock")
      : null;
    return h("div", { class: "item" + (item.locked ? " locked" : "") },
      h("div", { class: "time" }, item.start ? `${clock(item.start)}–${clock(item.end)}` : ""),
      h("div", {},
        h("div", { class: "kind" }, item.type.replace("_", " ") + (item.locked ? " · locked" : "")),
        h("div", { class: "title" }, item.title || item.id),
        h("div", { class: "meta" }, meta.join(" · ")),
        item.link ? h("div", { class: "meta" }, h("a", { href: item.link, target: "_blank", rel: "noopener noreferrer" }, "View at provider")) : null),
      h("div", { class: "cost" },
        h("div", {}, money(plan, item.cost)),
        h("span", { class: "badge " + item.cost_status }, item.cost_status),
        item.stale ? h("span", { class: "badge stale" }, "stale") : null,
        lockButton ? h("div", {}, lockButton) : null));
  }

  function planSection(plan, canLock) {
    const out = [];
    if (plan.gaps.length) out.push(h("div", { class: "banner bad" }, h("strong", {}, "This plan is partial."), h("ul", {}, plan.gaps.map((g) => h("li", {}, g)))));
    if (plan.violations.length) {
      out.push(h("div", { class: "banner bad" }, h("strong", {}, "Problems found by the checker"),
        h("ul", {}, plan.violations.map((v) => h("li", {}, `[${v.rule}] ${v.target}: ${v.message}`)))));
    }
    for (const day of plan.days) {
      const date = new Date(day.date + "T00:00:00").toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long" });
      out.push(h("section", { class: "card" },
        h("div", { class: "day-head" }, h("h2", {}, `Day ${day.day}`), h("span", { class: "muted" }, date)),
        day.items.length ? day.items.map((item) => itemRow(plan, item, canLock)) : h("p", { class: "muted" }, "Nothing scheduled.")));
    }
    const b = plan.budget, check = plan.budget_check;
    const rows = [["Flights", b.flights], ["Lodging", b.lodging], ["Local transport", b.local_transport],
      ["Activities", b.activities], ["Meals (allowance)", b.meals_allowance], ["Other (allowance)", b.other_allowance]];
    out.push(h("section", { class: "card" },
      h("h2", {}, "Cost"),
      h("div", { class: "banner " + (check.within_budget === false ? "bad" : check.within_budget ? "ok" : "note") }, check.label),
      check.note ? h("p", { class: "muted" }, check.note) : null,
      h("table", { class: "budget" },
        rows.map(([name, value]) => h("tr", {}, h("td", {}, name), h("td", {}, money(plan, value)))),
        h("tr", { class: "total" }, h("td", {}, "Total"), h("td", {}, money(plan, b.total))),
        h("tr", {}, h("td", {}, h("span", { class: "badge confirmed" }, "confirmed"), " from provider results"), h("td", {}, money(plan, b.confirmed_total))),
        h("tr", {}, h("td", {}, h("span", { class: "badge estimated" }, "estimated"), " allowances and travel estimates"), h("td", {}, money(plan, b.estimated_total)))),
      plan.assumptions.length ? h("div", {}, h("h3", { style: "margin-top:14px" }, "Assumptions"), h("ul", {}, plan.assumptions.map((a) => h("li", {}, a)))) : null));
    return out;
  }

  function actions() {
    const state = view.state;
    const out = [];
    if (state === "PRESENTED" || state === "PARTIAL") {
      const box = h("textarea", { id: "change", placeholder: "For example: less walking on day 2" });
      out.push(h("section", { class: "card" },
        h("h2", {}, "Change something"),
        h("p", { class: "muted" }, "Only the days your change affects are replanned. Locked items stay as they are."),
        h("div", { class: "field" }, h("label", { for: "change" }, "What would you like to change?"), box),
        h("div", { class: "row" },
          h("button", { disabled: busy, onclick: () => box.value.trim() && act("Replanning…", () => api("POST", `/api/trips/${view.trip_id}/revise`, { instruction: box.value })) }, "Revise plan"),
          state === "PRESENTED" ? h("button", { class: "primary", disabled: busy, onclick: () => act("Accepting…", () => api("POST", `/api/trips/${view.trip_id}/accept`)) }, "Accept plan") : null,
          h("button", { disabled: busy, onclick: () => act("Cancelling…", () => api("POST", `/api/trips/${view.trip_id}/cancel`)) }, "Cancel trip"))));
    }
    if (view.plan) {
      const url = location.origin + view.share_path;
      out.push(h("section", { class: "card share" }, h("h2", {}, "Share"), h("p", {}, "Read-only link to this plan: ", h("a", { href: url }, url)),
        view.versions.length > 1 ? h("p", { class: "muted" }, "Versions: " + view.versions.map((v) => `${v.version} (${v.reason})`).join(", ") + `. Showing version ${view.version}.`) : null));
    }
    return out;
  }

  function render(working, error) {
    const planner = document.getElementById("planner");
    if (view) {
      planner.textContent = "Planner: " + view.planner;
      planner.className = "planner" + (/stand-in/i.test(view.planner) ? " standin" : "");
    }
    const out = [];
    if (working) out.push(h("div", { class: "banner note", role: "status" }, working));
    if (error) out.push(h("div", { class: "banner bad", role: "alert" }, error));

    document.body.classList.toggle("landing", !view && !readOnly);
    if (!view) {
      if (!readOnly) out.push(...landing());
    } else {
      const r = view.request;
      const summary = r.destination_city
        ? `${r.origin || "?"} to ${r.destination_city}, ${r.start_date || "?"} to ${r.end_date || "?"}, ${r.adults || "?"} adult(s)`
        : "New trip";
      out.push(h("section", { class: "card" },
        h("div", { class: "row" }, h("h1", { style: "margin:0" }, summary), h("span", { class: "state" }, view.state.replaceAll("_", " "))),
        readOnly ? h("p", { class: "muted" }, "Shared plan, read-only. Prices were fetched at the times shown and may have changed.") : null,
        !readOnly && view.limits && view.limits.stopped_by ? h("p", { class: "muted" }, "Planning stopped early: " + view.limits.stopped_by + ".") : null));
      for (const notice of view.notices || []) out.push(h("div", { class: "banner note" }, notice));
      if (view.give_up) {
        out.push(h("div", { class: "banner bad" }, h("strong", {}, "No plan: "), view.give_up.reason,
          view.give_up.suggestions.length ? h("ul", {}, view.give_up.suggestions.map((s) => h("li", {}, s))) : null));
      }
      if (view.state === "ACCEPTED") out.push(h("div", { class: "banner ok" }, "Plan accepted. Use the provider links to book; this app does not book or take payment."));
      if (view.state === "CANCELLED") out.push(h("div", { class: "banner note" }, "This trip was cancelled."));
      if (!readOnly && view.state === "WAITING_FOR_DETAILS" && view.form) out.push(formSection(view.form));
      if (view.plan) out.push(...planSection(view.plan, !readOnly && (view.state === "PRESENTED" || view.state === "PARTIAL")));
      if (!readOnly) {
        out.push(...actions());
        out.push(h("p", {}, h("a", { href: "/" }, "Start a new trip")));
      }
    }
    app.replaceChildren(...out);
  }

  // ---- start ----------------------------------------------------------------

  const tripId = new URLSearchParams(location.search).get("trip");
  if (readOnly) act("Loading the shared plan…", () => api("GET", "/api/share/" + shareMatch[1]));
  else if (tripId) act("Loading your trip…", () => api("GET", "/api/trips/" + tripId));
  else {
    fetch("/api/health").then((r) => r.json()).then((health) => {
      const planner = document.getElementById("planner");
      planner.textContent = "Planner: " + health.planner;
      planner.className = "planner" + (/stand-in/i.test(health.planner) ? " standin" : "");
    }).catch(() => {});
    render();
  }
})();
