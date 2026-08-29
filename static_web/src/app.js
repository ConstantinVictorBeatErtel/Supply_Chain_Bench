import { BeerEpisode, replayActions, scenarioFor } from "./sim/index.js";
import {
  configureTelemetry, createSessionUuid, sendTelemetry,
} from "./telemetry.js";

const TIER = 5;
const VARIANT = "headline";
/** Public play capacity: research Tier 5 stays at 22 in scenario.js; live play uses 400. */
const PUBLIC_FACTORY_CAPACITY = 400;
const ROLES = ["retailer_a", "retailer_b", "wholesaler", "distributor", "factory"];
const ROLE_LABEL = {
  retailer_a: "Retailer A", retailer_b: "Retailer B", wholesaler: "Wholesaler",
  distributor: "Distributor", factory: "Factory",
};
const ROLE_NOTE = {
  retailer_a: "Faces end customers directly. Shortest signal, sharpest noise.",
  retailer_b: "Competes with A for the same wholesaler inventory.",
  wholesaler: "Rations one pool between two retailers. The hard seat.",
  distributor: "Buffers the factory against wholesaler swings.",
  factory: `Produces to order, capped at ${PUBLIC_FACTORY_CAPACITY} units a week.`,
};

function publicScenario(spec) {
  return { ...spec, capacity: PUBLIC_FACTORY_CAPACITY };
}

/** Weeks per chapter in the debrief thought tracker. 36 weeks / 6 = six chapters. */
const THOUGHT_GROUP_SIZE = 6;

let catalog = null;
let replayCatalog = null;
let replayTimer = null;
let active = null;
let selectedRole = "wholesaler";
let logSent = false;
let fastForwardTimer = null;
let shippingPulse = false;
let thoughtView = { group: 0, query: "" };

export function parseOrderInput(raw) {
  if (typeof raw !== "string" || !/^(0|[1-9]\d{0,2})$/.test(raw)) {
    return { ok: false, message: "Enter one whole number from 0 through 128." };
  }
  const quantity = Number(raw);
  if (!Number.isInteger(quantity) || quantity < 0 || quantity > 128) {
    return { ok: false, message: "Enter one whole number from 0 through 128." };
  }
  return { ok: true, quantity };
}

function escapeHtml(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function n0(value) { return Math.round(Number(value) || 0).toLocaleString("en-US"); }
function n1(value) {
  return (Math.round((Number(value) || 0) * 10) / 10).toLocaleString("en-US", {
    minimumFractionDigits: 1, maximumFractionDigits: 1,
  });
}

function scenarioForTrace(trace) {
  if (trace?.scenario && typeof trace.scenario === "object") {
    return structuredClone(trace.scenario);
  }
  const split = trace.seed_set === "live_y_research_eval" ? "test" : trace.split;
  const spec = scenarioFor(TIER, split, trace.seed_index);
  if (trace.seed_set === "live_y_research_eval") spec.split = "research_eval";
  spec.master_seed_hex = trace.master_seed_hex;
  return spec;
}

function setHeader(_week = "—", _cost = "—") {
  // Header status chrome is intentionally hidden in the public UI.
}

function lineChart(values, max, width = 720, height = 160) {
  if (!values.length) return "";
  const top = Math.max(1, max);
  return values.map((value, index) => {
    const x = values.length === 1 ? 0 : (index / 35) * width;
    const y = height - Math.max(0, Math.min(1, value / top)) * (height - 8) - 4;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

function meterPct(inventory) {
  return Math.max(0, Math.min(100, Math.round((Number(inventory) || 0) / 24 * 100)));
}

function inventoryMeter({ visible, inventory = 0, backlog = 0 } = {}) {
  const danger = visible && backlog > 0;
  const pct = visible ? meterPct(inventory) : 0;
  const value = visible ? `${n0(inventory)}${backlog > 0 ? ` · BL ${n0(backlog)}` : ""}` : "——";
  return `<div class="inv-meter ${danger ? "backlog" : ""}" aria-hidden="true">
    <span class="inv-label">Inventory</span>
    <div class="track"><span class="fill" style="width:${pct}%"></span></div>
    <span class="inv-value">${value}</span>
  </div>`;
}

function buildingSvg(role) {
  if (role === "factory") {
    return `<svg class="building-svg" viewBox="0 0 120 88" aria-hidden="true">
      <circle class="smoke" cx="86" cy="10" r="3"/><circle class="smoke" cx="90" cy="4" r="2.4"/>
      <circle class="smoke" cx="82" cy="3" r="2"/><circle class="smoke" cx="94" cy="9" r="1.8"/>
      <rect class="soft" x="78" y="14" width="10" height="28"/>
      <path class="soft" d="M14 38 L30 22 L46 38 L62 22 L78 38 L94 22 L110 38 V78 H14 Z"/>
      <rect class="soft" x="22" y="44" width="10" height="10"/><rect class="soft" x="38" y="44" width="10" height="10"/>
      <rect class="soft" x="54" y="44" width="10" height="10"/><rect class="soft" x="70" y="44" width="10" height="10"/>
      <circle class="gear" cx="96" cy="58" r="8"/><circle class="gear" cx="96" cy="58" r="3"/>
      <path class="stroke" d="M104 70 L116 78 L104 78 Z"/>
    </svg>`;
  }
  if (role === "retailer_a" || role === "retailer_b") {
    return `<svg class="building-svg" viewBox="0 0 120 88" aria-hidden="true">
      <path class="soft" d="M18 36 H102 V78 H18 Z"/>
      <path class="awning" d="M14 36 H106 L100 48 H20 Z"/>
      <path class="awning" d="M22 36 L28 48 M36 36 L42 48 M50 36 L56 48 M64 36 L70 48 M78 36 L84 48 M92 36 L98 48" fill="none"/>
      <rect class="soft" x="28" y="52" width="36" height="26"/>
      <rect class="crate-fill" x="36" y="58" width="12" height="14" rx="1"/>
      <rect class="soft" x="78" y="52" width="16" height="26"/>
      <circle cx="90" cy="66" r="1.4" fill="var(--ink)"/>
    </svg>`;
  }
  return `<svg class="building-svg" viewBox="0 0 120 88" aria-hidden="true">
    <path class="soft" d="M20 40 L60 18 L100 40 V78 H20 Z"/>
    <rect class="soft" x="34" y="46" width="12" height="12"/><rect class="soft" x="74" y="46" width="12" height="12"/>
    <rect class="soft" x="48" y="52" width="24" height="26"/>
    <path class="stroke" d="M56 52 V78 M64 52 V78"/>
  </svg>`;
}

function ordersRailSvg() {
  return `<svg viewBox="0 0 720 28" preserveAspectRatio="none" aria-hidden="true">
    <line x1="24" y1="14" x2="696" y2="14" stroke="var(--accent)" stroke-width="1.5" stroke-dasharray="5 5"/>
    <circle cx="90" cy="14" r="3.5" fill="var(--accent)"/>
    <circle cx="250" cy="14" r="3.5" fill="var(--accent)"/>
    <circle cx="410" cy="14" r="3.5" fill="var(--accent)"/>
    <circle cx="570" cy="14" r="3.5" fill="var(--accent)"/>
    <path d="M160 14 L148 10 V18 Z" fill="var(--accent)"/>
    <path d="M320 14 L308 10 V18 Z" fill="var(--accent)"/>
    <path d="M480 14 L468 10 V18 Z" fill="var(--accent)"/>
    <path d="M640 14 L628 10 V18 Z" fill="var(--accent)"/>
  </svg>`;
}

function conveyorSvg() {
  return `<svg class="conveyor-svg" viewBox="0 0 720 36" preserveAspectRatio="none" aria-hidden="true">
    <line class="belt-line" x1="28" y1="18" x2="692" y2="18"/>
    <line class="belt-line" x1="28" y1="24" x2="692" y2="24"/>
    <circle class="belt-roller" cx="22" cy="21" r="9"/><path d="M17 16 L27 26 M27 16 L17 26" stroke="var(--ink)" stroke-width="1.4"/>
    <circle class="belt-roller" cx="698" cy="21" r="9"/><path d="M693 16 L703 26 M703 16 L693 26" stroke="var(--ink)" stroke-width="1.4"/>
    <g class="belt-crates">
      <g transform="translate(100 8)">${crateGlyph()}</g>
      <g transform="translate(260 8)">${crateGlyph()}</g>
      <g transform="translate(420 8)">${crateGlyph()}</g>
      <g transform="translate(580 8)">${crateGlyph()}</g>
    </g>
  </svg>`;
}

function crateGlyph() {
  return `<rect class="crate-block" width="22" height="14" rx="1"/>
    <circle class="crate-dot" cx="6" cy="5" r="1.2"/><circle class="crate-dot" cx="11" cy="5" r="1.2"/><circle class="crate-dot" cx="16" cy="5" r="1.2"/>
    <circle class="crate-dot" cx="6" cy="10" r="1.2"/><circle class="crate-dot" cx="11" cy="10" r="1.2"/><circle class="crate-dot" cx="16" cy="10" r="1.2"/>`;
}

function stationCard(node, { briefing, role, states }) {
  const isYou = node === role;
  const state = states[node];
  const visible = !briefing && isYou && state;
  const selectable = briefing && node === "wholesaler";
  const tag = isYou ? "YOU" : briefing ? (node === "wholesaler" ? "AVAILABLE" : "LOCKED") : "SEALED";
  const detail = briefing
    ? `<p>${escapeHtml(ROLE_NOTE[node])}</p>`
    : inventoryMeter({
      visible,
      inventory: state?.inventory,
      backlog: state?.backlog,
    });
  const classes = [
    "chain-card",
    isYou ? "selected" : "",
    selectable ? "selectable" : "",
    briefing && !selectable ? "locked" : "",
  ].filter(Boolean).join(" ");
  return `<button class="${classes}" data-role="${node}" type="button" ${selectable ? "" : "disabled"}>
    <span class="chain-tag">${tag}</span>
    ${buildingSvg(node)}
    <strong>${ROLE_LABEL[node]}</strong>${detail}</button>`;
}

function chainHtml({ briefing = false } = {}) {
  const role = active?.role || selectedRole;
  const transition = active?.episode.operationalTransitions.at(-1);
  const states = transition?.states_after_fulfillment || {};
  const ctx = { briefing, role, states };
  const cards = ROLES.map((node) => {
    const playerCard = stationCard(node, ctx);
    if (node !== role) return playerCard;
    const modelDetail = briefing
      ? "<p>The same seat, same seed, played in a sealed parallel episode.</p>"
      : inventoryMeter({ visible: false });
    return `<div class="chain-pair chain-${node}">${playerCard}
      <div class="chain-card companion"><span class="chain-tag">TRAINED QWEN</span>${buildingSvg(node)}<strong>${ROLE_LABEL[node]}</strong>${modelDetail}</div></div>`;
  });
  return `<div class="chain-board" aria-label="Five-node Y supply chain">
    <div class="orders-rail">${ordersRailSvg()}<span class="rail-label">Orders</span></div>
    <div class="chain">
      <div class="retailers">${cards[0]}${cards[1]}</div>
      <div class="fork" aria-hidden="true"><i></i><i></i><b></b></div>
      ${cards[2]}<span class="link link-one" aria-hidden="true"></span>${cards[3]}<span class="link link-two" aria-hidden="true"></span>${cards[4]}
    </div>
    <div class="conveyor">${conveyorSvg()}<span class="rail-label">Inventory</span></div>
  </div>`;
}

function briefingHtml() {
  return `<section class="briefing" aria-labelledby="briefing-title">
    <div class="hero">
      <h1 id="briefing-title">You hold one seat in a five-node supply chain.</h1>
      <p>Every week you see your own inventory, your own backlog, and the order that arrived from downstream. Nothing else. You place one order upstream and it lands three weeks later. Thirty-six weeks. Holding costs 0.5 per unit per week, backlog costs 1.0.</p>
      <p>The trained Qwen3.5-4B GRPO policy plays the same seat, on the same benchmark seed, against the same counterparties, in a separate sealed episode. Neither of you sees the other until week 36.</p>
    </div>
    <p class="section-label seat-label">Choose your seat</p>
    ${chainHtml({ briefing: true })}
    <div class="start-row"><button id="start-game" class="primary-button" type="button">Begin week 1 <span>→</span></button></div>
  </section>`;
}

function statHtml(label, value, danger = false) {
  return `<div class="stat ${danger ? "stat-danger" : ""}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function chartArea(values, max, width = 720, height = 160) {
  if (!values.length) return "";
  const line = lineChart(values, max, width, height);
  const lastX = values.length === 1 ? 0 : width;
  return `0,${height} ${line} ${lastX.toFixed(1)},${height}`;
}

function graphHtml(title, labels, lines, max) {
  const legend = labels.map((label) => `<span class="chart-pill ${label.className}"><i></i>${escapeHtml(label.text)}</span>`).join("");
  const areas = lines.map((line) => (
    line.fill
      ? `<polygon class="area ${line.className}" points="${chartArea(line.values, max)}"></polygon>`
      : ""
  )).join("");
  const polylines = lines.map((line) => `<polyline class="${line.className}" points="${lineChart(line.values, max, 720, 160)}"></polyline>`).join("");
  return `<section class="graph">
    <div class="graph-head"><span>${escapeHtml(title)}</span><div class="chart-legend">${legend}</div></div>
    <div class="chart"><svg viewBox="0 0 720 160" preserveAspectRatio="none" role="img" aria-label="${escapeHtml(title)}">
      ${areas}${polylines}
    </svg></div>
    <div class="chart-weeks"><span>Week 1</span><span>Week 36</span></div>
  </section>`;
}

function gameHtml() {
  const { observation, episode, order } = active;
  const { state, costs } = observation;
  const history = episode.histories[active.role];
  const inventory = history.map((row) => row.ending_inventory);
  const backlog = history.map((row) => row.ending_backlog);
  const orders = history.map((row) => row.order_placed);
  const incoming = history.map((row) => row.incoming_demand_or_order);
  const stateMax = Math.max(8, ...inventory, ...backlog);
  const orderMax = Math.max(8, ...orders, ...incoming);
  return `<section class="game" aria-label="Weekly decision">
    ${chainHtml()}
    <div class="game-grid">
      <div class="decision-panel">
        <section class="position-board" aria-label="Your position">
          <div class="stats-grid">
            ${statHtml("On hand", n0(state.inventory_on_hand))}
            ${statHtml("Backlog", n0(state.backlog), state.backlog > 0)}
            ${statHtml("On order", n0(state.on_order))}
            ${statHtml("Incoming", n0(state.shipment_received))}
            ${statHtml("Demand", n0(state.incoming_demand_or_order))}
            ${statHtml("Week cost", n1(costs.current_inventory_backlog_cost))}
          </div>
        </section>
        <form id="order-form" class="order-form" novalidate>
          <p class="section-label">Place order upstream</p>
          <div class="order-number"><output id="order-value">${order}</output></div>
          <input id="order-range" type="range" min="0" max="128" step="1" value="${order}" aria-label="Order quantity">
          <div class="range-labels"><span>0</span><span>128</span></div>
          <p id="order-error" class="error-message" role="alert"></p>
          <div class="order-actions"><button id="minus-order" type="button" aria-label="Decrease order">−</button><button id="plus-order" type="button" aria-label="Increase order">+</button><button class="primary-button" type="submit">Advance week</button></div>
          <button id="fast-forward" class="hold-button" type="button">Hold to repeat ▸▸</button>
        </form>
      </div>
      <div class="graphs">
        ${graphHtml("Stock", [{ text: "On hand", className: "blue" }, { text: "Backlog", className: "red" }], [{ values: inventory, className: "blue", fill: true }, { values: backlog, className: "red", fill: true }], stateMax)}
        ${graphHtml("Flow", [{ text: "You ordered", className: "accent" }, { text: "Demand", className: "muted" }], [{ values: incoming, className: "muted", fill: true }, { values: orders, className: "accent", fill: true }], orderMax)}
      </div>
    </div>
  </section>`;
}

function ratio(values) {
  if (values.length < 2) return "—";
  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - average) ** 2, 0) / (values.length - 1);
  return n1(variance);
}

function traceWeeks() {
  const weeks = active?.trace?.weeks;
  if (!Array.isArray(weeks)) return [];
  return weeks.filter((entry) => entry && Number.isFinite(Number(entry.week)));
}

export function thoughtGroups(weeks) {
  const groups = [];
  for (let start = 0; start < weeks.length; start += THOUGHT_GROUP_SIZE) {
    groups.push(weeks.slice(start, start + THOUGHT_GROUP_SIZE));
  }
  return groups;
}

function groupLabel(group) {
  return `Weeks ${group[0].week}–${group.at(-1).week}`;
}

/** Split on the query first, escape each piece: never highlights inside an entity. */
export function highlight(text, query) {
  const needle = query.trim();
  if (!needle) return escapeHtml(text);
  const pattern = needle.replaceAll(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return text.split(new RegExp(`(${pattern})`, "gi"))
    .map((part, index) => (index % 2 ? `<mark>${escapeHtml(part)}</mark>` : escapeHtml(part)))
    .join("");
}

/** Free-text over the notes, plus a bare week number so "14" jumps to week 14. */
export function searchWeeks(weeks, query) {
  const needle = query.trim().toLowerCase();
  if (!needle) return null;
  const asWeek = /^(?:week\s*)?(\d{1,2})$/.exec(needle);
  const wanted = asWeek ? Number(asWeek[1]) : null;
  return weeks.filter((entry) => (
    String(entry.thought || "").toLowerCase().includes(needle) || entry.week === wanted
  ));
}

function thoughtBubble(entry, query) {
  const humanOrder = active.actions[entry.week - 1];
  const body = entry.thought
    ? `<p>${highlight(entry.thought, query)}</p>`
    : '<p class="thought-silent">No note recorded this week.</p>';
  const backlogged = Number(entry.ending_backlog) > 0;
  const facts = [
    entry.demand === undefined ? "" : `Demand ${n0(entry.demand)}`,
    backlogged
      ? `<b class="danger">Backlog ${n0(entry.ending_backlog)}</b>`
      : (entry.ending_inventory === undefined ? "" : `Stock ${n0(entry.ending_inventory)}`),
    humanOrder === undefined ? "" : `You ordered ${n0(humanOrder)}`,
  ].filter(Boolean).map((fact) => `<span>${fact}</span>`).join("");
  // No inline style: the page CSP forbids style attributes, so the entrance
  // stagger is nth-child based in styles.css.
  return `<article class="thought">
    <div class="thought-mark">
      <span class="thought-week">Week ${n0(entry.week)}</span>
      <span class="thought-order">${n0(entry.quantity)}</span>
      <span class="thought-order-note">ordered</span>
    </div>
    <div class="thought-bubble">${body}<footer>${facts}</footer></div>
  </article>`;
}

function thoughtStreamHtml() {
  const weeks = traceWeeks();
  const groups = thoughtGroups(weeks);
  const matches = searchWeeks(weeks, thoughtView.query);
  if (matches) {
    const found = matches.length === 1 ? "1 week matches" : `${matches.length} weeks match`;
    const bubbles = matches.map((entry) => thoughtBubble(entry, thoughtView.query)).join("");
    return `<p class="thought-count">${found} “${escapeHtml(thoughtView.query.trim())}”</p>${
      bubbles || '<p class="thought-empty">Nothing in the model\'s notes for that. Clear the search to browse by chapter.</p>'}`;
  }
  const index = Math.max(0, Math.min(groups.length - 1, thoughtView.group));
  const bubbles = groups[index].map((entry) => thoughtBubble(entry, "")).join("");
  const previous = groups[index - 1];
  const next = groups[index + 1];
  return `${bubbles}<div class="thought-nav">
    <button type="button" data-step="-1" ${previous ? "" : "disabled"}>← ${previous ? groupLabel(previous) : "Start"}</button>
    <span>Chapter ${index + 1} of ${groups.length}</span>
    <button type="button" data-step="1" ${next ? "" : "disabled"}>${next ? groupLabel(next) : "End"} →</button>
  </div>`;
}

function thoughtsHtml() {
  const weeks = traceWeeks();
  if (!weeks.length) return "";
  const tabs = thoughtGroups(weeks).map((group, index) => `<button class="thought-tab ${
    index === thoughtView.group ? "current" : ""}" type="button" role="tab" aria-selected="${
    index === thoughtView.group}" data-group="${index}">${groupLabel(group)}</button>`).join("");
  return `<section class="thoughts" aria-labelledby="thoughts-title">
    <p class="section-label">Week by week · the model's own notes</p>
    <div class="thoughts-head">
      <h2 id="thoughts-title">What the model was thinking</h2>
      <p>Every note was written in the same breath as that week's order, before the model saw how it turned out. Six weeks to a chapter.</p>
    </div>
    <div class="thought-controls">
      <div class="thought-tabs" role="tablist" aria-label="Chapters of the recorded episode">${tabs}</div>
      <label class="thought-search"><span class="visually-hidden">Search the notes</span>
        <input id="thought-search" type="search" placeholder="Search notes or a week number" value="${escapeHtml(thoughtView.query)}" autocomplete="off"></label>
    </div>
    <div id="thought-stream" class="thought-stream" aria-live="polite">${thoughtStreamHtml()}</div>
  </section>`;
}

function renderThoughtStream() {
  const stream = document.querySelector("#thought-stream");
  if (!stream) return;
  stream.innerHTML = thoughtStreamHtml();
  const browsing = !thoughtView.query.trim();
  document.querySelectorAll(".thought-tab").forEach((tab) => {
    const current = browsing && Number(tab.dataset.group) === thoughtView.group;
    tab.classList.toggle("current", current);
    tab.setAttribute("aria-selected", String(current));
  });
}

function bindDebrief() {
  document.querySelector("#new-game").addEventListener("click", renderBriefing);
  const search = document.querySelector("#thought-search");
  if (!search) return;
  search.addEventListener("input", () => {
    thoughtView = { ...thoughtView, query: search.value };
    renderThoughtStream();
  });
  document.querySelectorAll(".thought-tab").forEach((tab) => tab.addEventListener("click", () => {
    thoughtView = { group: Number(tab.dataset.group), query: "" };
    search.value = "";
    renderThoughtStream();
  }));
  document.querySelector("#thought-stream").addEventListener("click", (event) => {
    const step = event.target.closest("[data-step]");
    if (!step || step.disabled) return;
    const groups = thoughtGroups(traceWeeks()).length;
    thoughtView = {
      group: Math.max(0, Math.min(groups - 1, thoughtView.group + Number(step.dataset.step))),
      query: "",
    };
    search.value = "";
    renderThoughtStream();
  });
}

function debriefHtml() {
  const grade = active.episode.outcome.grade;
  const humanCost = grade.primary.local_total_cost;
  const baseCost = grade.primary.paired_base_stock_local_total_cost;
  let modelCost = null;
  let modelEpisode = null;
  if (active.role === "wholesaler" && active.trace) {
    const replay = replayActions(publicScenario(scenarioForTrace(active.seed)), active.role, active.trace.actions).episode;
    modelEpisode = replay;
    modelCost = replay.outcome.grade.primary.local_total_cost;
  }
  const lead = modelCost === null ? "Your 36-week episode is complete." : humanCost <= modelCost ? "You beat trained Qwen." : "Trained Qwen finished lower.";
  const rows = ROLES.map((role) => {
    const history = active.episode.histories[role];
    const orders = history.map((row) => row.order_placed);
    const demand = history.map((row) => row.incoming_demand_or_order);
    const avg = orders.reduce((sum, value) => sum + value, 0) / Math.max(1, orders.length);
    return `<div class="chain-row ${role === active.role ? "you" : ""}"><span>${ROLE_LABEL[role]}</span><span>${n1(avg)}</span><span>${n0(Math.max(0, ...orders))}</span><span>${ratio(orders)}/${ratio(demand)}</span><span>${n1(active.episode.cumulativeCosts[role])}</span></div>`;
  }).join("");
  const yours = active.episode.histories[active.role].map((row) => row.order_placed);
  const demand = active.episode.histories[active.role].map((row) => row.incoming_demand_or_order);
  const comparison = modelCost === null ? [] : active.trace.actions;
  const max = Math.max(8, ...yours, ...demand, ...comparison);
  const cumulativeCosts = (history) => {
    let running = 0;
    return history.map((row) => {
      running += Number(row.local_cost);
      return running;
    });
  };
  const humanCosts = cumulativeCosts(active.episode.histories[active.role]);
  const modelCosts = Array.isArray(active.trace?.costs_over_time)
    ? active.trace.costs_over_time
    : modelEpisode ? cumulativeCosts(modelEpisode.histories[active.role]) : [];
  const costMax = Math.max(1, ...humanCosts, ...modelCosts);
  const headToHeadScore = modelCost === null
    ? null
    : humanCost + modelCost > 0 ? 100 * modelCost / (humanCost + modelCost) : 50;
  return `<section class="debrief" aria-labelledby="debrief-title">
    <div class="hero"><h1 id="debrief-title">${lead}</h1><p>Totals include the 36 operational weeks, deterministic settlement, and terminal inventory-position exposure.</p></div>
    <div class="final-cards">
      <article><span>YOUR COST</span><strong>${n1(humanCost)}</strong><p>Local total cost at ${ROLE_LABEL[active.role]}. Adaptive base-stock cost: ${n1(baseCost)}.</p></article>
      <article><span>${modelCost === null ? "REFERENCE COST" : "TRAINED QWEN"}</span><strong>${n1(modelCost ?? baseCost)}</strong><p>${modelCost === null ? "Adaptive base-stock comparison." : "Qwen3.5-4B GRPO on your exact benchmark seed."}</p></article>
      <article><span>HEAD-TO-HEAD SCORE</span><strong>${headToHeadScore === null ? "—" : headToHeadScore.toFixed(1)}</strong><p>50 is a tie. Higher means you finished with lower cost than trained Qwen.</p></article>
    </div>
    ${graphHtml("Cumulative local cost · operational weeks", [{ text: "You", className: "light" }, { text: modelCost === null ? "Qwen unavailable" : "Trained Qwen", className: "blue" }], [{ values: humanCosts, className: "light" }, { values: modelCosts, className: "blue" }], costMax)}
    ${graphHtml("Orders placed · fog lifted", [{ text: "You", className: "light" }, { text: modelCost === null ? "Qwen unavailable" : "Trained Qwen", className: "blue" }, { text: "Demand", className: "muted" }], [{ values: demand, className: "muted" }, { values: comparison, className: "blue" }, { values: yours, className: "light" }], max)}
    <section class="chain-table"><p class="section-label">Full chain · your episode</p><div class="chain-row heading"><span>Node</span><span>Mean order</span><span>Peak order</span><span>Order/demand variance</span><span>Operational cost</span></div>${rows}</section>
    ${thoughtsHtml()}
    <div class="start-row"><button id="new-game" class="primary-button" type="button">Play again</button></div>
  </section>`;
}

function recordFor(status) {
  const completed = status === "completed";
  const grade = completed ? active.episode.outcome.grade : null;
  return {
    session_uuid: active.sessionUuid, timestamp: active.timestamp,
    env_version: active.episode.spec.environment_version,
    tier: TIER, role: active.role, split: active.episode.spec.split,
    seed_index: active.seed.seed_index, seed: active.episode.spec.master_seed_hex,
    scenario_id: active.episode.spec.scenario_id, variant: VARIANT, status, completed,
    actions: [...active.actions], weekly: [...active.weekly],
    final_total_cost: completed ? grade.primary.local_total_cost : null,
    base_stock_cost: completed ? grade.primary.paired_base_stock_local_total_cost : null,
    episode_reward: completed ? grade.episode_reward : null,
  };
}

function sendCurrentRecord(status) {
  if (!active || logSent) return false;
  logSent = true;
  return sendTelemetry(recordFor(status));
}

function bindBriefing() {
  document.querySelectorAll("[data-role].selectable").forEach((card) => card.addEventListener("click", () => {
    selectedRole = "wholesaler";
    renderBriefing();
  }));
  document.querySelector("#start-game").addEventListener("click", () => {
    selectedRole = "wholesaler";
    const random = new Uint32Array(1);
    globalThis.crypto?.getRandomValues?.(random);
    const catalogIndex = random[0] % catalog.seeds.length;
    const seed = catalog.seeds[catalogIndex];
    const spec = publicScenario(scenarioForTrace(seed));
    const episode = new BeerEpisode(spec, "wholesaler");
    active = { seed, episode, role: "wholesaler", observation: episode.start(), order: 8, actions: [], weekly: [], sessionUuid: createSessionUuid(), timestamp: new Date().toISOString(), trace: seed };
    logSent = false;
    renderGame();
  });
}

function renderBriefing() {
  clearInterval(fastForwardTimer);
  active = null;
  logSent = false;
  shippingPulse = false;
  document.querySelector("#app").innerHTML = briefingHtml();
  setHeader();
  bindBriefing();
}

function updateOrder(value) {
  active.order = Math.max(0, Math.min(128, Math.round(value)));
  document.querySelector("#order-range").value = String(active.order);
  document.querySelector("#order-value").textContent = String(active.order);
}

function commitOrder() {
  if (!active || active.episode.done) return;
  const result = active.episode.placeOrder(active.order);
  active.actions.push(active.order);
  const transition = active.episode.operationalTransitions.at(-1);
  const state = transition.states_after_fulfillment[active.role];
  active.weekly.push({ week: transition.week, inventory: state.inventory, backlog: state.backlog, local_cost: transition.local_costs[active.role] });
  if (result.done) {
    clearInterval(fastForwardTimer);
    sendCurrentRecord("completed");
    thoughtView = { group: 0, query: "" };
    document.querySelector("#app").innerHTML = debriefHtml();
    setHeader("36/36", n1(active.episode.cumulativeCosts[active.role]));
    bindDebrief();
    return;
  }
  shippingPulse = true;
  active.observation = result.next_observation;
  renderGame();
}

function renderGame() {
  document.querySelector("#app").innerHTML = gameHtml();
  setHeader(`${active.observation.week}/36`, n1(active.observation.costs.cumulative_local_cost_through_previous_week));
  if (shippingPulse) {
    shippingPulse = false;
    const board = document.querySelector(".chain-board");
    if (board) {
      // Force a reflow so the shipping class can replay after each week.
      void board.offsetWidth;
      board.classList.add("chain--shipping");
      window.setTimeout(() => board.classList.remove("chain--shipping"), 700);
    }
  }
  const range = document.querySelector("#order-range");
  range.addEventListener("input", () => updateOrder(Number(range.value)));
  document.querySelector("#minus-order").addEventListener("click", () => updateOrder(active.order - 1));
  document.querySelector("#plus-order").addEventListener("click", () => updateOrder(active.order + 1));
  document.querySelector("#order-form").addEventListener("submit", (event) => { event.preventDefault(); commitOrder(); });
  const hold = document.querySelector("#fast-forward");
  const stop = () => clearInterval(fastForwardTimer);
  hold.addEventListener("pointerdown", () => { commitOrder(); fastForwardTimer = setInterval(commitOrder, 220); });
  ["pointerup", "pointerleave", "pointercancel"].forEach((event) => hold.addEventListener(event, stop));
}

async function loadCatalog() {
  const response = await fetch("./data/llm-comparison.json", { cache: "no-cache" });
  if (!response.ok) throw new Error(`trace catalog returned ${response.status}`);
  const payload = await response.json();
  if (!Array.isArray(payload.seeds) || payload.seeds.length !== 16) throw new Error("trace catalog must contain all 16 trained-Qwen benchmark traces");
  return payload;
}

function replayHtml() {
  const week = Number(replayCatalog?.week || 0);
  const maxWeek = Math.max(1, ...replayCatalog.models.map((model) => model.frames.length - 1));
  const cards = replayCatalog.models.map((model) => {
    const frame = model.frames[Math.min(week, model.frames.length - 1)];
    return `<article class="replay-card"><span class="section-label">${escapeHtml(model.label)}</span>
      <strong>Week ${n0(frame.week)}</strong><p>Order ${frame.order == null ? "—" : n0(frame.order)} · Inventory ${n0(frame.inventory)} · Backlog ${n0(frame.backlog)}</p>
      <p class="replay-cost">Running local cost ${n1(frame.cost)}</p><p>Final cost ${n1(model.local_total_cost)}</p></article>`;
  }).join("");
  return `<section class="replay" aria-labelledby="replay-title"><div class="hero"><p class="section-label">Deterministic standard replay</p><h1 id="replay-title">Three policies, one hidden supply chain.</h1><p>Every card is replayed on seed ${escapeHtml(replayCatalog.seed)}. Move through the same delayed consequences together.</p></div>
    <div class="replay-controls"><button id="replay-play" class="primary-button" type="button">${replayTimer ? "Pause" : "Play"}</button><label for="replay-week">Week <output id="replay-week-value">${week}</output></label><input id="replay-week" type="range" min="0" max="${maxWeek}" value="${week}" step="1" aria-label="Replay week"></div>
    <div class="replay-grid">${cards}</div><div class="start-row"><button id="replay-back" class="primary-button" type="button">Back to game</button></div></section>`;
}

function renderReplay() {
  if (!replayCatalog) return;
  document.querySelector("#app").innerHTML = replayHtml();
  const slider = document.querySelector("#replay-week");
  slider.addEventListener("input", () => { replayCatalog.week = Number(slider.value); renderReplay(); });
  document.querySelector("#replay-play").addEventListener("click", () => {
    if (replayTimer) { clearInterval(replayTimer); replayTimer = null; renderReplay(); return; }
    replayTimer = setInterval(() => {
      const max = Math.max(...replayCatalog.models.map((model) => model.frames.length - 1));
      replayCatalog.week = replayCatalog.week >= max ? 0 : replayCatalog.week + 1;
      renderReplay();
    }, 550);
    renderReplay();
  });
  document.querySelector("#replay-back").addEventListener("click", () => { if (replayTimer) clearInterval(replayTimer); replayTimer = null; renderBriefing(); });
}

export async function initialize() {
  configureTelemetry(globalThis.BEER_GAME_CONFIG?.loggingEndpoint || "");
  try {
    catalog = await loadCatalog();
    const replayResponse = await fetch("./data/benchmark-replay.json", { cache: "no-cache" });
    if (replayResponse.ok) replayCatalog = { ...(await replayResponse.json()), week: 0 };
    if (new URLSearchParams(window.location.search).get("view") === "replay" && replayCatalog) renderReplay();
    else renderBriefing();
  } catch (error) {
    console.error(error);
    document.querySelector("#app").innerHTML = '<section class="loading-card"><p class="section-label">Unable to start</p><h1>Scenario data could not be loaded.</h1><p>Reload the page to try again.</p></section>';
  }
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  window.addEventListener("pagehide", () => { if (active && !active.episode.done) sendCurrentRecord("abandoned"); });
  window.addEventListener("keydown", (event) => {
    if (!active || active.episode.done) return;
    if (event.key === "Enter") commitOrder();
    if (event.key === "ArrowUp") { event.preventDefault(); updateOrder(active.order + 1); }
    if (event.key === "ArrowDown") { event.preventDefault(); updateOrder(active.order - 1); }
  });
  void initialize();
}
