(() => {
  const ROLES = ["retailer_a", "retailer_b", "wholesaler", "distributor", "factory"];
  const RETAILERS = ["retailer_a", "retailer_b"];
  const TRUNK = ["wholesaler", "distributor", "factory"];
  const COLORS = {
    retailer_a: "#5ec4a0",
    retailer_b: "#3dd6c6",
    wholesaler: "#6aa8e8",
    distributor: "#b8a0d8",
    factory: "#e8a54b",
  };
  const AI_LABELS = {
    llm: "OpenRouter LLM",
    sterman: "Sterman",
    ippo: "IPPO",
  };
  const YOU_COLOR = "#5ec4a0";
  const AI_COLOR = "#e8a54b";
  const COST_COLOR = "#e8a54b";
  const DEMAND_COLOR = "rgba(238, 243, 248, 0.9)";

  const els = {
    setup: document.getElementById("screen-setup"),
    play: document.getElementById("screen-play"),
    end: document.getElementById("screen-end"),
    setupForm: document.getElementById("setup-form"),
    setupError: document.getElementById("setup-error"),
    seed: document.getElementById("seed-input"),
    start: document.getElementById("btn-start"),
    modelBlock: document.getElementById("model-block"),
    modelSelect: document.getElementById("model-select"),
    dualModels: document.getElementById("dual-models"),
    retailerAModel: document.getElementById("retailer-a-model"),
    retailerBModel: document.getElementById("retailer-b-model"),
    openrouterStatus: document.getElementById("openrouter-status"),
    youRole: document.getElementById("you-role"),
    aiLabel: document.getElementById("ai-label"),
    modelLine: document.getElementById("model-line"),
    week: document.getElementById("week"),
    horizon: document.getElementById("horizon"),
    weekCost: document.getElementById("week-cost"),
    ownCost: document.getElementById("own-cost"),
    panelTitle: document.getElementById("panel-title"),
    inv: document.getElementById("inv"),
    bl: document.getElementById("bl"),
    demandLabel: document.getElementById("demand-label"),
    demandOrIncoming: document.getElementById("demand-or-incoming"),
    received: document.getElementById("received"),
    onOrder: document.getElementById("on-order"),
    lastOrder: document.getElementById("last-order"),
    shipPipe: document.getElementById("ship-pipe"),
    orderPipe: document.getElementById("order-pipe"),
    orderForm: document.getElementById("order-form"),
    orderQty: document.getElementById("order-qty"),
    orderBtn: document.getElementById("btn-order"),
    orderError: document.getElementById("order-error"),
    demandA: document.getElementById("demand-a"),
    demandB: document.getElementById("demand-b"),
    demandASlot: document.getElementById("demand-a-slot"),
    demandBSlot: document.getElementById("demand-b-slot"),
    retailers: document.getElementById("retailers"),
    trunk: document.getElementById("trunk"),
    canvas: document.getElementById("order-chart"),
    restart: document.getElementById("btn-restart"),
    again: document.getElementById("btn-again"),
    conn: document.getElementById("conn-status"),
    endRole: document.getElementById("end-role"),
    endAi: document.getElementById("end-ai"),
    endAiRoles: document.getElementById("end-ai-roles"),
    endModels: document.getElementById("end-models"),
    endOwn: document.getElementById("end-own"),
    endAiOwn: document.getElementById("end-ai-own"),
    endSystem: document.getElementById("end-system"),
    endAiSystem: document.getElementById("end-ai-system"),
    endWeeks: document.getElementById("end-weeks"),
    endVerdict: document.getElementById("end-verdict"),
    endCumChart: document.getElementById("end-cum-chart"),
    endWeekChart: document.getElementById("end-week-chart"),
    endOrderChart: document.getElementById("end-order-chart"),
    endInvChart: document.getElementById("end-inv-chart"),
    endBlChart: document.getElementById("end-bl-chart"),
    costBreakdown: document.getElementById("cost-breakdown"),
    costFormula: document.getElementById("cost-formula"),
    legendOrderSwatch: document.getElementById("legend-order-swatch"),
    legendOrderLabel: document.getElementById("legend-order-label"),
    legendSignalLabel: document.getElementById("legend-signal-label"),
  };

  /** @type {Record<string, HTMLElement>} */
  const nodes = {};

  let history = [];
  let humanRole = null;
  let aiMode = null;
  let modelMap = {};
  let catalog = [];
  let phase = "setup";
  let awaiting = false;
  let orderCap = 128;
  let reconnectTimer = null;
  let ws = null;
  let lastReveal = null;

  function prettyRole(role) {
    return String(role || "—").replace(/_/g, " ");
  }

  function fmt(n, digits = 1) {
    if (n == null || Number.isNaN(n)) return "—";
    return Number(n).toFixed(digits);
  }

  function selectedRole() {
    const el = els.setupForm.querySelector('input[name="role"]:checked');
    return el ? el.value : "retailer_a";
  }

  function selectedAiMode() {
    const el = els.setupForm.querySelector('input[name="ai_mode"]:checked');
    return el ? el.value : "llm";
  }

  function showScreen(name) {
    els.setup.hidden = name !== "setup";
    els.play.hidden = name !== "play";
    els.end.hidden = name !== "end";
  }

  function setConn(text, cls) {
    els.conn.textContent = text;
    els.conn.className = `status-line ${cls || ""}`;
  }

  function showError(el, message) {
    if (!message) {
      el.hidden = true;
      el.textContent = "";
      return;
    }
    el.hidden = false;
    el.textContent = message;
  }

  function fillModelSelect(select, models, preferred) {
    select.innerHTML = "";
    models.forEach((m, idx) => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.label || m.id;
      select.appendChild(opt);
      if (m.id === preferred) select.selectedIndex = idx;
    });
    if (!preferred && models.length > 1 && select === els.retailerBModel) {
      select.selectedIndex = Math.min(1, models.length - 1);
    }
  }

  function updateSetupModelUI() {
    const mode = selectedAiMode();
    const role = selectedRole();
    const isLlm = mode === "llm";
    els.modelBlock.hidden = !isLlm;
    const needsDual = isLlm && role !== "retailer_a" && role !== "retailer_b";
    els.dualModels.hidden = !needsDual;
  }

  async function loadModels() {
    try {
      const res = await fetch("/api/models");
      const data = await res.json();
      catalog = data.models || [];
      fillModelSelect(els.modelSelect, catalog, catalog[0]?.id);
      fillModelSelect(els.retailerAModel, catalog, catalog[0]?.id);
      fillModelSelect(els.retailerBModel, catalog, catalog[1]?.id || catalog[0]?.id);
      if (data.openrouter_configured) {
        els.openrouterStatus.textContent = "OpenRouter key detected — LLM opponents ready.";
        els.openrouterStatus.style.color = "var(--good)";
      } else {
        els.openrouterStatus.textContent =
          "OPENROUTER_API_KEY missing. Export it or put it in a local .env before using LLM mode.";
        els.openrouterStatus.style.color = "var(--warn)";
      }
    } catch (err) {
      els.openrouterStatus.textContent = "Could not load model catalog.";
      els.openrouterStatus.style.color = "var(--warn)";
    }
    updateSetupModelUI();
  }

  function makeNode(role) {
    const node = document.createElement("div");
    node.className = "node fogged";
    node.dataset.role = role;
    node.innerHTML = `
      <h3 class="node-title">${prettyRole(role)} <span class="ai-badge" data-k="badge">AI</span></h3>
      <div class="node-body">
        <div><span>Inv</span><strong data-k="inv">—</strong></div>
        <div><span>Backlog</span><strong data-k="bl">—</strong></div>
      </div>
      <div class="fog-mark" aria-hidden="true">—</div>
    `;
    nodes[role] = node;
    return node;
  }

  function buildBoard() {
    els.retailers.innerHTML = "";
    els.trunk.innerHTML = "";
    for (const role of RETAILERS) {
      const wrap = document.createElement("div");
      wrap.className = "retailer-slot";
      wrap.appendChild(makeNode(role));
      els.retailers.appendChild(wrap);
    }
    TRUNK.forEach((role, i) => {
      const wrap = document.createElement("div");
      wrap.className = "node-wrap";
      if (i > 0) {
        const edge = document.createElement("div");
        edge.className = "trunk-edge";
        wrap.appendChild(edge);
      }
      wrap.appendChild(makeNode(role));
      els.trunk.appendChild(wrap);
    });
  }

  function formatModelMap(map) {
    if (!map || !Object.keys(map).length) return "";
    return Object.entries(map)
      .map(([role, info]) => `${prettyRole(role)}: ${info.label || info.model}`)
      .join(" · ");
  }

  function updateCostHelp(frame) {
    const h = frame?.holding_cost ?? 0.5;
    const b = frame?.backlog_cost ?? 1.0;
    const inv = frame?.inventory ?? 0;
    const bl = frame?.backlog ?? 0;
    const week = frame?.week_cost ?? 0;
    els.costFormula.textContent =
      `Week cost = (${h} × inventory) + (${b} × backlog)`;
    if (frame && frame.t > 0) {
      els.costBreakdown.textContent =
        `This week: (${h} × ${inv}) + (${b} × ${bl}) = ${fmt(week)}`;
    } else {
      els.costBreakdown.textContent =
        `Your rates: ${h} per unit in inventory, ${b} per unit in backlog`;
    }
  }

  function updateChartLegend() {
    const color = COLORS[humanRole] || YOU_COLOR;
    els.legendOrderSwatch.style.background = color;
    els.legendOrderLabel.textContent = "Your order";
    const isRetailer = humanRole === "retailer_a" || humanRole === "retailer_b";
    els.legendSignalLabel.textContent = isRetailer
      ? "Customer demand"
      : "Incoming orders";
  }

  function updateFog() {
    for (const role of ROLES) {
      const node = nodes[role];
      if (!node) continue;
      const yours = role === humanRole;
      node.classList.toggle("fogged", !yours);
      node.classList.toggle("yours", yours);
      const badge = node.querySelector('[data-k="badge"]');
      if (badge) {
        if (yours) {
          badge.textContent = "YOU";
        } else if (modelMap[role]) {
          badge.textContent = modelMap[role].label || modelMap[role].model || "AI";
        } else {
          badge.textContent = "AI";
        }
      }
      if (yours) {
        node.querySelector('[data-k="inv"]').textContent = els.inv.textContent;
        node.querySelector('[data-k="bl"]').textContent = els.bl.textContent;
      } else {
        node.querySelector('[data-k="inv"]').textContent = "—";
        node.querySelector('[data-k="bl"]').textContent = "—";
      }
    }

    const showA = humanRole === "retailer_a";
    const showB = humanRole === "retailer_b";
    els.demandASlot.classList.toggle("visible", showA);
    els.demandBSlot.classList.toggle("visible", showB);
    if (!showA) els.demandA.textContent = "—";
    if (!showB) els.demandB.textContent = "—";
  }

  function applyFrame(frame, { flash = true } = {}) {
    if (!frame) return;
    if (frame.models) modelMap = frame.models;

    const prevWeek = Number(els.week.textContent) || 0;
    const week = frame.t ?? 0;
    els.week.textContent = String(week);
    els.horizon.textContent = `/ ${frame.horizon ?? 52}`;
    if (flash && week > prevWeek) {
      els.week.classList.remove("pulse");
      void els.week.offsetWidth;
      els.week.classList.add("pulse");
    }

    els.weekCost.textContent = fmt(frame.week_cost ?? 0);
    els.ownCost.textContent = fmt(frame.cumulative_own_cost ?? 0);
    els.inv.textContent = String(frame.inventory ?? 0);
    const bl = frame.backlog ?? 0;
    els.bl.textContent = String(bl);
    els.bl.classList.toggle("warn", bl > 0);
    els.demandOrIncoming.textContent = String(frame.last_demand_or_order ?? 0);
    els.received.textContent = String(frame.last_shipment_received ?? 0);
    els.onOrder.textContent = String(frame.on_order ?? 0);
    els.lastOrder.textContent = String(frame.last_order_placed ?? frame.own_order ?? 0);
    els.shipPipe.textContent = (frame.ship_pipeline || []).join(" · ") || "—";
    els.orderPipe.textContent = (frame.order_pipeline || []).join(" · ") || "—";

    if (frame.order_cap != null) {
      orderCap = Number(frame.order_cap);
      els.orderQty.max = String(orderCap);
    }

    if (frame.is_retailer) {
      els.demandLabel.textContent = "Customer demand";
      if (humanRole === "retailer_a") {
        els.demandA.textContent = String(frame.customer_demand ?? frame.last_demand_or_order ?? "—");
      }
      if (humanRole === "retailer_b") {
        els.demandB.textContent = String(frame.customer_demand ?? frame.last_demand_or_order ?? "—");
      }
    } else {
      els.demandLabel.textContent = "Incoming orders";
    }

    awaiting = !!frame.awaiting_order && !frame.terminated;
    els.orderBtn.disabled = !awaiting;
    els.orderQty.disabled = !awaiting;

    const line = formatModelMap(modelMap);
    if (line) {
      els.modelLine.hidden = false;
      els.modelLine.textContent = `Models: ${line}`;
    } else {
      els.modelLine.hidden = true;
    }

    updateCostHelp(frame);
    updateChartLegend();
    updateFog();
    if (flash && humanRole && nodes[humanRole] && week > 0) {
      nodes[humanRole].classList.remove("flash");
      void nodes[humanRole].offsetWidth;
      nodes[humanRole].classList.add("flash");
    }
    drawPlayChart();
  }

  function applyStatus(msg) {
    if (msg.phase) phase = msg.phase;
    if (typeof msg.awaiting_order === "boolean") awaiting = msg.awaiting_order;
    if (msg.human_role) humanRole = msg.human_role;
    if (msg.models) modelMap = msg.models;
    if (msg.ai_mode) {
      aiMode = msg.ai_mode;
      els.aiLabel.textContent = AI_LABELS[aiMode] || aiMode;
    }
    if (msg.horizon != null) els.horizon.textContent = `/ ${msg.horizon}`;
    if (msg.order_cap != null) {
      orderCap = Number(msg.order_cap);
      els.orderQty.max = String(orderCap);
    }
    if (humanRole) {
      els.youRole.textContent = prettyRole(humanRole);
      els.panelTitle.textContent = `Your station · ${prettyRole(humanRole)}`;
      updateChartLegend();
    }
    els.orderBtn.disabled = !awaiting;
    els.orderQty.disabled = !awaiting;
  }

  function drawLineChart(canvas, seriesList, { height = 200 } = {}) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 640;
    const cssH = height;
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = "rgba(20, 28, 40, 0.45)";
    ctx.fillRect(0, 0, cssW, cssH);

    const pad = { l: 36, r: 12, t: 14, b: 26 };
    const plotW = cssW - pad.l - pad.r;
    const plotH = cssH - pad.t - pad.b;

    let maxY = 1;
    let maxT = 1;
    for (const series of seriesList) {
      for (const pt of series.points) {
        maxY = Math.max(maxY, pt.y ?? 0);
        maxT = Math.max(maxT, pt.t ?? 1);
      }
    }
    maxY = Math.ceil(maxY * 1.12) || 1;

    ctx.strokeStyle = "rgba(143, 163, 187, 0.2)";
    ctx.lineWidth = 1;
    ctx.font = "11px IBM Plex Mono, monospace";
    ctx.fillStyle = "#8fa3bb";
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (plotH * i) / 4;
      const val = Math.round(maxY * (1 - i / 4));
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + plotW, y);
      ctx.stroke();
      ctx.fillText(String(val), 4, y + 4);
    }
    ctx.fillText("week", pad.l + plotW - 28, cssH - 8);

    for (const series of seriesList) {
      if (!series.points.length) continue;
      ctx.beginPath();
      ctx.strokeStyle = series.color;
      ctx.lineWidth = series.width || 2.2;
      ctx.setLineDash(series.dash || []);
      series.points.forEach((pt, idx) => {
        const x = pad.l + ((pt.t - 1) / Math.max(1, maxT - 1)) * plotW;
        const y = pad.t + plotH - ((pt.y ?? 0) / maxY) * plotH;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  function drawPlayChart() {
    const frames = history.filter((f) => f.t > 0);
    const orderColor = COLORS[humanRole] || YOU_COLOR;
    drawLineChart(els.canvas, [
      {
        color: orderColor,
        width: 2.4,
        points: frames.map((f) => ({
          t: f.t,
          y: f.own_order ?? f.last_order_placed ?? 0,
        })),
      },
      {
        color: DEMAND_COLOR,
        width: 2,
        dash: [6, 5],
        points: frames.map((f) => ({
          t: f.t,
          y: f.last_demand_or_order ?? 0,
        })),
      },
      {
        color: COST_COLOR,
        width: 2,
        points: frames.map((f) => ({
          t: f.t,
          y: f.week_cost ?? 0,
        })),
      },
    ], { height: 220 });
  }

  function drawEndCharts(reveal) {
    const human = reveal.human_series || [];
    const ai = reveal.ai_series || [];
    const demandPoints = human.map((p) => ({
      t: p.t,
      y: p.demand_or_incoming ?? 0,
    }));
    drawLineChart(els.endCumChart, [
      { color: YOU_COLOR, points: human.map((p) => ({ t: p.t, y: p.cumulative_own_cost })) },
      { color: AI_COLOR, points: ai.map((p) => ({ t: p.t, y: p.cumulative_own_cost })) },
    ]);
    drawLineChart(els.endWeekChart, [
      { color: YOU_COLOR, points: human.map((p) => ({ t: p.t, y: p.week_cost })) },
      { color: AI_COLOR, points: ai.map((p) => ({ t: p.t, y: p.week_cost })) },
    ]);
    drawLineChart(els.endOrderChart, [
      { color: YOU_COLOR, points: human.map((p) => ({ t: p.t, y: p.order })) },
      { color: AI_COLOR, points: ai.map((p) => ({ t: p.t, y: p.order })) },
      { color: DEMAND_COLOR, width: 2, dash: [6, 5], points: demandPoints },
    ]);
    drawLineChart(els.endInvChart, [
      { color: YOU_COLOR, points: human.map((p) => ({ t: p.t, y: p.inventory ?? 0 })) },
      { color: AI_COLOR, points: ai.map((p) => ({ t: p.t, y: p.inventory ?? 0 })) },
    ]);
    drawLineChart(els.endBlChart, [
      { color: YOU_COLOR, points: human.map((p) => ({ t: p.t, y: p.backlog ?? 0 })) },
      { color: AI_COLOR, points: ai.map((p) => ({ t: p.t, y: p.backlog ?? 0 })) },
    ]);
  }

  function applyReveal(reveal) {
    if (!reveal) return;
    lastReveal = reveal;
    els.endRole.textContent = prettyRole(reveal.human_role);
    els.endAi.textContent = AI_LABELS[reveal.ai_mode] || reveal.ai_mode || "—";
    const aiRoles = (reveal.ai_roles || []).map(prettyRole).join(", ");
    els.endAiRoles.textContent = aiRoles || "the other roles";
    const models = reveal.models || {};
    const shadow = reveal.shadow_model;
    const modelBits = formatModelMap(models);
    els.endModels.textContent = [
      modelBits ? `Opponent models: ${modelBits}` : "",
      shadow ? `Comparison model in your seat: ${shadow.label || shadow.model}` : "",
    ].filter(Boolean).join("\n");

    els.endOwn.textContent = fmt(reveal.cumulative_own_cost);
    els.endAiOwn.textContent = fmt(reveal.ai_own_cost);
    els.endSystem.textContent = fmt(reveal.cumulative_system_cost);
    els.endAiSystem.textContent = fmt(reveal.ai_system_cost);
    els.endWeeks.textContent = String(reveal.horizon ?? "—");

    const you = Number(reveal.cumulative_own_cost);
    const ai = Number(reveal.ai_own_cost);
    let verdict;
    let cls = "tie";
    if (Number.isFinite(you) && Number.isFinite(ai)) {
      const delta = you - ai;
      if (Math.abs(delta) < 1e-6) {
        verdict = "Tie — same cost as the AI in your seat.";
      } else if (delta < 0) {
        verdict = `You beat the AI by ${fmt(Math.abs(delta))} (lower cost wins).`;
        cls = "win";
      } else {
        verdict = `AI beat you by ${fmt(delta)} in your seat (lower cost wins).`;
        cls = "lose";
      }
    } else {
      verdict = "Comparison unavailable.";
    }
    els.endVerdict.textContent = verdict;
    els.endVerdict.className = `end-verdict ${cls}`;

    showScreen("end");
    requestAnimationFrame(() => drawEndCharts(reveal));
  }

  async function postJson(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data.detail || `Request failed (${res.status})`;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  }

  function applySnapshot(data) {
    if (Array.isArray(data.history)) history = data.history;
    applyStatus(data);
    if (data.phase === "setup" || (!data.human_role && data.phase !== "finished")) {
      showScreen("setup");
      return;
    }
    if (data.reveal || data.phase === "finished") {
      if (data.frame) applyFrame(data.frame, { flash: false });
      applyReveal(data.reveal || lastReveal);
      return;
    }
    showScreen("play");
    if (data.frame) applyFrame(data.frame, { flash: false });
    else if (history.length) applyFrame(history[history.length - 1], { flash: false });
  }

  els.setupForm.addEventListener("change", updateSetupModelUI);

  els.setupForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    showError(els.setupError, "");
    const role = selectedRole();
    const mode = selectedAiMode();
    const seed = Number(els.seed.value || 0);
    const body = { role, ai_mode: mode, seed };
    if (mode === "llm") {
      body.model = els.modelSelect.value;
      if (role !== "retailer_a" && role !== "retailer_b") {
        body.retailer_a_model = els.retailerAModel.value;
        body.retailer_b_model = els.retailerBModel.value;
        if (body.retailer_a_model === body.retailer_b_model) {
          showError(els.setupError, "Pick two different models for Retailer A and Retailer B.");
          return;
        }
      }
    }
    els.start.disabled = true;
    els.start.textContent = mode === "llm" ? "Starting (calling LLMs)…" : "Starting…";
    try {
      const data = await postJson("/api/start", body);
      humanRole = role;
      aiMode = mode;
      modelMap = data.models || {};
      history = data.history || [];
      applyStatus(data);
      showScreen("play");
      updateCostHelp(data.frame);
      if (data.frame) applyFrame(data.frame, { flash: false });
      els.orderQty.focus();
    } catch (err) {
      showError(els.setupError, err.message || String(err));
    } finally {
      els.start.disabled = false;
      els.start.textContent = "Start week 1";
    }
  });

  els.orderForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    showError(els.orderError, "");
    if (!awaiting) return;
    const quantity = Number(els.orderQty.value);
    els.orderBtn.disabled = true;
    els.orderBtn.textContent = aiMode === "llm" ? "Waiting on LLMs…" : "Commit";
    try {
      const data = await postJson("/api/order", { quantity });
      if (Array.isArray(data.history)) history = data.history;
      applyStatus(data);
      els.orderForm.classList.remove("committed");
      void els.orderForm.offsetWidth;
      els.orderForm.classList.add("committed");
      if (data.frame) applyFrame(data.frame);
      if (data.reveal || data.phase === "finished") {
        applyReveal(data.reveal);
      } else {
        els.orderQty.focus();
        els.orderQty.select();
      }
    } catch (err) {
      showError(els.orderError, err.message || String(err));
      els.orderBtn.disabled = !awaiting;
    } finally {
      els.orderBtn.textContent = "Commit";
    }
  });

  async function goSetup() {
    showError(els.setupError, "");
    try {
      const data = await postJson("/api/control", { action: "reset" });
      history = [];
      humanRole = null;
      aiMode = null;
      modelMap = {};
      phase = "setup";
      lastReveal = null;
      applySnapshot(data);
      showScreen("setup");
    } catch (err) {
      showError(els.setupError, err.message || String(err));
      showScreen("setup");
    }
  }

  els.restart.addEventListener("click", () => goSetup().catch(console.error));
  els.again.addEventListener("click", () => goSetup().catch(console.error));

  function connect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => setConn("Live", "ok");
    ws.onclose = () => {
      setConn("Reconnecting…", "bad");
      reconnectTimer = setTimeout(connect, 1000);
    };
    ws.onerror = () => setConn("Connection error", "bad");
    ws.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === "snapshot") {
        applySnapshot(msg);
        return;
      }
      if (msg.type === "status") {
        applyStatus(msg);
        if (msg.phase === "setup") showScreen("setup");
        return;
      }
      if (msg.type === "reveal") {
        applyReveal(msg);
        return;
      }
      if (msg.type === "frame") {
        const { type: _t, ...frame } = msg;
        if (frame.t === 0) history = [frame];
        else {
          const last = history[history.length - 1];
          if (!last || last.t !== frame.t) history.push(frame);
          else history[history.length - 1] = frame;
        }
        if (phase !== "finished" && els.end.hidden) showScreen("play");
        applyFrame(frame);
      }
    };
  }

  window.addEventListener("resize", () => {
    drawPlayChart();
    if (lastReveal) drawEndCharts(lastReveal);
  });
  buildBoard();
  updateCostHelp(null);
  showScreen("setup");
  loadModels();
  connect();
})();
