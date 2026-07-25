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
    sterman: "Sterman",
    ippo: "IPPO",
  };

  const els = {
    setup: document.getElementById("screen-setup"),
    play: document.getElementById("screen-play"),
    end: document.getElementById("screen-end"),
    setupForm: document.getElementById("setup-form"),
    setupError: document.getElementById("setup-error"),
    seed: document.getElementById("seed-input"),
    start: document.getElementById("btn-start"),
    youRole: document.getElementById("you-role"),
    aiLabel: document.getElementById("ai-label"),
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
    endOwn: document.getElementById("end-own"),
    endSystem: document.getElementById("end-system"),
    endWeeks: document.getElementById("end-weeks"),
  };

  /** @type {Record<string, HTMLElement>} */
  const nodes = {};

  let history = [];
  let humanRole = null;
  let aiMode = null;
  let phase = "setup";
  let awaiting = false;
  let orderCap = 128;
  let reconnectTimer = null;
  let ws = null;

  function prettyRole(role) {
    return String(role || "—").replace(/_/g, " ");
  }

  function fmt(n, digits = 1) {
    if (n == null || Number.isNaN(n)) return "—";
    return Number(n).toFixed(digits);
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

  function makeNode(role) {
    const node = document.createElement("div");
    node.className = "node fogged";
    node.dataset.role = role;
    node.innerHTML = `
      <h3 class="node-title">${prettyRole(role)}</h3>
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

  function updateFog() {
    for (const role of ROLES) {
      const node = nodes[role];
      if (!node) continue;
      const yours = role === humanRole;
      node.classList.toggle("fogged", !yours);
      node.classList.toggle("yours", yours);
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

    updateFog();
    if (flash && humanRole && nodes[humanRole] && week > 0) {
      nodes[humanRole].classList.remove("flash");
      void nodes[humanRole].offsetWidth;
      nodes[humanRole].classList.add("flash");
    }
    drawChart();
  }

  function applyStatus(msg) {
    if (msg.phase) phase = msg.phase;
    if (typeof msg.awaiting_order === "boolean") awaiting = msg.awaiting_order;
    if (msg.human_role) humanRole = msg.human_role;
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
    }
    els.orderBtn.disabled = !awaiting;
    els.orderQty.disabled = !awaiting;
  }

  function applyReveal(reveal) {
    if (!reveal) return;
    els.endRole.textContent = prettyRole(reveal.human_role);
    els.endAi.textContent = AI_LABELS[reveal.ai_mode] || reveal.ai_mode || "—";
    els.endOwn.textContent = fmt(reveal.cumulative_own_cost);
    els.endSystem.textContent = fmt(reveal.cumulative_system_cost);
    els.endWeeks.textContent = String(reveal.horizon ?? "—");
    showScreen("end");
  }

  function drawChart() {
    const canvas = els.canvas;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 640;
    const cssH = 180;
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = "rgba(20, 28, 40, 0.45)";
    ctx.fillRect(0, 0, cssW, cssH);

    const pad = { l: 34, r: 10, t: 14, b: 24 };
    const plotW = cssW - pad.l - pad.r;
    const plotH = cssH - pad.t - pad.b;
    const frames = history.filter((f) => f.t > 0);

    let maxY = 1;
    for (const f of frames) {
      maxY = Math.max(
        maxY,
        f.own_order ?? f.last_order_placed ?? 0,
        f.last_demand_or_order ?? 0,
      );
    }
    maxY = Math.ceil(maxY * 1.1) || 1;

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

    if (frames.length < 1) return;
    const horizon = frames[frames.length - 1].horizon || 52;
    const maxT = Math.max(horizon, ...frames.map((f) => f.t));
    const color = COLORS[humanRole] || "#e8a54b";

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    frames.forEach((f, idx) => {
      const x = pad.l + ((f.t - 1) / Math.max(1, maxT - 1)) * plotW;
      const y = pad.t + plotH - ((f.own_order ?? f.last_order_placed ?? 0) / maxY) * plotH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.beginPath();
    ctx.strokeStyle = "rgba(238, 243, 248, 0.75)";
    ctx.setLineDash([6, 5]);
    frames.forEach((f, idx) => {
      const x = pad.l + ((f.t - 1) / Math.max(1, maxT - 1)) * plotW;
      const y = pad.t + plotH - ((f.last_demand_or_order ?? 0) / maxY) * plotH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
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
      applyReveal(data.reveal || {
        human_role: data.human_role,
        ai_mode: data.ai_mode,
        cumulative_own_cost: data.frame?.cumulative_own_cost,
        cumulative_system_cost: null,
        horizon: data.horizon,
      });
      return;
    }
    showScreen("play");
    if (data.frame) applyFrame(data.frame, { flash: false });
    else if (history.length) applyFrame(history[history.length - 1], { flash: false });
  }

  els.setupForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    showError(els.setupError, "");
    const fd = new FormData(els.setupForm);
    const role = String(fd.get("role"));
    const mode = String(fd.get("ai_mode"));
    const seed = Number(els.seed.value || 0);
    els.start.disabled = true;
    try {
      const data = await postJson("/api/start", { role, ai_mode: mode, seed });
      humanRole = role;
      aiMode = mode;
      history = data.history || [];
      applyStatus(data);
      showScreen("play");
      if (data.frame) applyFrame(data.frame, { flash: false });
      els.orderQty.focus();
    } catch (err) {
      showError(els.setupError, err.message || String(err));
    } finally {
      els.start.disabled = false;
    }
  });

  els.orderForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    showError(els.orderError, "");
    if (!awaiting) return;
    const quantity = Number(els.orderQty.value);
    els.orderBtn.disabled = true;
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
    }
  });

  async function goSetup() {
    showError(els.setupError, "");
    try {
      const data = await postJson("/api/control", { action: "reset" });
      history = [];
      humanRole = null;
      aiMode = null;
      phase = "setup";
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

  window.addEventListener("resize", () => drawChart());
  buildBoard();
  showScreen("setup");
  connect();
})();
