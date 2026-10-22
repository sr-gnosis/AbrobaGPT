(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const esc = (t) => String(t).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  async function api(path, body) {
    const res = await fetch("/api/" + path, body === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({ error: "Resposta inválida do servidor." }));
    if (!res.ok || data.error) throw new Error(data.error || "Erro " + res.status);
    return data;
  }

  // ── abas ──
  const tabs = ["chat", "treino", "checkpoints"];
  function showTab(name) {
    if (!tabs.includes(name)) name = "chat";
    tabs.forEach((t) => { $("#tab-" + t).hidden = t !== name; });
    document.querySelectorAll(".dev-tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    history.replaceState(null, "", "#" + name);
    if (name === "treino") drawChart(lastMetrics);
  }
  document.querySelectorAll(".dev-tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

  // ── chat ──
  const log = $("#chat-log"), input = $("#chat-input"), sendBtn = $("#chat-send");
  let turns = [];
  function bubble(cls, html) {
    const el = document.createElement("div");
    el.className = "msg " + cls; el.innerHTML = html;
    log.querySelector(".dev-empty")?.remove();
    log.appendChild(el); log.scrollTop = log.scrollHeight;
    return el;
  }
  let chatGeneration = 0;
  let activeChatController = null;
  async function send() {
    const message = input.value.trim();
    if (!message || sendBtn.disabled) return;
    input.value = ""; sendBtn.disabled = true;
    const generation = ++chatGeneration;
    const controller = new AbortController();
    activeChatController = controller;
    bubble("user", esc(message));
    const pending = bubble("bot pending", "pensando…");
    let streamed = "";
    try {
      const payload = {
        message, history: turns, ckpt: $("#chat-ckpt").value || null,
        temperature: $("#chat-temp").value.trim() === "" ? null : Number($("#chat-temp").value),
        max_tokens: $("#chat-max").value.trim() === "" ? 80 : Number($("#chat-max").value),
        top_k: $("#chat-topk").value.trim() === "" ? null : Number($("#chat-topk").value),
      };
      const res = await fetch("/api/chat/stream", {
        method: "POST", headers: { "Content-Type": "application/json", "Accept": "application/x-ndjson" },
        body: JSON.stringify(payload), signal: controller.signal,
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({ error: "Resposta inválida do servidor." }));
        throw new Error(data.error || "Erro " + res.status);
      }
      if (!res.body) throw new Error("Este navegador não oferece suporte ao streaming de respostas.");
      pending.className = "msg bot";
      pending.textContent = "";
      const reader = res.body.getReader(), decoder = new TextDecoder();
      let buffer = "", finalData = null;
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const lines = buffer.split("\n"); buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);
          if (generation !== chatGeneration) { controller.abort(); return; }
          if (event.type === "delta") {
            streamed += event.text;
            pending.textContent = streamed;
            log.scrollTop = log.scrollHeight;
          } else if (event.type === "error") {
            throw new Error(event.error || "Erro durante a geração.");
          } else if (event.type === "done") finalData = event;
        }
        if (done) break;
      }
      if (buffer.trim()) {
        const event = JSON.parse(buffer);
        if (event.type === "error") throw new Error(event.error || "Erro durante a geração.");
        if (event.type === "done") finalData = event;
      }
      if (generation !== chatGeneration) return;
      if (!finalData) throw new Error("A conexão terminou antes da resposta final.");
      // A versão final limpa marcadores de parada que podem ter aparecido parcialmente no stream.
      pending.textContent = finalData.response;
      const meta = document.createElement("small");
      meta.textContent = `${finalData.checkpoint} · T=${finalData.temperature} · ${finalData.device} · ${finalData.seconds}s`;
      pending.appendChild(meta);
      turns.push([message, finalData.response]);
    } catch (e) {
      if (generation !== chatGeneration || e.name === "AbortError") return;
      pending.className = "msg err";
      pending.textContent = e.message;
    } finally {
      if (generation === chatGeneration) {
        sendBtn.disabled = false;
        if (activeChatController === controller) activeChatController = null;
        input.focus();
      }
    }
  }
  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
  $("#chat-clear").addEventListener("click", () => {
    // Invalidar antes de abortar impede callbacks tardios de restaurarem o histórico.
    chatGeneration += 1;
    if (activeChatController) activeChatController.abort();
    activeChatController = null;
    turns = [];
    sendBtn.disabled = false;
    log.innerHTML = '<p class="dev-empty">Conversa limpa.</p>';
    input.focus();
  });

  // ── treino ──
  const form = $("#train-form"), startBtn = $("#tr-start"), stopBtn = $("#tr-stop"), msg = $("#tr-msg");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(form), body = Object.fromEntries(fd.entries());
    body.resume = form.resume.checked; body.use_existing_data = form.use_existing_data.checked;
    msg.textContent = "";
    try { await api("train/start", body); poll(); } catch (err) { msg.textContent = err.message; }
  });
  stopBtn.addEventListener("click", async () => {
    stopBtn.disabled = true;
    try { await api("train/stop", {}); } catch (err) { msg.textContent = err.message; }
    poll();
  });

  let lastMetrics = [];
  function drawChart(rows) {
    const c = $("#tr-chart"), dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = 220;
    c.width = w * dpr; c.height = h * dpr;
    const g = c.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
    const series = [["train_loss", "#b48cff"], ["validation_loss", "#a6d5ff"]].map(([k, col]) => [
      rows.filter((r) => r[k] !== "" && r[k] != null && r.iteration !== "").map((r) => [+r.iteration, +r[k]]), col,
    ]);
    const pts = series.flatMap(([p]) => p);
    if (!pts.length) { g.fillStyle = "#8b8d93"; g.font = "12px monospace"; g.fillText("Sem avaliações ainda.", 12, 24); return; }
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1), y0 = Math.min(...ys), y1 = Math.max(...ys, y0 + 0.01);
    const px = (x) => 36 + ((x - x0) / (x1 - x0)) * (w - 48), py = (y) => h - 22 - ((y - y0) / (y1 - y0)) * (h - 40);
    g.strokeStyle = "rgba(255,255,255,.08)"; g.fillStyle = "#8b8d93"; g.font = "10px monospace";
    for (let i = 0; i <= 4; i++) { const y = y0 + ((y1 - y0) * i) / 4; g.beginPath(); g.moveTo(36, py(y)); g.lineTo(w - 12, py(y)); g.stroke(); g.fillText(y.toFixed(2), 2, py(y) + 3); }
    g.fillText(String(Math.round(x0)), 36, h - 6); g.fillText(String(Math.round(x1)), w - 44, h - 6);
    series.forEach(([p, col]) => { if (!p.length) return; g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); p.forEach(([x, y], i) => (i ? g.lineTo(px(x), py(y)) : g.moveTo(px(x), py(y)))); g.stroke(); });
  }

  // ── checkpoints ──
  function renderCheckpoints(list) {
    $("#ck-empty").hidden = list.length > 0;
    $("#ck-body").innerHTML = list.map((c) =>
      `<tr><td>${esc(c.name)}</td><td>${c.mb} MB</td><td>${new Date(c.mtime * 1000).toLocaleString("pt-BR")}</td><td><button class="dev-btn ghost" data-use="${esc(c.name)}" type="button">Usar no chat</button></td></tr>`).join("");
    const sel = $("#chat-ckpt"), cur = sel.value;
    sel.innerHTML = '<option value="">automático (melhor/mais recente)</option>' + list.map((c) => `<option value="${esc(c.name)}">${esc(c.name)}</option>`).join("");
    sel.value = list.some((c) => c.name === cur) ? cur : "";
  }
  $("#ck-body").addEventListener("click", (e) => {
    const name = e.target.dataset?.use; if (!name) return;
    $("#chat-ckpt").value = name; showTab("chat");
  });

  // ── status periódico ──
  let timer = null, lastCk = "", fails = 0;
  function setPill(state, text) {
    const pill = $("#pill");
    pill.classList.remove("is-ok", "is-wait", "is-off");
    pill.classList.add("is-" + state);
    $("#pill-text").textContent = text;
  }
  async function poll() {
    clearTimeout(timer);
    let delay = 4000;
    try {
      const s = await api("status"), t = s.training;
      fails = 0;
      setPill("ok", t.running ? "Conectado · treinando" : s.model_loaded ? "Conectado · modelo carregado" : "Conectado");
      startBtn.disabled = t.running; stopBtn.disabled = !t.running;
      $("#tr-progress").textContent = t.running ? (t.progress || "Iniciando…") : t.cmd ? `Treino encerrado (código ${t.returncode ?? "?"}).` : "Nenhum treino em andamento.";
      const logEl = $("#tr-log"), atEnd = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 20;
      logEl.textContent = t.lines.join("\n"); if (atEnd) logEl.scrollTop = logEl.scrollHeight;
      lastMetrics = s.metrics; if (!$("#tab-treino").hidden) drawChart(lastMetrics);
      const sig = JSON.stringify(s.checkpoints);
      if (sig !== lastCk) { lastCk = sig; renderCheckpoints(s.checkpoints); }
      if (t.running) delay = 1500;
    } catch (e) {
      fails += 1;
      setPill(fails >= 2 ? "off" : "wait", fails >= 2 ? "Desconectado" : "Reconectando…");
      delay = 2000;
    }
    timer = setTimeout(poll, delay);
  }

  async function init() {
    try {
      const o = await api("options");
      $("#chat-temp").value = o.temperature;
      $("#tr-dataset").innerHTML = o.dataset_types.length ? o.dataset_types.map((d) => `<option>${esc(d)}</option>`).join("") : "<option value=''>padrão</option>";
      Object.entries(o.lr_presets).forEach(([k, label]) => $("#tr-lr").insertAdjacentHTML("beforeend", `<option value="${esc(k)}">${esc(label)}</option>`));
    } catch (e) { /* o status mostra o erro */ }
    showTab(location.hash.slice(1)); poll();
  }
  window.addEventListener("resize", () => { if (!$("#tab-treino").hidden) drawChart(lastMetrics); });
  init();
})();
