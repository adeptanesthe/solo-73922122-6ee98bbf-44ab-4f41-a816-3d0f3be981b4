/* 技术编辑选题推荐系统前端（原生 JS） */
"use strict";

const $ = (sel) => document.querySelector(sel);
const state = {
  data: null,
  pollTimer: null,
  outline: { rows: [], token: null, timer: null, lastRendered: null },
};
const OUTLINE_MIN = 3;
const OUTLINE_MAX = 6;
const DEFAULT_THRESHOLD = 0.2;

async function api(path, options = {}) {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(body.detail || `请求失败 (${resp.status})`);
  }
  return body;
}

function articleMap() {
  const m = new Map();
  state.data.articles.forEach((a) => m.set(a.id, a));
  return m;
}

function renderQuotas() {
  const quotas = state.data.state.quotas;
  const grid = $("#quota-grid");
  grid.innerHTML = "";
  state.data.topics.forEach((topic) => {
    const wrap = document.createElement("div");
    wrap.className = "quota-item";
    wrap.innerHTML = `<label>${topic}</label>`;
    const input = document.createElement("input");
    input.type = "number";
    input.min = 0;
    input.max = 6;
    input.value = quotas[topic];
    input.dataset.topic = topic;
    input.addEventListener("input", renderQuotaSum);
    wrap.appendChild(input);
    grid.appendChild(wrap);
  });
  renderQuotaSum();
}

function renderQuotaSum() {
  const sum = [...document.querySelectorAll("#quota-grid input")]
    .reduce((acc, el) => acc + (parseInt(el.value, 10) || 0), 0);
  const el = $("#quota-sum");
  el.innerHTML = `当前合计：<b>${sum}</b> / 6`;
  el.className = `quota-sum ${sum === 6 ? "valid" : "invalid"}`;
}

function renderExclusions() {
  const excluded = new Set(state.data.state.excluded_ids);
  const select = $("#exclude-select");
  const revealedIds = new Set(state.data.state.revealed.map((r) => r.article_id));
  select.innerHTML = "";
  state.data.articles.forEach((a) => {
    if (revealedIds.has(a.id)) return;
    const opt = document.createElement("option");
    opt.value = a.id;
    opt.textContent = `#${a.id} ${a.title}（${a.topic}/${a.author}）${excluded.has(a.id) ? " — 已排除" : ""}`;
    if (excluded.has(a.id)) opt.disabled = true;
    select.appendChild(opt);
  });

  const list = $("#exclusion-list");
  list.innerHTML = "";
  if (excluded.size === 0) {
    list.innerHTML = `<span class="msg" style="color:#64748b">暂无排除文章</span>`;
    return;
  }
  const amap = articleMap();
  excluded.forEach((id) => {
    const a = amap.get(id);
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `#${id} ${a ? a.title : "未知"} <button title="取消排除" data-id="${id}">×</button>`;
    chip.querySelector("button").addEventListener("click", () => removeExclusion(id));
    list.appendChild(chip);
  });
}

function renderQueue() {
  const { state: st, plan, job } = state.data;
  const jobEl = $("#job-status");
  const revealedIds = new Set(st.revealed.map((r) => r.article_id));
  const amap = articleMap();

  if (job) {
    const label = {
      pending: "等待计算",
      running: "正在重算（慢重算期间仍可修改条件，迟到结果不会揭示旧推荐）",
      done: "重算完成",
      failed: `重算失败：${job.error || "未知错误"}（进度未推进）`,
      stale: "该结果已过期（条件已变更）",
    }[job.status] || job.status;
    jobEl.className = `job-status ${job.status}`;
    jobEl.textContent = `世代 ${job.generation} · ${label}`;
  }

  const queueEl = $("#queue");
  queueEl.innerHTML = "";
  const activePlan = plan && job.status === "done" && job.generation === st.generation ? plan : null;

  for (let pos = 1; pos <= 6; pos++) {
    const li = document.createElement("li");
    const revealedItem = st.revealed.find((r) => r.position === pos);
    if (revealedItem) {
      const a = amap.get(revealedItem.article_id);
      li.className = "revealed";
      li.innerHTML = `
        <span class="pos">${pos}</span>
        <div class="q-main">
          <div class="q-title">${a ? a.title : "文章 #" + revealedItem.article_id}</div>
          <div class="q-sub">
            <span class="tag">${revealedItem.topic}</span>
            <span class="tag author">${revealedItem.author}</span>
            已揭示 · 固定
          </div>
        </div>
        <div class="q-rel">相关度<br/><b>${revealedItem.relevance.toFixed(4)}</b><br/>权重 ${7 - pos}</div>`;
    } else if (activePlan && activePlan.feasible) {
      const item = activePlan.queue.find((q) => q.position === pos);
      li.className = "hidden-item";
      li.innerHTML = `
        <span class="pos">${pos}</span>
        <div class="q-main">
          <div class="q-title">未揭示位置 ${pos}（候选已锁定，等待揭示）</div>
          <div class="q-sub">主题：${item ? item.topic : "—"}</div>
        </div>
        <div class="q-rel">相关度<br/><b>${item ? item.relevance.toFixed(4) : "—"}</b><br/>权重 ${7 - pos}</div>`;
    } else {
      li.className = "hidden-item";
      const reason = activePlan && !activePlan.feasible ? activePlan.reason : "等待重算结果";
      li.innerHTML = `
        <span class="pos">${pos}</span>
        <div class="q-main"><div class="q-title">未揭示</div><div class="q-sub">${reason}</div></div>
        <div class="q-rel">权重 ${7 - pos}</div>`;
    }
    queueEl.appendChild(li);
  }

  $("#generation-line").textContent =
    `当前世代 ${st.generation} · 已揭示 ${st.revealed.length}/6 篇` +
    (activePlan && activePlan.feasible ? ` · 队列目标分 Σ(7−位置)×相关度 = ${activePlan.score.toFixed(4)}` : "");

  const revealBtn = $("#btn-reveal");
  const blocked = !activePlan || !activePlan.feasible || st.revealed.length >= 6;
  revealBtn.disabled = !!blocked;
  if (st.revealed.length >= 6) revealBtn.textContent = "队列已全部揭示";
  else revealBtn.textContent = "揭示下一组（两篇）";

  renderExclusions();
  renderArticleList();
}

function renderArticleList() {
  const filter = $("#article-filter").value.trim().toLowerCase();
  const excluded = new Set(state.data.state.excluded_ids);
  const revealedIds = new Set(state.data.state.revealed.map((r) => r.article_id));
  const list = $("#article-list");
  list.innerHTML = "";
  state.data.articles
    .filter((a) => {
      if (!filter) return true;
      return `${a.id} ${a.title} ${a.topic} ${a.author}`.toLowerCase().includes(filter);
    })
    .forEach((a) => {
      const card = document.createElement("div");
      card.className = `article-card ${excluded.has(a.id) ? "excluded" : ""}`;
      card.innerHTML = `
        <div class="t">#${a.id} ${a.title}</div>
        <div class="s">
          <span class="tag">${a.topic}</span>
          <span class="tag author">${a.author}</span>
          ${excluded.has(a.id) ? '<span class="tag excluded-tag">已排除</span>' : ""}
          ${revealedIds.has(a.id) ? '<span class="tag" style="background:#dcfce7;color:#166534">已揭示</span>' : ""}
        </div>`;
      card.addEventListener("click", () => openArticle(a.id));
      list.appendChild(card);
    });
}

async function openArticle(id) {
  const a = await api(`/api/articles/${id}`);
  $("#dlg-title").textContent = `#${a.id} ${a.title}`;
  $("#dlg-meta").innerHTML = `<span class="tag">${a.topic}</span><span class="tag author">${a.author}</span>`;
  $("#dlg-body").textContent = a.body;
  $("#article-dialog").showModal();
}

function readQuotas() {
  const quotas = {};
  document.querySelectorAll("#quota-grid input").forEach((el) => {
    quotas[el.dataset.topic] = parseInt(el.value, 10) || 0;
  });
  return quotas;
}

function showMsg(el, text, ok = false) {
  el.textContent = text;
  el.className = `msg ${ok ? "ok" : ""}`;
  if (text) setTimeout(() => { el.textContent = ""; el.className = "msg"; }, 5000);
}

async function refresh() {
  state.data = await api("/api/status");
  $("#interest").value = state.data.state.interest;
  renderQuotas();
  renderQueue();
  schedulePoll();
  // 首次加载时恢复最近一次核查结果；页面内提交后的状态由令牌逻辑自行管理
  if (state.outline.token === null && state.data.outline_job) {
    const job = state.data.outline_job;
    state.outline.token = job.check_id;
    state.outline.lastRendered = job.check_id;
    renderOutlineResults(job);
    if (job.status === "pending" || job.status === "running") {
      scheduleOutlinePoll();
    }
  }
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  const j = state.data.job;
  if (j && (j.status === "pending" || j.status === "running")) {
    state.pollTimer = setTimeout(refresh, 600);
  }
}

async function applyCondition() {
  const msg = $("#condition-msg");
  try {
    const payload = {
      interest: $("#interest").value,
      quotas: readQuotas(),
      compute_delay_ms: parseInt($("#delay").value, 10) || 0,
    };
    $("#btn-apply").disabled = true;
    state.data = await api("/api/condition", { method: "PUT", body: JSON.stringify(payload) });
    renderQuotas();
    renderQueue();
    showMsg(msg, "条件已更新，新世代重算已开始", true);
    schedulePoll();
  } catch (err) {
    showMsg(msg, err.message);
  } finally {
    $("#btn-apply").disabled = false;
  }
}

async function revealNext() {
  const msg = $("#reveal-msg");
  try {
    const data = await api("/api/reveal", { method: "POST" });
    state.data = data;
    renderQueue();
    const outcome = data.reveal_outcome;
    if (outcome.advanced) {
      showMsg(msg, `已揭示位置 ${outcome.new_positions.join("、")}`, true);
    } else {
      showMsg(msg, outcome.reason || "没有推进");
    }
  } catch (err) {
    showMsg(msg, err.message);
  }
}

async function addExclusion() {
  const msg = $("#condition-msg");
  const id = parseInt($("#exclude-select").value, 10);
  if (!id) return;
  try {
    state.data = await api("/api/exclusions", {
      method: "POST",
      body: JSON.stringify({ article_id: id, compute_delay_ms: parseInt($("#delay").value, 10) || 0 }),
    });
    renderQueue();
    showMsg(msg, `已排除文章 #${id}，触发新世代重算`, true);
    schedulePoll();
  } catch (err) {
    showMsg(msg, err.message);
  }
}

async function removeExclusion(id) {
  const msg = $("#condition-msg");
  try {
    const delay = parseInt($("#delay").value, 10) || 0;
    state.data = await api(`/api/exclusions/${id}?compute_delay_ms=${delay}`, {
      method: "DELETE",
    });
    renderQueue();
    showMsg(msg, `已取消排除 #${id}`, true);
    schedulePoll();
  } catch (err) {
    showMsg(msg, err.message);
  }
}

async function resetDemo() {
  if (!confirm("确定重置全部演示数据？已揭示前缀将被清空。")) return;
  state.data = await api("/api/reset", { method: "POST" });
  renderQuotas();
  renderQueue();
  schedulePoll();
  // 重置同时清空提纲核查的本地状态，旧核查结果不再展示
  state.outline.token = null;
  state.outline.lastRendered = null;
  clearTimeout(state.outline.timer);
  renderOutlineResults(null);
  $("#outline-status").style.display = "none";
  $("#outline-msg").textContent = "";
}

/* ---------------- 提纲素材核查 ---------------- */

function renderOutlinePoints() {
  const wrap = $("#outline-points");
  wrap.innerHTML = "";
  state.outline.rows.forEach((row, i) => {
    const div = document.createElement("div");
    div.className = "outline-point";
    div.innerHTML = `
      <span class="idx">${i + 1}</span>
      <textarea rows="2" placeholder="提纲点 ${i + 1} 的短句……"></textarea>
      <div class="threshold-box">
        <label>相似度下限（0.05~0.95）</label>
        <input type="number" min="0.05" max="0.95" step="0.01" />
      </div>
      <button class="btn-remove" title="删除该提纲点">×</button>`;
    const [textarea, thresholdInput, removeBtn] =
      [div.querySelector("textarea"), div.querySelector("input[type=number]"), div.querySelector(".btn-remove")];
    textarea.value = row.text;
    thresholdInput.value = row.threshold;
    textarea.addEventListener("input", () => { state.outline.rows[i].text = textarea.value; });
    thresholdInput.addEventListener("input", () => {
      state.outline.rows[i].threshold = thresholdInput.value;
    });
    removeBtn.disabled = state.outline.rows.length <= OUTLINE_MIN;
    removeBtn.addEventListener("click", () => {
      if (state.outline.rows.length <= OUTLINE_MIN) return;
      state.outline.rows.splice(i, 1);
      renderOutlinePoints();
    });
    wrap.appendChild(div);
  });
  $("#btn-outline-add").disabled = state.outline.rows.length >= OUTLINE_MAX;
}

function addOutlinePoint() {
  if (state.outline.rows.length >= OUTLINE_MAX) return;
  state.outline.rows.push({ text: "", threshold: DEFAULT_THRESHOLD });
  renderOutlinePoints();
}

function readOutlinePoints() {
  const points = [];
  document.querySelectorAll("#outline-points .outline-point").forEach((el, i) => {
    const text = el.querySelector("textarea").value.trim();
    const raw = el.querySelector("input[type=number]").value;
    const threshold = parseFloat(raw);
    if (!text) throw new Error(`第 ${i + 1} 个提纲点短句不能为空`);
    if (text.length > 500) throw new Error(`第 ${i + 1} 个提纲点过长（上限 500 字）`);
    if (!(threshold >= 0.05 && threshold <= 0.95)) {
      throw new Error(`第 ${i + 1} 个提纲点的相似度下限必须在 0.05~0.95 之间`);
    }
    points.push({ text, threshold });
  });
  if (points.length < OUTLINE_MIN || points.length > OUTLINE_MAX) {
    throw new Error("提纲点数量必须在 3~6 个之间");
  }
  return points;
}

function renderOutlineResults(job) {
  const box = $("#outline-results");
  const statusEl = $("#outline-status");
  box.innerHTML = "";
  if (!job) {
    statusEl.style.display = "none";
    return;
  }
  const labelMap = {
    pending: "等待核查",
    running: "正在核查（慢核查期间可修改提纲重新提交，迟到结果不会覆盖新核查）",
    done: "核查完成",
    failed: `核查失败：${job.error || "未知错误"}`,
    stale: "该核查已过期（提纲已更新，结果作废）",
  };
  statusEl.style.display = "block";
  statusEl.className = `job-status ${job.status === "failed" ? "failed" : job.status === "done" ? "done" : "pending"}`;
  statusEl.textContent = `核查 #${job.check_id} · ${labelMap[job.status] || job.status}`;

  if (job.status !== "done" || !job.result) {
    if (job.status === "stale") {
      box.innerHTML = `<div class="msg">本次核查已被更新的提纲取代，未展示任何旧结果。</div>`;
    }
    return;
  }

  const result = job.result;
  result.results.forEach((r) => {
    const card = document.createElement("div");
    card.className = `outline-card ${r.assigned ? "assigned" : "unassigned"}`;
    const badge = r.assigned
      ? `<span class="oc-badge ok">已找到来源</span>`
      : `<span class="oc-badge todo">待补材料</span>`;
    let inner = `
      <div class="oc-head">
        <div class="oc-point">${r.index + 1}. ${escapeHtml(r.text)}</div>
        ${badge}
      </div>
      <div class="oc-meta">
        <span class="tag">下限 ${Number(r.threshold).toFixed(2)}</span>
      </div>`;
    if (r.assigned) {
      inner += `
        <div class="oc-meta">
          <span class="tag">${escapeHtml(r.topic)}</span>
          <span class="tag author">${escapeHtml(r.author)}</span>
          <span>#${r.article_id} ${escapeHtml(r.article_title)}</span>
        </div>
        <div class="oc-sentence">${escapeHtml(r.sentence)}</div>
        <div class="oc-score">句子相似度：<b>${Number(r.score).toFixed(4)}</b></div>`;
    } else {
      inner += `<div class="oc-hint">没有文章句子达到该点下限，需补充稿件材料。</div>`;
      if (r.best_unsatisfied) {
        const b = r.best_unsatisfied;
        inner += `
          <div class="oc-hint">
            全库最接近：#${b.article_id}《${escapeHtml(b.article_title)}》
            （相似度 ${Number(b.score).toFixed(4)}，低于下限 ${Number(r.threshold).toFixed(2)}）<br/>
            原句：${escapeHtml(b.sentence)}
          </div>`;
      }
    }
    card.innerHTML = inner;
    box.appendChild(card);
  });

  const summary = document.createElement("div");
  summary.className = "outline-summary";
  summary.innerHTML =
    `达标点 <b>${result.matched_count}</b> / ${result.point_count} 个 · ` +
    `所选句子总相似度 <b>${Number(result.total_score).toFixed(4)}</b> · ` +
    `每篇文章至多分配给一个点`;
  box.appendChild(summary);
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function scheduleOutlinePoll() {
  clearTimeout(state.outline.timer);
  const tick = async () => {
    const token = state.outline.token;
    if (token === null) return;
    try {
      const job = await api(`/api/outline-checks/${token}`);
      // 编辑已提交更新的核查：迟到的旧响应直接丢弃，绝不覆盖新结果
      if (state.outline.token !== token) return;
      if (job.status === "pending" || job.status === "running") {
        state.outline.timer = setTimeout(tick, 600);
        return;
      }
      renderOutlineResults(job);
      state.outline.lastRendered = token;
    } catch (err) {
      showMsg($("#outline-msg"), err.message);
    }
  };
  state.outline.timer = setTimeout(tick, 500);
}

async function submitOutlineCheck() {
  const msg = $("#outline-msg");
  let points;
  try {
    points = readOutlinePoints();
  } catch (err) {
    showMsg(msg, err.message);
    return;
  }
  try {
    $("#btn-outline-check").disabled = true;
    const payload = {
      points,
      compute_delay_ms: parseInt($("#outline-delay").value, 10) || 0,
    };
    const data = await api("/api/outline-checks", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const job = data.outline_job;
    state.outline.token = job.check_id;  // 新核查令牌：旧令牌的响应之后一律忽略
    state.outline.lastRendered = null;
    // 同步输入行的标准值
    state.outline.rows = points.map((p) => ({ text: p.text, threshold: p.threshold }));
    renderOutlinePoints();
    renderOutlineResults(job);
    showMsg(msg, "核查已提交，后台正在比对文章句子", true);
    scheduleOutlinePoll();
  } catch (err) {
    showMsg(msg, err.message);
  } finally {
    $("#btn-outline-check").disabled = false;
  }
}

$("#btn-apply").addEventListener("click", applyCondition);
$("#btn-reveal").addEventListener("click", revealNext);
$("#btn-exclude").addEventListener("click", addExclusion);
$("#btn-reset").addEventListener("click", resetDemo);
$("#article-filter").addEventListener("input", renderArticleList);
$("#dlg-close").addEventListener("click", () => $("#article-dialog").close());
$("#btn-outline-add").addEventListener("click", addOutlinePoint);
$("#btn-outline-check").addEventListener("click", submitOutlineCheck);

// 初始 3 个提纲点（可增到 6 个、删到 3 个）
state.outline.rows = [
  { text: "", threshold: DEFAULT_THRESHOLD },
  { text: "", threshold: DEFAULT_THRESHOLD },
  { text: "", threshold: DEFAULT_THRESHOLD },
];
renderOutlinePoints();

refresh().catch((err) => {
  $("#job-status").textContent = `加载失败：${err.message}`;
});
