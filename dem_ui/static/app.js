// Operator screen: Setup (edit a scenario, launch), Run (live progress), Results (playback,
// post-run measurements).  All state lives on the server; this file only renders it.
import { Viewer, colormapCss } from "/static/viewer.js";
import { lineChart, fmt, PALETTE } from "/static/chart.js";

const $ = s => document.querySelector(s);
const INF = 1e30;
const api = async (path, opts = {}) => {
  const r = await fetch(path, { headers: { "content-type": "application/json" }, ...opts });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.arrayBuffer();
};

// Errors go on the page, not only to the console -- as a dismissible note in the corner
// that says what failed, not a stack line over the legend.
function showError(msg) {
  let e = document.getElementById("viewer-error");
  if (!e) {
    e = document.createElement("div"); e.id = "viewer-error";
    e.innerHTML = '<button class="small" title="dismiss">×</button><span></span>';
    e.querySelector("button").onclick = () => e.hidden = true;
    $("#stage").appendChild(e);
  }
  e.querySelector("span").textContent = msg; e.hidden = false;
  clearTimeout(showError.t); showError.t = setTimeout(() => e.hidden = true, 15000);
}
window.addEventListener("error", ev => showError(ev.message));
window.addEventListener("unhandledrejection", ev => showError(ev.reason?.message || String(ev.reason)));
let viewer;
try { viewer = new Viewer($("#viewer")); }
catch (e) {
  showError("3D view unavailable (WebGL could not start): " + e.message);
  viewer = new Proxy({}, { get: () => () => {} });     // keep the rest of the UI working
}
window.__viewer = viewer;                 // for debugging and automated UI tests
const num = v => (typeof v === "number" && isFinite(v)) ? +v.toPrecision(6) : v;
const S = { tab: "setup", scenPath: null, scen: null, geoKey: null, run: null, runInfo: null,
            frames: [], frameIdx: 0, playing: false, analysis: null, poll: null };

// ---------------------------------------------------------------- tabs
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => showTab(b.dataset.tab));
async function showTab(t) {
  S.tab = t;
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === t));
  for (const x of ["setup", "run", "results"]) $(`#tab-${x}`).hidden = x !== t;
  $("#frame-bar").hidden = t !== "results" || !S.frames.length;
  if (t === "setup" && S.scen) await loadGeometry("scenario", S.scenPath);
  if ((t === "run" || t === "results") && S.run && S.runInfo?.meta?.parts) await loadGeometry("run", S.run);
  if (t !== "results") { viewer.showParticles(false); viewer.clearWallMap(); $("#legend").hidden = true; }
  else refreshColouring();
}

// ---------------------------------------------------------------- sidebar
async function loadProjects() {
  const ps = await api("/api/projects");
  const el = $("#projects"); el.innerHTML = "";
  const nb = document.createElement("button"); nb.className = "small"; nb.textContent = "+ new project";
  nb.onclick = async () => {
    const name = prompt("Project name (letters, digits, _ . -)");
    if (!name) return;
    try { const r = await api("/api/projects", { method: "POST", body: JSON.stringify({ name }) });
          await loadProjects(); await openScenario(r.scenario); }
    catch (e) { showError("New project: " + e.message); }
  };
  el.appendChild(nb);
  for (const p of ps) {
    const d = document.createElement("div"); d.className = "proj";
    d.innerHTML = `<div class="name">${p.name}</div>`;
    for (const s of p.scenarios) {
      const it = document.createElement("div");
      it.className = "item" + (s === S.scenPath ? " sel" : "");
      it.innerHTML = `${s.split("/").pop()}`;
      it.onclick = () => openScenario(s);
      d.appendChild(it);
    }
    if (p.bfa) {
      const b = document.createElement("button"); b.className = "small";
      b.textContent = p.scenarios.length ? "re-import BFA" : "import BFA project";
      b.onclick = () => importProject(p.name, b);
      d.appendChild(b);
    }
    el.appendChild(d);
  }
}

async function importProject(name, btn) {
  btn.disabled = true; btn.textContent = "importing…";
  try {
    const r = await api(`/api/projects/import?project=${encodeURIComponent(name)}&preset=fast`, { method: "POST" });
    await loadProjects(); await openScenario(r.scenario);
  } catch (e) { alert("Import failed: " + e.message); }
  btn.disabled = false;
}

async function loadRuns() {
  const rs = await api("/api/runs");
  const el = $("#runs"); el.innerHTML = "";
  for (const r of rs.filter(r => r.scenario_based).slice(0, 40)) {
    const it = document.createElement("div");
    it.className = "item" + (r.id === S.run ? " sel" : "");
    const prog = r.duration ? ` ${fmt(r.t)}/${fmt(r.duration)} s` : "";
    it.innerHTML = `<div>${r.id}</div><div class="meta"><span class="badge ${r.status}">${r.status}</span>${prog} · ${r.frames} frames</div>`;
    it.onclick = () => openRun(r.id, r.status === "running" ? "run" : "results");
    el.appendChild(it);
  }
}
$("#refresh-runs").onclick = loadRuns;

// ---------------------------------------------------------------- geometry
async function loadGeometry(kind, key) {
  const k = kind + ":" + key;
  if (S.geoKey === k) return;
  S.geoKey = k;
  const geo = await api(kind === "scenario" ? `/api/geometry?scenario=${encodeURIComponent(key)}`
                                             : `/api/runs/${encodeURIComponent(key)}/geometry`);
  const r = kind === "scenario" ? S.scen?.material?.radius : S.runInfo?.meta?.material?.radius;
  viewer.setGeometry(geo, r);
  S.geo = geo;
}

// ---------------------------------------------------------------- setup
const MATERIAL_FIELDS = [
  ["name", "text"], ["radius", "number", "grain radius (m)"], ["density", "number", "density (kg/m³)"],
  ["youngs", "number", "Young's modulus (Pa)"], ["poisson", "number", "Poisson ratio"],
  ["restitution", "number", "restitution"], ["friction", "number", "grain-grain sliding μ"],
  ["rolling_friction", "number", "grain-grain rolling μ"], ["wall_rolling_friction", "number", "wall rolling μ"],
  ["contact", "select:hertz,linear", "contact law"], ["tangential_ratio", "number", "Mindlin kt/kn"],
];

async function openScenario(path) {
  S.scenPath = path; S.scen = await api(`/api/scenario?path=${encodeURIComponent(path)}`);
  S.geoKey = null;
  await showTab("setup");
  renderSetup(); loadProjects();
}

function get(obj, path) { return path.split(".").reduce((o, k) => o?.[k], obj); }
function set(obj, path, v) { const ks = path.split("."); ks.slice(0, -1).reduce((o, k) => o[k] ??= {}, obj)[ks.at(-1)] = v; }

function renderSetup() {
  const sc = S.scen;
  $("#setup-empty").hidden = true; $("#setup-form").hidden = false;
  $("#setup-title").textContent = `${sc.name}  ·  ${S.scenPath}`;
  $("#run-duration").value = sc.output?.duration ?? 10;
  document.querySelectorAll("[data-bind]").forEach(inp => {
    const v = get(sc, inp.dataset.bind);
    if (inp.type === "checkbox") inp.checked = !!v; else inp.value = num(v) ?? "";
    inp.onchange = () => set(sc, inp.dataset.bind, inp.type === "checkbox" ? inp.checked : +inp.value);
  });
  const s = sc.solver || {};
  $("#preset").value = (s.youngs_divisor ?? 10) === 10 && (s.neighbor_every ?? 4) === 4 ? "fast"
    : (s.youngs_divisor === 1 && !s.neighbor_every) ? "reference" : "custom";
  $("#preset").onchange = e => {
    sc.solver = sc.solver || {};
    if (e.target.value === "fast") Object.assign(sc.solver, { youngs_divisor: 10, dt: "auto", neighbor_every: 4, skin_speed: 6 });
    if (e.target.value === "reference") Object.assign(sc.solver, { youngs_divisor: 1, neighbor_every: 0, dt: sc.notes?.bfa_timestep || "auto" });
  };
  // material
  const mf = $("#material-fields"); mf.innerHTML = "";
  for (const [k, type, label] of MATERIAL_FIELDS) {
    const l = document.createElement("label"); l.textContent = label || k;
    let inp;
    if (type.startsWith("select:")) {
      inp = document.createElement("select");
      for (const o of type.slice(7).split(",")) inp.add(new Option(o, o));
    } else { inp = document.createElement("input"); inp.type = type; if (type === "number") inp.step = "any"; }
    inp.value = num(sc.material[k]);
    inp.onchange = () => sc.material[k] = type === "number" ? +inp.value : inp.value;
    l.appendChild(inp); mf.appendChild(l);
  }
  // parts
  const tb = $("#parts-table tbody"); tb.innerHTML = "";
  sc.parts.forEach((p, k) => {
    const tr = document.createElement("tr");
    const kind = p.motion?.type || "wall";
    tr.innerHTML = `<td><input type="checkbox" checked></td>
      <td><span class="swatch" style="background:${PALETTE[k % PALETTE.length]}"></span>${p.name}</td>
      <td><select><option value="wall">wall</option><option value="belt">conveyor belt</option><option value="rotating">rotating</option></select></td>
      <td><input type="checkbox" ${p.two_sided ? "checked" : ""}></td>
      <td><input type="number" step="any" value="${p.friction}"></td>
      <td><input type="number" step="any" value="${p.active[0]}"></td>
      <td><input type="number" step="any" value="${p.active[1] >= INF ? "" : p.active[1]}" placeholder="end"></td>
      <td><input type="checkbox" ${p.corners ? "checked" : ""}></td>
      <td><button class="small" title="remove part">×</button></td>`;
    const [show, two, fric, on, off, corn] = tr.querySelectorAll("input");
    const type = tr.querySelector("select"), del = tr.querySelector("button");
    type.value = kind;
    show.onchange = () => viewer.setPartVisible(p.name, show.checked);
    two.onchange = () => p.two_sided = two.checked;
    fric.onchange = () => p.friction = +fric.value;
    on.onchange = () => p.active[0] = +on.value;
    off.onchange = () => p.active[1] = off.value === "" ? INF : +off.value;
    corn.onchange = () => p.corners = corn.checked;
    type.onchange = () => {
      if (type.value === "wall") delete p.motion;
      if (type.value === "belt") { p.motion = { type: "belt", velocity: [1, 0, 0] }; p.corners = true; }
      if (type.value === "rotating") p.motion = { type: "rotating", point: viewer.partCentroid(p.name), axis: [0, 0, 1], rpm: 30 };
      renderSetup();
    };
    del.onclick = () => { if (confirm(`Remove part ${p.name} from the scenario? (the STL file stays)`)) { sc.parts.splice(k, 1); renderSetup(); } };
    tb.appendChild(tr);
    if (p.motion) tb.appendChild(motionRow(p));
  });
  viewer.setMotionArrows(sc.parts);
  // injector
  renderInjector(sc);
  // flow planes
  $("#flow-planes").innerHTML = (sc.flow_planes || []).map(f =>
    `<div class="item">${f.name}: ${"xyz"[f.axis]} = ${f.value}</div>`).join("") || "<div class='msg'>none</div>";
  // notes
  const n = sc.notes || {};
  $("#notes").innerHTML = (n.interpretations || []).map(s => `<p>• ${s}</p>`).join("") +
    (n.warnings || []).map(s => `<p class="warn">⚠ ${s}</p>`).join("");
}

// ---------------------------------------------------------------- injection
// Source: an STL face, or a box placed relative to a belt or a horizontal plane (BFA's
// "injection volume from belt reference").  A box can pass at most
// width x height x packing x density x speed; the lattice packing is (4/3 pi r^3)/(2.2 r)^3.
const LATTICE_PACKING = (4 / 3) * Math.PI / Math.pow(2.2, 3);

async function renderInjector(sc) {
  const inf = $("#injector-fields"); inf.innerHTML = "";
  let inj = sc.injectors[0];
  const src = document.createElement("label"); src.textContent = "source";
  const sel = document.createElement("select");
  for (const [v, t] of [["none", "— none —"], ["face", "STL face"], ["belt", "box on belt"], ["plane", "box on plane"]]) sel.add(new Option(t, v));
  sel.value = !inj ? "none" : inj.box ? inj.box.reference : "face";
  src.appendChild(sel); inf.appendChild(src);
  sel.onchange = () => {
    const base = inj || { name: "injection", face_stl: "", mass_rate: 10, velocity: [0, -1, 0] };
    const belts = sc.parts.filter(p => p.motion?.type === "belt");
    if (sel.value === "none") sc.injectors = [];
    if (sel.value === "face") { delete base.box; sc.injectors = [base]; }
    if (sel.value === "belt") {
      if (!belts.length) { showError("Make a part a conveyor belt first (Parts: type)."); sel.value = inj?.box?.reference || (inj ? "face" : "none"); return; }
      base.face_stl = ""; base.box = { reference: "belt", part: belts[0].name, position: 0.5, lateral: 0, clearance: 0.05, length: 2, width: 1, height: 0.5, match_belt: true };
      sc.injectors = [base];
    }
    if (sel.value === "plane") {
      base.face_stl = ""; base.box = { reference: "plane", plane_height: 0, center: [0, 0, 0], direction: [1, 0, 0], clearance: 0, length: 1, width: 1, height: 0.5 };
      sc.injectors = [base];
    }
    renderInjector(sc); refreshInjectorPreview();
  };
  if (!inj) { const d = document.createElement("div"); d.className = "msg"; d.textContent = "No injection yet: choose a source, or upload an STL as “injection face”."; inf.appendChild(d); return; }
  const field = (label, get_, set_, opts = {}) => {
    const l = document.createElement("label"); l.textContent = label;
    const i = document.createElement("input"); i.type = "number"; i.step = "any"; i.value = get_();
    i.onchange = () => { set_(+i.value); updateCapacity(); if (opts.preview) refreshInjectorPreview(); };
    l.appendChild(i); inf.appendChild(l); return i;
  };
  // one quantity, two units: typing in either updates the other at once
  const kgs = field("mass rate (kg/s)", () => +inj.mass_rate.toFixed(4), v => inj.mass_rate = v);
  const tph = field("mass rate (t/h)", () => +(inj.mass_rate * 3.6).toFixed(2), v => inj.mass_rate = v / 3.6);
  kgs.oninput = () => { inj.mass_rate = +kgs.value; tph.value = +(inj.mass_rate * 3.6).toFixed(2); updateCapacity(); };
  tph.oninput = () => { inj.mass_rate = +tph.value / 3.6; kgs.value = +inj.mass_rate.toFixed(4); updateCapacity(); };
  const bx = inj.box;
  if (!bx) {
    const l = document.createElement("label"); l.textContent = "face STL";
    const fs = document.createElement("select");
    const files = await api(`/api/stl-files?scenario=${encodeURIComponent(S.scenPath)}`);
    fs.add(new Option("— choose —", ""));
    for (const f of files) fs.add(new Option(f, f));
    fs.value = inj.face_stl || "";
    fs.onchange = () => { inj.face_stl = fs.value; refreshInjectorPreview(); };
    l.appendChild(fs); inf.appendChild(l);
  } else if (bx.reference === "belt") {
    const l = document.createElement("label"); l.textContent = "belt";
    const bs = document.createElement("select");
    for (const p of sc.parts.filter(p => p.motion?.type === "belt")) bs.add(new Option(p.name, p.name));
    bs.value = bx.part; bs.onchange = () => { bx.part = bs.value; refreshInjectorPreview(); updateCapacity(); };
    l.appendChild(bs); inf.appendChild(l);
    field("position from belt tail (m)", () => bx.position, v => bx.position = v, { preview: true });
    field("lateral offset (m)", () => bx.lateral, v => bx.lateral = v, { preview: true });
    field("clearance above belt (m)", () => bx.clearance, v => bx.clearance = v, { preview: true });
  } else {
    field("plane height (m)", () => bx.plane_height, v => bx.plane_height = v, { preview: true });
    field("centre x (m)", () => bx.center[0], v => bx.center[0] = v, { preview: true });
    field("centre z (m)", () => bx.center[2], v => bx.center[2] = v, { preview: true });
    const l = document.createElement("label"); l.textContent = "length along";
    const ds = document.createElement("select");
    for (const [v, t] of [["1,0,0", "+x"], ["-1,0,0", "−x"], ["0,0,1", "+z"], ["0,0,-1", "−z"]]) ds.add(new Option(t, v));
    ds.value = bx.direction.join(","); ds.onchange = () => { bx.direction = ds.value.split(",").map(Number); refreshInjectorPreview(); };
    l.appendChild(ds); inf.appendChild(l);
    field("clearance above plane (m)", () => bx.clearance, v => bx.clearance = v, { preview: true });
  }
  if (bx) {
    field("box length (m)", () => bx.length, v => bx.length = v, { preview: true });
    field("box width (m)", () => bx.width, v => bx.width = v, { preview: true });
    field("box height (m)", () => bx.height, v => bx.height = v, { preview: true });
  }
  if (bx?.reference === "belt") {
    const l = document.createElement("label"); l.textContent = "move at belt velocity";
    const c = document.createElement("input"); c.type = "checkbox"; c.checked = bx.match_belt !== false;
    c.onchange = () => { bx.match_belt = c.checked; renderInjector(sc); };
    l.appendChild(c); inf.appendChild(l);
  }
  if (!(bx?.reference === "belt" && bx.match_belt !== false))
    ["x", "y", "z"].forEach((ax, k) => field(`velocity ${ax} (m/s)`, () => inj.velocity[k], v => inj.velocity[k] = v));
  field("start (s)", () => inj.start ?? 0, v => inj.start = v);
  field("stop (s)", () => ((inj.stop ?? INF) >= INF ? "" : inj.stop), v => inj.stop = v || INF);
  const cap = document.createElement("div"); cap.id = "inj-capacity"; cap.className = "msg"; cap.style.gridColumn = "1 / -1";
  inf.appendChild(cap);
  updateCapacity();
}

function updateCapacity() {
  const el = $("#inj-capacity"), sc = S.scen, inj = sc?.injectors?.[0];
  if (!el || !inj?.box) { if (el) el.textContent = ""; return; }
  const bx = inj.box;
  let speed = Math.hypot(...inj.velocity);
  if (bx.reference === "belt" && bx.match_belt !== false) {
    const p = sc.parts.find(q => q.name === bx.part);
    speed = p?.motion ? Math.hypot(...p.motion.velocity) : 0;
  }
  const cap = bx.width * bx.height * LATTICE_PACKING * sc.material.density * speed;
  const ok = cap >= inj.mass_rate * 1.1;
  el.className = "msg " + (ok ? "ok" : "err");
  el.textContent = `box can pass ≈ ${fmt(cap)} kg/s (${fmt(cap * 3.6)} t/h) at ${fmt(speed)} m/s — ` +
    (ok ? "enough for the mass rate" : "LESS than the mass rate: enlarge width × height (or speed), or the inlet will choke");
}

async function refreshInjectorPreview() {
  // save-free preview: geometry is recomputed from the current (unsaved) scenario
  clearTimeout(refreshInjectorPreview.t);
  refreshInjectorPreview.t = setTimeout(async () => {
    try {
      const geo = await api(`/api/geometry-preview?scenario=${encodeURIComponent(S.scenPath)}`, { method: "POST", body: JSON.stringify(S.scen) });
      const err = geo.injectors?.find(i => i.error);
      if (err) showError("Injection: " + err.error);
      viewer.setGeometry(geo, S.scen.material.radius); viewer.setMotionArrows(S.scen.parts);
    } catch (e) { showError(e.message); }
  }, 300);
}

// The editor row under a moving part: belt speed and running direction (picked on an edge
// in the 3D view, as in BFA), or rotation speed and axis.
// A picked edge gives a LINE, not a sense: keep the current running direction's sense
// (flip reverses it deliberately), and snap to a coordinate axis within 5 deg, which is
// nearly always what is meant; genuinely inclined edges (inclined conveyors) are kept.
function orientPicked(dir, prev) {
  let d = dir.slice();
  if (prev && d[0] * prev[0] + d[1] * prev[1] + d[2] * prev[2] < 0) d = d.map(c => -c);
  const k = d.map(Math.abs).indexOf(Math.max(...d.map(Math.abs)));
  if (Math.abs(d[k]) > Math.cos(5 * Math.PI / 180)) d = [0, 0, 0].map((_, j) => j === k ? Math.sign(d[k]) : 0);
  return d;
}

function motionRow(p) {
  const tr = document.createElement("tr"); tr.className = "motion-row";
  const td = document.createElement("td"); td.colSpan = 9; tr.appendChild(td);
  const m = p.motion;
  const dirText = v => { const n = Math.hypot(...v) || 1; return v.map(c => (c / n).toFixed(2)).join(", "); };
  if (m.type === "belt") {
    const speed = Math.hypot(...m.velocity);
    td.innerHTML = `↳ belt speed <input type="number" step="any" value="${+speed.toFixed(4)}"> m/s
      · direction (${dirText(m.velocity)})
      <button class="small">pick edge</button> <button class="small">flip</button>
      <span class="msg">direction of the CARRYING side; pulley wrap and return follow it</span>`;
    const [spIn] = td.querySelectorAll("input");
    const [pick, flip] = td.querySelectorAll("button");
    spIn.onchange = () => { const n = Math.hypot(...m.velocity) || 1; m.velocity = m.velocity.map(c => c / n * +spIn.value); renderSetup(); };
    flip.onclick = () => { m.velocity = m.velocity.map(c => -c); renderSetup(); };
    pick.onclick = async () => {
      pick.textContent = "click an edge on the part… (Esc cancels)";
      const r = await viewer.pickEdge(p.name);
      if (r) { const s = Math.hypot(...m.velocity) || 1; m.velocity = orientPicked(r.dir, m.velocity).map(c => c * s); }
      renderSetup();
    };
  } else {
    const rpm = m.rpm ?? (m.omega || 0) * 60 / (2 * Math.PI);
    td.innerHTML = `↳ rotation <input type="number" step="any" value="${+rpm.toFixed(3)}"> rpm
      · axis (${dirText(m.axis)}) through (${m.point.map(c => (+c).toFixed(3)).join(", ")})
      <button class="small">pick axis edge</button> <button class="small">flip</button>`;
    const [rIn] = td.querySelectorAll("input");
    const [pick, flip] = td.querySelectorAll("button");
    rIn.onchange = () => { m.rpm = +rIn.value; delete m.omega; renderSetup(); };
    flip.onclick = () => { m.axis = m.axis.map(c => -c); renderSetup(); };
    pick.onclick = async () => {
      pick.textContent = "click an edge along the axis… (Esc cancels)";
      const r = await viewer.pickEdge(p.name);
      if (r) m.axis = orientPicked(r.dir, m.axis);
      renderSetup();
    };
  }
  return tr;
}

async function uploadStl() {
  const files = $("#upload-files").files;
  const msg = $("#setup-msg");
  if (!files.length) { msg.className = "msg err"; msg.textContent = "choose STL file(s) first"; return; }
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  fd.append("units", $("#upload-units").value);
  fd.append("role", $("#upload-role").value);
  fd.append("scenario", S.scenPath);
  msg.className = "msg"; msg.textContent = "uploading…";
  // save unsaved edits first (e.g. a removed part): the server adds to the file on disk
  await api(`/api/scenario?path=${encodeURIComponent(S.scenPath)}`, { method: "PUT", body: JSON.stringify(S.scen) });
  const r = await fetch(`/api/projects/upload`, { method: "POST", body: fd });
  const j = await r.json();
  if (!r.ok) { msg.className = "msg err"; msg.textContent = j.detail || r.statusText; return; }
  S.scen = j.scenario; S.geoKey = null;
  await loadGeometry("scenario", S.scenPath); renderSetup();
  msg.className = "msg ok";
  msg.textContent = "added " + j.added.map(x => `${x.file} (${x.triangles} tris, ${x.size_m.join(" × ")} m)`).join(", ") +
    (j.problems.length ? ` — still needed: ${j.problems.join("; ")}` : "");
  $("#upload-files").value = "";
}

$("#save-scenario").onclick = async () => {
  const m = $("#setup-msg");
  try { const r = await api(`/api/scenario?path=${encodeURIComponent(S.scenPath)}`, { method: "PUT", body: JSON.stringify(S.scen) });
        m.className = r.problems.length ? "msg err" : "msg ok";
        m.textContent = r.problems.length ? "saved as draft — before it can run: " + r.problems.join("; ") : "saved";
        S.geoKey = null; await loadGeometry("scenario", S.scenPath); renderSetup(); }
  catch (e) { m.className = "msg err"; m.textContent = e.message; }
};

$("#launch").onclick = async () => {
  const m = $("#setup-msg");
  try {
    await api(`/api/scenario?path=${encodeURIComponent(S.scenPath)}`, { method: "PUT", body: JSON.stringify(S.scen) });
    const r = await api("/api/runs", { method: "POST", body: JSON.stringify({ scenario: S.scenPath, duration: +$("#run-duration").value }) });
    m.className = "msg ok"; m.textContent = `started ${r.id}`;
    await loadRuns(); openRun(r.id, "run");
  } catch (e) { m.className = "msg err"; m.textContent = e.message; }
};

$("#upload-btn").onclick = () => uploadStl().catch(e => showError(e.message));
$("#fit-domain").onclick = async () => {
  try { const r = await api(`/api/scenario/auto-domain?path=${encodeURIComponent(S.scenPath)}`, { method: "POST" });
        S.scen.domain = r.domain; S.geoKey = null; await loadGeometry("scenario", S.scenPath); }
  catch (e) { showError(e.message); }
};

// ---------------------------------------------------------------- run
async function openRun(id, tab) {
  S.run = id; S.analysis = null; S.geoKey = null;
  const info = await api(`/api/runs/${encodeURIComponent(id)}`);
  S.runInfo = info;
  if (!info.meta?.parts) {               // failed before start: show the log, nothing to draw
    await showTab("run");
    $("#run-empty").hidden = true; $("#run-view").hidden = false;
    $("#run-title").textContent = id;
    const st = $("#run-status"); st.textContent = info.status; st.className = "badge " + info.status;
    $("#run-log").textContent = info.log; $("#run-log").closest("details").open = true;
    loadRuns(); return;
  }
  await showTab(tab);
  await refreshRun();
  loadRuns();
  clearInterval(S.poll);
  S.poll = setInterval(async () => {
    if (S.runInfo?.status === "running") { await refreshRun(); loadRuns(); }
  }, 2000);
}

async function refreshRun() {
  const r = await api(`/api/runs/${encodeURIComponent(S.run)}`);
  S.runInfo = r;
  const dur = r.meta?.output?.duration || 1;
  const h = r.history, t = h.time_s || [];
  // run tab
  $("#run-empty").hidden = true; $("#run-view").hidden = false;
  $("#run-title").textContent = r.id;
  const st = $("#run-status"); st.textContent = r.status; st.className = "badge " + r.status;
  $("#run-progress").style.width = `${Math.min(100, 100 * (t.at(-1) || 0) / dur)}%`;
  $("#stop-run").disabled = r.status !== "running";
  $("#run-log").textContent = r.log;
  $("#status-line").textContent = r.status === "running" ? `${r.id}: t = ${fmt(t.at(-1) || 0)} / ${fmt(dur)} s` : "";
  lineChart($("#chart-mass"), { title: "Mass (kg)", x: t, series: [
    { name: "held", y: h.mass_kg || [] }, { name: "injected", y: h.injected_kg || [] },
    { name: "discharged", y: h.discharged_kg || [] }] });
  const regs = Object.keys(h).filter(k => k.endsWith("_mass_kg") && k !== "mass_kg");
  lineChart($("#chart-regions"), { title: "Region mass (kg)", x: t,
    series: regs.map(k => ({ name: k.replace("_mass_kg", ""), y: h[k] })) });
  const flows = Object.keys(h).filter(k => k.startsWith("flow_"));
  lineChart($("#chart-flows"), { title: "Flow through planes (kg/s, 1 s mean)", x: t,
    series: flows.map(k => ({ name: k.slice(5, -3), y: rate(t, h[k], 1.0) })) });
  // results tab
  $("#results-empty").hidden = true; $("#results-view").hidden = false;
  $("#results-title").textContent = r.id;
  S.frames = r.frames; S.fps = r.fps;
  const slider = $("#frame-slider"); slider.max = Math.max(0, S.frames.length - 1);
  if ($("#win0").value === "" || S.lastRun !== r.id) {
    const tEnd = t.at(-1) || dur;
    $("#win0").value = +(Math.max(0, tEnd - Math.min(5, tEnd / 2))).toFixed(2); $("#win1").value = +tEnd.toFixed(2);
    S.lastRun = r.id; S.frameIdx = Math.max(0, S.frames.length - 1); slider.value = S.frameIdx;
  }
  $("#frame-bar").hidden = S.tab !== "results" || !S.frames.length;
  if (S.tab === "results") { await loadAnalysis(); showFrame(S.frameIdx); }
}

function rate(t, cum, span) {
  return t.map((ti, i) => {
    let j = i; while (j > 0 && ti - t[j - 1] <= span) j--;
    return i > j ? (cum[i] - cum[j]) / (ti - t[j]) : null;
  });
}

$("#stop-run").onclick = async () => { try { await api(`/api/runs/${S.run}/stop`, { method: "POST" }); } catch (e) { alert(e.message); } };

// ---------------------------------------------------------------- results: playback
async function showFrame(i) {
  if (!S.frames.length) return;
  S.frameIdx = i;
  const k = S.frames[i];
  $("#frame-label").textContent = `frame ${k}  ·  t = ${(k / S.fps).toFixed(2)} s`;
  const buf = await api(`/api/runs/${S.run}/frame/${k}`);
  const n = new Uint32Array(buf, 0, 1)[0];
  const pos = new Float32Array(buf, 4, n * 3), spd = new Float32Array(buf, 4 + 12 * n, n);
  S.lastSpeed = spd;
  const hi = S.speedMax || (S.speedMax = Math.max(1, percentile(spd, 0.99)));
  viewer.setParticles(pos, spd, 0, hi);
  viewer.showParticles($("#show-particles").checked);
  if ($("#colour-by").value === "speed") legend("speed (m/s)", 0, hi);
}

function percentile(a, q) { if (!a.length) return 0; const s = Float32Array.from(a).sort(); return s[Math.floor(q * (s.length - 1))]; }

$("#frame-slider").oninput = e => showFrame(+e.target.value);
$("#play").onclick = () => {
  S.playing = !S.playing; $("#play").textContent = S.playing ? "❚❚" : "▶";
  const step = async () => {
    if (!S.playing) return;
    const i = (S.frameIdx + 1) % S.frames.length;
    $("#frame-slider").value = i; await showFrame(i);
    setTimeout(step, 1000 / 15);
  };
  step();
};
$("#show-particles").onchange = e => viewer.showParticles(e.target.checked);

// ---------------------------------------------------------------- results: measurements
document.querySelectorAll("[data-analyze]").forEach(b => b.onclick = async () => {
  const msg = $("#analysis-msg");
  try {
    await api(`/api/runs/${S.run}/analyze`, { method: "POST", body: JSON.stringify({
      what: b.dataset.analyze, window: [+$("#win0").value, +$("#win1").value] }) });
    msg.className = "msg"; msg.textContent = `computing ${b.dataset.analyze}…`;
    const wait = async () => {
      const a = await api(`/api/runs/${S.run}/analysis`);
      if (a.running) return setTimeout(wait, 1000);
      msg.textContent = "done"; S.analysis = a; renderAnalysis();
    };
    wait();
  } catch (e) { msg.className = "msg err"; msg.textContent = e.message; }
});

async function loadAnalysis() {
  try { S.analysis = await api(`/api/runs/${S.run}/analysis`); renderAnalysis(); } catch (_) {}
}

function renderAnalysis() {
  const a = S.analysis || {};
  const w0 = +$("#win0").value, w1 = +$("#win1").value;
  // part loads: window means
  if (a.part_loads?.rows?.length) {
    const h = a.part_loads.header, rows = a.part_loads.rows.map(r => r.map((x, i) => i === 1 ? x : +x));
    const parts = [...new Set(rows.map(r => r[1]))];
    const byPart = parts.map(p => {
      const rr = rows.filter(r => r[1] === p);
      const mean = k => rr.reduce((s, r) => s + r[k], 0) / rr.length;
      return { p, F: Math.hypot(mean(2), mean(3), mean(4)), Fx: mean(2), Fy: mean(3), Fz: mean(4), rr };
    }).filter(x => x.F > 1e-9);
    $("#loads-out").innerHTML = `<h3>Wall loads — mean over ${fmt(rows[0][0])}–${fmt(rows.at(-1)[0])} s</h3>
      <table class="tbl"><tr><th>part</th><th>|F| (N)</th><th>Fx</th><th>Fy</th><th>Fz</th></tr>` +
      byPart.map(x => `<tr><td>${x.p}</td><td class="num">${fmt(x.F)}</td><td class="num">${fmt(x.Fx)}</td><td class="num">${fmt(x.Fy)}</td><td class="num">${fmt(x.Fz)}</td></tr>`).join("") + "</table>";
    const c = $("#chart-loads"); c.hidden = false;
    lineChart(c, { title: "Force on part |F| (N)", x: byPart[0]?.rr.map(r => r[0]) || [],
      series: byPart.slice(0, 8).map(x => ({ name: x.p, y: x.rr.map(r => Math.hypot(r[2], r[3], r[4])) })) });
  }
  if (a.flows?.rows?.length) {
    const h = a.flows.header, rows = a.flows.rows.map(r => r.map(Number));
    const span = rows.at(-1)[0] - rows[0][0] || 1;
    $("#flows-out").innerHTML = `<h3>Flows — ${fmt(rows[0][0])}–${fmt(rows.at(-1)[0])} s</h3><table class="tbl"><tr><th>plane</th><th>mass (kg)</th><th>rate (kg/s)</th><th>share</th></tr>` +
      h.slice(1).map((n, k) => {
        const tot = h.slice(1).reduce((s, _, j) => s + rows.at(-1)[j + 1] - rows[0][j + 1], 0) || 1;
        const m = rows.at(-1)[k + 1] - rows[0][k + 1];
        return `<tr><td>${n.replace(/_kg$/, "")}</td><td class="num">${fmt(m)}</td><td class="num">${fmt(m / span)}</td><td class="num">${(100 * m / tot).toFixed(1)}%</td></tr>`;
      }).join("") + "</table>";
    const c = $("#chart-aflows"); c.hidden = false;
    const t = rows.map(r => r[0]);
    lineChart(c, { title: "Flow rate (kg/s, 1 s mean)", x: t,
      series: h.slice(1).map((n, k) => ({ name: n.replace(/_kg$/, ""), y: rate(t, rows.map(r => r[k + 1]), 1.0) })) });
  }
  if (a.regions?.rows?.length) {
    const h = a.regions.header, rows = a.regions.rows.map(r => r.map(Number));
    const mean = k => rows.reduce((s, r) => s + r[k], 0) / rows.length;
    $("#regions-out").innerHTML = `<h3>Regions — mean over ${fmt(rows[0][0])}–${fmt(rows.at(-1)[0])} s</h3><table class="tbl"><tr><th>region</th><th>mass (kg)</th><th>speed (m/s)</th></tr>` +
      Array.from({ length: (h.length - 1) / 2 }, (_, k) => `<tr><td>${h[1 + 2 * k].replace(/_kg$/, "")}</td><td class="num">${fmt(mean(1 + 2 * k))}</td><td class="num">${fmt(mean(2 + 2 * k))}</td></tr>`).join("") + "</table>";
  }
  refreshColouring();
}

$("#colour-by").onchange = refreshColouring;
function refreshColouring() {
  const key = $("#colour-by").value;
  if (S.tab !== "results") return;
  if (key === "speed") {
    viewer.clearWallMap();
    if (S.lastSpeed) legend("speed (m/s)", 0, S.speedMax || 1);
    return;
  }
  const maps = S.analysis?.wall_maps;
  if (!maps?.[key]) { $("#legend").hidden = false; $("#legend").innerHTML = "Run <b>Wall loads</b> for this window first."; viewer.clearWallMap(); return; }
  const bin = atob(maps[key]), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  const vals = new Float32Array(u8.buffer);
  const pos = vals.filter(v => v > 0);
  const hi = pos.length ? percentile(pos, 0.98) : 1;
  viewer.setWallMap(vals, 0, hi);
  const labels = { pressure_Pa: "pressure (Pa)", shear_Pa: "shear (Pa)", wear_rate_W_m2: "wear rate (W/m²)", contacts_per_frame: "contacts per frame" };
  const w = S.analysis.wall_maps_window;
  legend(`${labels[key]}${w ? `, mean ${fmt(w[0])}–${fmt(w[1])} s` : ""}`, 0, hi);
}

function legend(title, lo, hi) {
  const L = $("#legend"); L.hidden = false;
  L.innerHTML = `<div>${title}</div><div class="bar" style="background:${colormapCss()}"></div><div class="ends"><span>${fmt(lo)}</span><span>${fmt(hi)}</span></div>`;
}

// ---------------------------------------------------------------- start
// deep links: #scenario=<path>  |  #run=<id>&tab=run|results&frame=<index>&colour=<key>
async function route() {
  const q = new URLSearchParams(location.hash.slice(1));
  if (q.get("scenario")) await openScenario(q.get("scenario"));
  else if (q.get("run")) {
    if (q.get("colour")) $("#colour-by").value = q.get("colour");
    await openRun(q.get("run"), q.get("tab") || "results");
    if (q.get("frame") != null) { $("#frame-slider").value = +q.get("frame"); await showFrame(+q.get("frame")); }
    refreshColouring();
  }
  document.body.dataset.ready = "1";
}
loadProjects(); loadRuns(); route();
