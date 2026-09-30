// Minimal canvas line chart: lineChart(canvas, {title, x, series: [{name, y, color}], ylabel})
export const PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2",
                        "#ca8a04", "#db2777", "#4b5563", "#65a30d"];

export function lineChart(canvas, { title = "", x = [], series = [], ylabel = "", xlabel = "t (s)" }) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.height;
  canvas.width = w * dpr; canvas.style.height = h + "px"; canvas.height = h * dpr;
  const c = canvas.getContext("2d");
  c.scale(dpr, dpr);
  c.clearRect(0, 0, w, h);
  const L = 48, R = 10, T = 22, B = 30;
  const xs = x.filter(v => v != null);
  const ys = series.flatMap(s => s.y.filter(v => v != null && isFinite(v)));
  c.fillStyle = "#1d2330"; c.font = "600 12px system-ui"; c.fillText(title, L, 14);
  if (!xs.length || !ys.length) {
    c.fillStyle = "#6b7384"; c.font = "12px system-ui"; c.fillText("no data yet", L, h / 2);
    return;
  }
  let x0 = Math.min(...xs), x1 = Math.max(...xs); if (x1 === x0) x1 = x0 + 1;
  let y0 = Math.min(0, ...ys), y1 = Math.max(...ys); if (y1 === y0) y1 = y0 + 1;
  const pad = (y1 - y0) * 0.06; y1 += pad; if (y0 < 0) y0 -= pad;
  const X = v => L + (v - x0) / (x1 - x0) * (w - L - R);
  const Y = v => h - B - (v - y0) / (y1 - y0) * (h - T - B);
  c.strokeStyle = "#e5e7eb"; c.fillStyle = "#6b7384"; c.font = "10px system-ui"; c.lineWidth = 1;
  for (const v of ticks(y0, y1, 4)) {
    c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(w - R, Y(v)); c.stroke();
    c.fillText(fmt(v), 4, Y(v) + 3);
  }
  for (const v of ticks(x0, x1, 6)) c.fillText(fmt(v), X(v) - 8, h - B + 13);
  c.fillText(xlabel, w - R - 30, h - 4);
  if (ylabel) c.fillText(ylabel, 4, T - 8);
  series.forEach((s, k) => {
    c.strokeStyle = s.color || PALETTE[k % PALETTE.length]; c.lineWidth = 1.5; c.beginPath();
    let started = false;
    s.y.forEach((v, i) => {
      if (v == null || !isFinite(v) || x[i] == null) { started = false; return; }
      if (!started) { c.moveTo(X(x[i]), Y(v)); started = true; } else c.lineTo(X(x[i]), Y(v));
    });
    c.stroke();
  });
  // legend
  let lx = L + 8;
  c.font = "10px system-ui";
  series.forEach((s, k) => {
    const col = s.color || PALETTE[k % PALETTE.length];
    const tw = c.measureText(s.name).width;
    if (lx + tw + 20 > w - R) return;
    c.fillStyle = col; c.fillRect(lx, T + 2, 10, 3);
    c.fillStyle = "#1d2330"; c.fillText(s.name, lx + 13, T + 7);
    lx += tw + 26;
  });
}

function ticks(a, b, n) {
  const step = niceStep((b - a) / n);
  const out = [];
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}
export function fmt(v) {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e5 || a < 1e-2) return v.toExponential(1);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}
