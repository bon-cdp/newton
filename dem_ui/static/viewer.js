// 3D view: parts (semi-transparent, crease edges), injector faces, flow planes, domain box,
// particles coloured by a scalar, and per-triangle wall maps.
import * as THREE from "three";
import { OrbitControls } from "/static/vendor/OrbitControls.js";
import { PALETTE } from "/static/chart.js";

const VIRIDIS = [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
                 [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]];

export function colormap(t) {
  t = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const i = Math.min(Math.floor(t), VIRIDIS.length - 2), f = t - i;
  return VIRIDIS[i].map((c, k) => (c + (VIRIDIS[i + 1][k] - c) * f) / 255);
}
export function colormapCss() {
  return `linear-gradient(90deg, ${VIRIDIS.map(c => `rgb(${c})`).join(",")})`;
}

function b64(s, Type) {
  const bin = atob(s), buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Type(buf.buffer);
}

function discSprite() {
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(28, 26, 4, 32, 32, 31);
  grd.addColorStop(0, "#ffffff"); grd.addColorStop(0.7, "#d8d8d8"); grd.addColorStop(1, "#9a9a9a");
  g.fillStyle = grd; g.beginPath(); g.arc(32, 32, 31, 0, 2 * Math.PI); g.fill();
  return new THREE.CanvasTexture(c);
}

export class Viewer {
  constructor(el) {
    this.el = el;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setClearColor(0xf0f2f5);
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
    this.camera.position.set(3, 2, 5);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8890a0, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 0.8); sun.position.set(3, 8, 5);
    this.scene.add(sun);
    this.world = new THREE.Group(); this.scene.add(this.world);
    this.parts = [];                  // {name, mesh, edges, triOffset, nTri, color}
    this.points = null;
    this.sprite = discSprite();
    this.grainRadius = 0.006;
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    const loop = () => { this.controls.update(); this.renderer.render(this.scene, this.camera); requestAnimationFrame(loop); };
    loop();
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  // Geometry reloads clear the world group only; particles live directly in the scene, so a
  // frame drawn while geometry was still loading survives it.
  clear() {
    for (const o of [...this.world.children]) { this.world.remove(o); o.geometry?.dispose?.(); }
    this.parts = [];
  }

  setGeometry(geo, grainRadius) {
    this.clear();
    this.grainRadius = grainRadius || this.grainRadius;
    const box = new THREE.Box3();
    let triOffset = 0;
    geo.parts.forEach((p, k) => {
      const v = b64(p.vertices, Float32Array), f = b64(p.faces, Uint32Array);
      const pos = new Float32Array(f.length * 3);
      for (let i = 0; i < f.length; i++) pos.set(v.subarray(f[i] * 3, f[i] * 3 + 3), i * 3);
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.computeVertexNormals();
      const col = new THREE.Color(PALETTE[k % PALETTE.length]);
      const cols = new Float32Array(pos.length);
      for (let i = 0; i < cols.length; i += 3) { cols[i] = 0.55 + 0.45 * col.r; cols[i + 1] = 0.55 + 0.45 * col.g; cols[i + 2] = 0.55 + 0.45 * col.b; }
      g.setAttribute("color", new THREE.BufferAttribute(cols, 3));
      const mesh = new THREE.Mesh(g, new THREE.MeshLambertMaterial({
        vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.45, depthWrite: false }));
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(g, 25),
        new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.8 }));
      this.world.add(mesh, edges);
      g.computeBoundingBox(); box.union(g.boundingBox);
      this.parts.push({ name: p.name, mesh, edges, triOffset, nTri: p.n_tri, color: PALETTE[k % PALETTE.length], baseColors: cols.slice() });
      triOffset += p.n_tri;
    });
    for (const inj of geo.injectors || []) {
      const t = b64(inj.triangles, Float32Array);
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(t, 3));
      this.world.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0x16a34a, side: THREE.DoubleSide, transparent: true, opacity: 0.5 })));
    }
    for (const fp of geo.flow_planes || []) this.world.add(this.plane(fp));
    const d = geo.domain;
    if (d) {
      const lo = new THREE.Vector3(...d.lo), hi = new THREE.Vector3(...d.hi);
      const bx = new THREE.Box3Helper(new THREE.Box3(lo, hi), 0x9aa3b2);
      this.world.add(bx);
    }
    this.fit(box);
  }

  plane(fp) {
    const a = fp.axis, [u, v] = [0, 1, 2].filter(k => k !== a);
    const c = (lo, hi) => { const p = [0, 0, 0]; p[a] = fp.value; p[u] = lo; p[v] = hi; return p; };
    const lo = fp.lo, hi = fp.hi;
    const q = [c(lo[u], lo[v]), c(hi[u], lo[v]), c(hi[u], hi[v]), c(lo[u], hi[v])];
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array([...q[0], ...q[1], ...q[2], ...q[0], ...q[2], ...q[3]]), 3));
    return new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0xf59e0b, side: THREE.DoubleSide, transparent: true, opacity: 0.18, depthWrite: false }));
  }

  fit(box) {
    if (box.isEmpty()) return;
    const c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3()).length();
    this.controls.target.copy(c);
    this.camera.position.copy(c).add(new THREE.Vector3(0.55, 0.35, 0.75).multiplyScalar(s));
    this.camera.near = s / 1000; this.camera.far = s * 20; this.camera.updateProjectionMatrix();
  }

  setPartVisible(name, on) {
    const p = this.parts.find(p => p.name === name);
    if (p) { p.mesh.visible = on; p.edges.visible = on; }
  }

  setParticles(pos, scalar, lo, hi) {
    const n = pos.length / 3, cols = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) cols.set(colormap((scalar[i] - lo) / (hi - lo || 1)), i * 3);
    if (!this.points) {
      const g = new THREE.BufferGeometry();
      this.points = new THREE.Points(g, new THREE.PointsMaterial({
        size: 2 * this.grainRadius, sizeAttenuation: true, vertexColors: true,
        map: this.sprite, alphaTest: 0.5 }));
      this.scene.add(this.points);
    }
    this.points.material.size = 2 * this.grainRadius;
    const g = this.points.geometry;
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(cols, 3));
    g.computeBoundingSphere();
  }

  showParticles(on) { if (this.points) this.points.visible = on; }

  // values: one per triangle across all parts, in the solver's order
  setWallMap(values, lo, hi) {
    for (const p of this.parts) {
      const col = p.mesh.geometry.getAttribute("color");
      for (let t = 0; t < p.nTri; t++) {
        const x = values[p.triOffset + t];
        const c = x > 0 ? colormap((x - lo) / (hi - lo || 1)) : [0.82, 0.84, 0.88];
        for (let k = 0; k < 3; k++) col.array.set(c, (t * 3 + k) * 3);
      }
      col.needsUpdate = true;
      p.mesh.material.opacity = 0.9; p.mesh.material.depthWrite = true;
    }
  }

  clearWallMap() {
    for (const p of this.parts) {
      const col = p.mesh.geometry.getAttribute("color");
      col.array.set(p.baseColors); col.needsUpdate = true;
      p.mesh.material.opacity = 0.45; p.mesh.material.depthWrite = false;
    }
  }
}
