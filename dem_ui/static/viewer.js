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
    this.parts = []; this.arrows = [];
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
      if (inj.triangles) {
        const t = b64(inj.triangles, Float32Array);
        const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(t, 3));
        this.world.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0x16a34a, side: THREE.DoubleSide, transparent: true, opacity: 0.5 })));
      }
      if (inj.box_corners) this.world.add(...this.injectorBox(inj.box_corners));
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

  // corners indexed i*4 + j*2 + k over (along, across, up)
  injectorBox(c) {
    const P = c.map(p => new THREE.Vector3(...p));
    const edges = [[0, 4], [2, 6], [1, 5], [3, 7], [0, 2], [4, 6], [1, 3], [5, 7], [0, 1], [2, 3], [4, 5], [6, 7]];
    const lg = new THREE.BufferGeometry().setFromPoints(edges.flatMap(([a, b]) => [P[a], P[b]]));
    const lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x16a34a }));
    const faces = [[0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4], [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6]];
    const pos = faces.flatMap(([a, b, c2, d]) => [P[a], P[b], P[c2], P[a], P[c2], P[d]]);
    const fg = new THREE.BufferGeometry().setFromPoints(pos);
    const mesh = new THREE.Mesh(fg, new THREE.MeshBasicMaterial({ color: 0x16a34a, side: THREE.DoubleSide, transparent: true, opacity: 0.15, depthWrite: false }));
    return [lines, mesh];
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

  // ---- conveyor direction: pick an edge by clicking a part, BFA-style ----------------
  // Resolves {dir, mid, a, b} for the triangle edge nearest the click, or null on Esc.
  pickEdge(partName) {
    const part = this.parts.find(p => p.name === partName);
    if (!part) return Promise.resolve(null);
    const canvas = this.renderer.domElement;
    const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
    canvas.style.cursor = "crosshair";
    const prevOpacity = part.mesh.material.opacity;
    part.mesh.material.opacity = 0.8;
    return new Promise(resolve => {
      const done = r => {
        canvas.style.cursor = ""; part.mesh.material.opacity = prevOpacity;
        canvas.removeEventListener("pointerdown", onDown); window.removeEventListener("keydown", onKey);
        resolve(r);
      };
      const onKey = e => { if (e.key === "Escape") done(null); };
      const onDown = e => {
        if (e.button !== 0) return;
        const r = canvas.getBoundingClientRect();
        ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
        ray.setFromCamera(ndc, this.camera);
        const hit = ray.intersectObject(part.mesh, false)[0];
        if (!hit) return;                                   // keep waiting for a hit
        e.stopImmediatePropagation();                       // do not start an orbit drag
        const pos = part.mesh.geometry.getAttribute("position");
        const i0 = hit.faceIndex * 3;
        const V = k => new THREE.Vector3().fromBufferAttribute(pos, i0 + k);
        const tri = [V(0), V(1), V(2)];
        let best = null;
        for (let k = 0; k < 3; k++) {
          const a = tri[k], b = tri[(k + 1) % 3];
          const line = new THREE.Line3(a, b), cp = new THREE.Vector3();
          line.closestPointToPoint(hit.point, true, cp);
          const d = cp.distanceTo(hit.point);
          if (!best || d < best.d) best = { d, a, b };
        }
        const dir = best.b.clone().sub(best.a).normalize();
        this.flashEdge(best.a, best.b);
        done({ dir: dir.toArray(), mid: best.a.clone().add(best.b).multiplyScalar(0.5).toArray(),
               a: best.a.toArray(), b: best.b.toArray() });
      };
      canvas.addEventListener("pointerdown", onDown, true);
      window.addEventListener("keydown", onKey);
    });
  }

  flashEdge(a, b) {
    const g = new THREE.BufferGeometry().setFromPoints([a, b]);
    const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xdc2626, linewidth: 3 }));
    this.world.add(l);
    setTimeout(() => { this.world.remove(l); g.dispose(); }, 2500);
  }

  // Arrows showing each moving part's motion: belts along their running direction,
  // rotating parts along their axis.
  setMotionArrows(parts) {
    for (const a of this.arrows || []) this.world.remove(a);
    this.arrows = [];
    for (const p of parts) {
      const vp = this.parts.find(q => q.name === p.name);
      if (!vp || !p.motion) continue;
      const box = new THREE.Box3().setFromObject(vp.mesh);
      const size = box.getSize(new THREE.Vector3()), c = box.getCenter(new THREE.Vector3());
      const L = Math.max(0.3, 0.18 * size.length());
      if (p.motion.type === "belt") {
        const d = new THREE.Vector3(...p.motion.velocity);
        if (d.lengthSq() === 0) continue;
        d.normalize();
        // three arrows along the running direction, lifted to the top of the part
        for (const f of [-0.3, 0, 0.3]) {
          const o = c.clone().addScaledVector(d, f * size.length() * 0.6);
          o.y = box.max.y + 0.02 * size.length();
          const arr = new THREE.ArrowHelper(d, o.clone().addScaledVector(d, -L / 2), L, 0xdc2626, L * 0.3, L * 0.18);
          this.world.add(arr); this.arrows.push(arr);
        }
      } else if (p.motion.type === "rotating") {
        const ax = new THREE.Vector3(...p.motion.axis).normalize();
        const o = new THREE.Vector3(...p.motion.point);
        const arr = new THREE.ArrowHelper(ax, o.clone().addScaledVector(ax, -L / 2), L, 0x9333ea, L * 0.3, L * 0.18);
        this.world.add(arr); this.arrows.push(arr);
      }
    }
  }

  partCentroid(name) {
    const p = this.parts.find(q => q.name === name);
    if (!p) return [0, 0, 0];
    return new THREE.Box3().setFromObject(p.mesh).getCenter(new THREE.Vector3()).toArray();
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
        size: this.pointSize(), sizeAttenuation: true, vertexColors: true,
        map: this.sprite, alphaTest: 0.5 }));
      this.scene.add(this.points);
    }
    this.points.material.size = this.pointSize();
    const g = this.points.geometry;
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(cols, 3));
    g.computeBoundingSphere();
  }

  // three.js draws an attenuated point size * (viewport height / 2) / depth pixels across,
  // which leaves out the camera's field of view: a grain's true diameter on screen is
  // 2r * (height / 2) / (depth * tan(fov / 2)).  The disc sprite fills 31/32 of its texture.
  pointSize() {
    const t = Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2);
    return 2 * this.grainRadius / t * 32 / 31;
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
