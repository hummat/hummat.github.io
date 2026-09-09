(function () {
  "use strict";

  const DEFAULT_FOCAL_DISTANCE = 5;
  const DEFAULT_APERTURE_ANGLE = 0.02;
  let modulesPromise;

  function loadModules() {
    if (!modulesPromise) {
      modulesPromise = Promise.all([
        import("@sparkjsdev/spark"),
        import("three"),
        import("three/addons/controls/OrbitControls.js"),
      ]).then(([sparkModule, THREE, controlsModule]) => ({
        OrbitControls: controlsModule.OrbitControls,
        SparkRenderer: sparkModule.SparkRenderer,
        SplatMesh: sparkModule.SplatMesh,
        THREE,
      }));
    }

    return modulesPromise;
  }

  function parseStrictFinite(value) {
    if (value === undefined || value === null) {
      return null;
    }
    const str = String(value).trim();
    if (str === "") {
      return null;
    }
    const num = Number(str);
    return Number.isFinite(num) ? num : NaN;
  }

  function parseAuthoredCenter(value) {
    if (value === undefined || value === null) {
      return null;
    }
    const str = String(value).trim();
    if (str === "") {
      return null;
    }
    const parts = str.split(",");
    if (parts.length !== 3) {
      return NaN;
    }
    const coords = parts.map((p) => {
      const s = p.trim();
      if (s === "") {
        return NaN;
      }
      const n = Number(s);
      return Number.isFinite(n) ? n : NaN;
    });
    if (coords.some((n) => !Number.isFinite(n))) {
      return NaN;
    }
    return coords;
  }

  function readNumber(value, fallback) {
    const number = Number.parseFloat(value);
    return Number.isFinite(number) ? number : fallback;
  }
  function supportsWebGL2() {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2"));
  }

  function setState(viewer, state, message) {
    const button = viewer.querySelector("[data-spark-load]");
    const poster = viewer.querySelector(".spark-viewer__poster");
    const status = viewer.querySelector("[data-spark-message]");

    viewer.dataset.sparkState = state;
    viewer.setAttribute("aria-busy", state === "loading" ? "true" : "false");

    if (poster) {
      poster.setAttribute("aria-hidden", state === "ready" ? "true" : "false");
    }

    if (status) {
      status.hidden = !message;
      status.textContent = message || "";
    }

    if (!button) {
      return;
    }

    button.disabled = state === "loading";
    button.hidden = state === "ready" || state === "fallback";
    if (state === "loading") {
      button.textContent = "Loading…";
    } else if (state === "error") {
      button.textContent = "Try again";
    } else {
      button.textContent = "Load interactive view";
    }
  }

  function updateProgress(viewer, event) {
    if (event.lengthComputable && event.total > 0) {
      const percent = Math.round((event.loaded / event.total) * 100);
      setState(viewer, "loading", `Loading interactive view… ${percent}%`);
    } else {
      setState(viewer, "loading", "Loading interactive view…");
    }
  }

  function computeRobustBounds(THREE, splat) {
    const count = splat.numSplats;
    if (!count) {
      throw new Error("Splat contains no renderable points");
    }
    // Frame on what is visible: low-opacity splats are reconstruction noise
    // (floaters), so weight the centroid and framing radius by opacity.
    const center = new THREE.Vector3();
    let weight = 0;
    splat.forEachSplat((index, splatCenter, scales, quaternion, opacity) => {
      center.addScaledVector(splatCenter, opacity);
      weight += opacity;
    });
    if (weight <= 0) {
      weight = 1;
    }
    center.divideScalar(weight);
    const distances = new Float32Array(count);
    let seen = 0;
    splat.forEachSplat((index, splatCenter, scales, quaternion, opacity) => {
      if (opacity >= 0.1) {
        distances[seen++] = splatCenter.distanceTo(center);
      }
    });
    if (seen < count * 0.01) {
      seen = 0;
      splat.forEachSplat((index, splatCenter) => {
        distances[seen++] = splatCenter.distanceTo(center);
      });
    }
    const framed = distances.subarray(0, seen);
    framed.sort();
    // Ignore the 2% farthest visible splats so stray points cannot shrink
    // the subject in frame.
    const radius = framed[Math.min(seen - 1, Math.floor(seen * 0.98))];
    return { center, radius: Math.max(radius, 0.001) };
  }

  function frameSplat(THREE, splat, camera, controls, focalDistance) {
    const { center, radius } = computeRobustBounds(THREE, splat);
    if (!Number.isFinite(center.x) || !Number.isFinite(center.y) || !Number.isFinite(center.z)) {
      throw new Error("Splat contains no renderable points");
    }
    const cameraDistance = Math.max(focalDistance, 0.1);
    const fitRadius = Math.max(
      0.5,
      cameraDistance * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 0.85
    );
    const scale = fitRadius / radius;
    // Splat exports are z-up (nerfstudio/COLMAP); three.js is y-up, so map
    // source +z to display +y and keep the camera looking at the horizon.
    const upFix = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
    splat.quaternion.copy(upFix);
    splat.scale.setScalar(scale);
    splat.position.copy(center).applyQuaternion(upFix).multiplyScalar(-scale);

    camera.position.set(0, 0, cameraDistance);
    camera.near = Math.max(0.01, cameraDistance - fitRadius * 3);
    camera.far = Math.max(100, cameraDistance + fitRadius * 4);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();

    controls.target.set(0, 0, 0);
    controls.minDistance = Math.max(0.1, cameraDistance * 0.25);
    controls.maxDistance = Math.max(10, cameraDistance * 4);
    controls.update();

    return { center: center.clone(), radius };
  }
  function createCanvas(viewer, alt) {
    const canvas = document.createElement("canvas");
    canvas.className = "spark-viewer__canvas";
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", alt);
    viewer.querySelector(".spark-viewer__stage").appendChild(canvas);
    return canvas;
  }

  // Live viewer handles per figure, so a viewer that scrolls out of the
  // frame can be torn down and its GPU memory and WebGL context freed.
  const resources = new WeakMap();

  // WeakMap keyed by viewer figure, holding a Map keyed by asset URL.
  // Each resolved setting is { mode, center: [x, y, z], radius, transition, strength, showGuide, invalidAuthored, defaults: {...} }
  const blurSettings = new WeakMap();

  function getViewerBlurMap(viewer) {
    let map = blurSettings.get(viewer);
    if (!map) {
      map = new Map();
      blurSettings.set(viewer, map);
    }
    return map;
  }

  function resolveBlurSetting(viewer, url, bounds) {
    const map = getViewerBlurMap(viewer);
    const existing = map.get(url);
    if (existing) {
      return existing;
    }

    const defaultCenter = [bounds.center.x, bounds.center.y, bounds.center.z];
    const defaultRadius = 0.5 * bounds.radius;
    const defaultTransition = 0.25 * bounds.radius;
    const defaultStrength = 12;
    const defaultShowGuide = true;

    let invalidAuthored = false;
    let mode = "camera";
    const rawMode = viewer.dataset.sparkBlurMode;
    if (rawMode && rawMode.trim() !== "") {
      const trimmedMode = rawMode.trim();
      if (["camera", "sphere", "off"].includes(trimmedMode)) {
        mode = trimmedMode;
      } else {
        invalidAuthored = true;
      }
    }

    let center = [...defaultCenter];
    const authoredCenter = parseAuthoredCenter(viewer.dataset.sparkBlurCenter);
    if (Number.isNaN(authoredCenter)) {
      invalidAuthored = true;
    } else if (Array.isArray(authoredCenter)) {
      center = authoredCenter;
    }

    let radius = defaultRadius;
    const authoredRadius = parseStrictFinite(viewer.dataset.sparkBlurRadius);
    if (Number.isNaN(authoredRadius) || (authoredRadius !== null && authoredRadius <= 0)) {
      invalidAuthored = true;
    } else if (authoredRadius !== null) {
      radius = authoredRadius;
    }

    let transition = defaultTransition;
    const authoredTrans = parseStrictFinite(viewer.dataset.sparkBlurTransition);
    if (Number.isNaN(authoredTrans) || (authoredTrans !== null && authoredTrans <= 0)) {
      invalidAuthored = true;
    } else if (authoredTrans !== null) {
      transition = authoredTrans;
    }

    let strength = defaultStrength;
    const authoredStrength = parseStrictFinite(viewer.dataset.sparkBlurStrength);
    if (
      Number.isNaN(authoredStrength) ||
      (authoredStrength !== null && (authoredStrength < 0 || authoredStrength > 24))
    ) {
      invalidAuthored = true;
    } else if (authoredStrength !== null) {
      strength = authoredStrength;
    }

    if (invalidAuthored) {
      mode = "camera";
      center = [...defaultCenter];
      radius = defaultRadius;
      transition = defaultTransition;
      strength = defaultStrength;
    }

    const defaults = {
      center: [...center],
      radius,
      transition,
      strength,
      showGuide: defaultShowGuide,
    };

    const setting = {
      mode,
      center,
      radius,
      transition,
      strength,
      showGuide: defaultShowGuide,
      invalidAuthored,
      defaults,
    };

    map.set(url, setting);
    return setting;
  }

  function installSphereBlur(THREE, spark) {
    const material = spark.material;
    const vtx = material.vertexShader;
    const frg = material.fragmentShader;

    const vtxMainTarget = "void main() {";
    const vtxBlurTarget = "float fullBlurAmount = blurAmount;";
    const vtxDetOrigTarget = "float detOrig = a * d - b * b;";
    const vtxBlurAdjustTarget = "float blurAdjust = sqrt(max(0.0, detOrig / det));";
    const vtxAlphaCutTarget = "rgba.a *= blurAdjust;\n    if (rgba.a < minAlpha) {";
    const frgMainTarget = "void main() {";
    const frgAlphaCutTarget = "if (rgba.a < minAlpha)";

    function countOccurrences(src, target) {
      let count = 0;
      let pos = 0;
      while ((pos = src.indexOf(target, pos)) !== -1) {
        count++;
        pos += target.length;
      }
      return count;
    }

    if (
      countOccurrences(vtx, vtxMainTarget) !== 1 ||
      countOccurrences(vtx, vtxBlurTarget) !== 1 ||
      countOccurrences(vtx, vtxDetOrigTarget) !== 1 ||
      countOccurrences(vtx, vtxBlurAdjustTarget) !== 1 ||
      countOccurrences(vtx, vtxAlphaCutTarget) !== 1 ||
      countOccurrences(frg, frgMainTarget) !== 1 ||
      countOccurrences(frg, frgAlphaCutTarget) !== 1
    ) {
      throw new Error("Unsupported Spark shader for spherical background blur");
    }

    const vtxUniformsDecl = `
uniform bool sphereBlurEnabled;
uniform vec3 sphereBlurCenterView;
uniform float sphereBlurRadius;
uniform float sphereBlurTransition;
uniform float sphereBlurSigma;
flat out float vSphereMinAlpha;
`;
    let patchedVtx = vtx.replace(
      vtxMainTarget,
      `${vtxUniformsDecl}\nvoid main() {\n    vSphereMinAlpha = minAlpha;\n`
    );

    const vtxBlurCalc = `
    float fullBlurAmount = blurAmount;
    float extraVariance = 0.0;
    if (sphereBlurEnabled) {
        float distance = length(viewCenter - sphereBlurCenterView);
        float t = clamp((distance - sphereBlurRadius) / sphereBlurTransition, 0.0, 1.0);
        float weight = t * t * (3.0 - 2.0 * t);
        float sigma = sphereBlurSigma * weight;
        extraVariance = sigma * sigma;
    }
`;
    patchedVtx = patchedVtx.replace(vtxBlurTarget, vtxBlurCalc);

    const vtxBaselineDet = `
    float baselineDet = (a + fullBlurAmount) * (d + fullBlurAmount) - b * b;
    fullBlurAmount += extraVariance;
    float detOrig = a * d - b * b;
`;
    patchedVtx = patchedVtx.replace(vtxDetOrigTarget, vtxBaselineDet);

    const vtxAdjustWithCutoff = `
    float blurAdjust = sqrt(max(0.0, detOrig / det));
    if (sphereBlurEnabled && baselineDet > 0.0 && det > 0.0) {
        vSphereMinAlpha = minAlpha * sqrt(clamp(baselineDet / det, 0.0, 1.0));
    }
`;
    patchedVtx = patchedVtx.replace(vtxBlurAdjustTarget, vtxAdjustWithCutoff);

    patchedVtx = patchedVtx.replace(
      vtxAlphaCutTarget,
      "rgba.a *= blurAdjust;\n    if (rgba.a < vSphereMinAlpha) {"
    );

    const frgVaryingDecl = `
flat in float vSphereMinAlpha;
`;
    let patchedFrg = frg.replace(frgMainTarget, `${frgVaryingDecl}\nvoid main() {`);
    patchedFrg = patchedFrg.replace(frgAlphaCutTarget, "if (rgba.a < vSphereMinAlpha)");

    material.vertexShader = patchedVtx;
    material.fragmentShader = patchedFrg;

    const uniforms = {
      sphereBlurEnabled: { value: false },
      sphereBlurCenterView: { value: new THREE.Vector3() },
      sphereBlurRadius: { value: 0 },
      sphereBlurTransition: { value: 1 },
      sphereBlurSigma: { value: 0 },
    };
    Object.assign(material.uniforms, uniforms);
    material.needsUpdate = true;

    return uniforms;
  }

  function supportsFloatColorBuffer(renderer) {
    const gl = renderer.getContext();
    return Boolean(gl && gl.getExtension("EXT_color_buffer_float"));
  }

  function createBlurTarget(THREE, renderer) {
    if (!supportsFloatColorBuffer(renderer)) {
      return null;
    }

    const gl = renderer.getContext();
    const size = new THREE.Vector2();
    renderer.getDrawingBufferSize(size);
    const width = Math.max(Math.floor(size.x), 1);
    const height = Math.max(Math.floor(size.y), 1);

    const target = new THREE.WebGLRenderTarget(width, height, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      colorSpace: THREE.NoColorSpace,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: false,
      samples: 0,
    });

    const previousTarget = renderer.getRenderTarget();
    renderer.setRenderTarget(target);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    renderer.setRenderTarget(previousTarget);

    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      target.dispose();
      return null;
    }

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]);
    const uvs = new Float32Array([0, 0, 2, 0, 0, 2]);
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));

    const copyMaterial = new THREE.ShaderMaterial({
      uniforms: {
        sourceTexture: { value: target.texture },
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D sourceTexture;
        varying vec2 vUv;
        void main() {
          gl_FragColor = texture2D(sourceTexture, vUv);
        }
      `,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending,
      toneMapped: false,
    });

    const mesh = new THREE.Mesh(geometry, copyMaterial);
    mesh.frustumCulled = false;

    const copyScene = new THREE.Scene();
    copyScene.add(mesh);
    const copyCamera = new THREE.Camera();

    function setSize(newWidth, newHeight) {
      target.setSize(newWidth, newHeight);
      const prev = renderer.getRenderTarget();
      renderer.setRenderTarget(target);
      const valid = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      renderer.setRenderTarget(prev);
      return valid;
    }

    function dispose() {
      target.dispose();
      geometry.dispose();
      copyMaterial.dispose();
    }

    return {
      target,
      copyScene,
      copyCamera,
      setSize,
      dispose,
    };
  }

  function createCircleGeometry(THREE, segments) {
    const points = [];
    for (let i = 0; i <= segments; i++) {
      const theta = (i / segments) * Math.PI * 2;
      points.push(new THREE.Vector3(Math.cos(theta), Math.sin(theta), 0));
    }
    return new THREE.BufferGeometry().setFromPoints(points);
  }

  function createWireSphereGuide(THREE, circleGeometry, colorHex, opacity) {
    const group = new THREE.Group();
    const material = new THREE.LineBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity,
      depthTest: false,
      depthWrite: false,
    });

    const lineXY = new THREE.Line(circleGeometry, material);
    lineXY.renderOrder = 999;

    const lineXZ = new THREE.Line(circleGeometry, material);
    lineXZ.rotation.x = Math.PI / 2;
    lineXZ.renderOrder = 999;

    const lineYZ = new THREE.Line(circleGeometry, material);
    lineYZ.rotation.y = Math.PI / 2;
    lineYZ.renderOrder = 999;

    group.add(lineXY, lineXZ, lineYZ);
    return { group, material };
  }

  function syncBlurPanelUI(viewer, setting, bounds, hasFloatTarget) {
    const panel = viewer.querySelector("[data-spark-blur-panel]");
    if (!panel) {
      return;
    }

    const modeSelect = panel.querySelector("[data-spark-blur-mode-select]");
    const sphereOption = modeSelect ? modeSelect.querySelector('option[value="sphere"]') : null;
    const fieldset = panel.querySelector("[data-spark-blur-sphere-fields]");
    const cx = panel.querySelector("[data-spark-blur-cx]");
    const cy = panel.querySelector("[data-spark-blur-cy]");
    const cz = panel.querySelector("[data-spark-blur-cz]");
    const radiusRange = panel.querySelector("[data-spark-blur-radius-range]");
    const radiusVal = panel.querySelector("[data-spark-blur-radius-val]");
    const transRange = panel.querySelector("[data-spark-blur-trans-range]");
    const transVal = panel.querySelector("[data-spark-blur-trans-val]");
    const strengthRange = panel.querySelector("[data-spark-blur-strength-range]");
    const strengthVal = panel.querySelector("[data-spark-blur-strength-val]");
    const guideCheck = panel.querySelector("[data-spark-blur-guide-check]");
    const statusMsg = panel.querySelector("[data-spark-blur-status]");

    if (sphereOption) {
      sphereOption.disabled = !hasFloatTarget;
    }

    if (!hasFloatTarget && setting.mode === "sphere") {
      setting.mode = "camera";
    }

    if (modeSelect) {
      modeSelect.value = setting.mode;
    }

    if (fieldset) {
      fieldset.disabled = setting.mode !== "sphere";
    }

    const step = bounds ? Math.max(bounds.radius / 100, 0.001) : 0.01;

    if (cx) {
      cx.step = String(step);
      cx.value = String(setting.center[0]);
      cx.setCustomValidity("");
    }
    if (cy) {
      cy.step = String(step);
      cy.value = String(setting.center[1]);
      cy.setCustomValidity("");
    }
    if (cz) {
      cz.step = String(step);
      cz.value = String(setting.center[2]);
      cz.setCustomValidity("");
    }

    if (bounds) {
      const radiusPercent = Math.max(1, Math.round((setting.radius / bounds.radius) * 100));
      if (radiusRange) {
        if (radiusPercent > 200) {
          radiusRange.max = String(radiusPercent);
        } else {
          radiusRange.max = "200";
        }
        radiusRange.value = String(radiusPercent);
      }
      if (radiusVal) {
        radiusVal.textContent = `${setting.radius.toFixed(3)} (${radiusPercent}%)`;
      }

      const transPercent = Math.max(1, Math.round((setting.transition / bounds.radius) * 100));
      if (transRange) {
        if (transPercent > 200) {
          transRange.max = String(transPercent);
        } else {
          transRange.max = "200";
        }
        transRange.value = String(transPercent);
      }
      if (transVal) {
        transVal.textContent = `${setting.transition.toFixed(3)} (${transPercent}%)`;
      }
    }

    if (strengthRange) {
      strengthRange.value = String(setting.strength);
    }
    if (strengthVal) {
      strengthVal.textContent = `${setting.strength.toFixed(1)} px`;
    }

    if (guideCheck) {
      guideCheck.checked = Boolean(setting.showGuide);
    }

    if (statusMsg) {
      if (!hasFloatTarget) {
        statusMsg.hidden = false;
        statusMsg.textContent =
          "Sphere blur requires floating-point render targets on this device.";
      } else if (setting.invalidAuthored) {
        statusMsg.hidden = false;
        statusMsg.textContent = "Invalid background blur settings; using camera depth of field.";
      } else {
        statusMsg.hidden = true;
        statusMsg.textContent = "";
      }
    }
  }

  let blurPanelIdSeq = 0;

  function wireBlurPanel(viewer) {
    const toggle = viewer.querySelector("[data-spark-blur-toggle]");
    const panel = viewer.querySelector("[data-spark-blur-panel]");
    if (!toggle || !panel) {
      return;
    }

    blurPanelIdSeq += 1;
    const panelId = `spark-blur-panel-${blurPanelIdSeq}`;
    panel.id = panelId;
    toggle.setAttribute("aria-controls", panelId);
    toggle.setAttribute("aria-expanded", "false");

    const modeSelect = panel.querySelector("[data-spark-blur-mode-select]");
    const cx = panel.querySelector("[data-spark-blur-cx]");
    const cy = panel.querySelector("[data-spark-blur-cy]");
    const cz = panel.querySelector("[data-spark-blur-cz]");
    const radiusRange = panel.querySelector("[data-spark-blur-radius-range]");
    const radiusVal = panel.querySelector("[data-spark-blur-radius-val]");
    const transRange = panel.querySelector("[data-spark-blur-trans-range]");
    const transVal = panel.querySelector("[data-spark-blur-trans-val]");
    const strengthRange = panel.querySelector("[data-spark-blur-strength-range]");
    const strengthVal = panel.querySelector("[data-spark-blur-strength-val]");
    const guideCheck = panel.querySelector("[data-spark-blur-guide-check]");
    const resetBtn = panel.querySelector("[data-spark-blur-reset]");
    const fieldset = panel.querySelector("[data-spark-blur-sphere-fields]");

    function getActiveState() {
      const entry = resources.get(viewer);
      const url = viewer.dataset.sparkUrl;
      const map = blurSettings.get(viewer);
      const setting = map ? map.get(url) : null;
      return { entry, url, setting, bounds: entry?.bounds };
    }

    toggle.addEventListener("click", () => {
      const open = panel.hidden;
      panel.hidden = !open;
      toggle.setAttribute("aria-expanded", String(open));
      if (open && modeSelect) {
        modeSelect.focus();
      }
    });

    panel.addEventListener("pointerdown", (e) => e.stopPropagation());
    panel.addEventListener("mousedown", (e) => e.stopPropagation());
    panel.addEventListener("touchstart", (e) => e.stopPropagation(), { passive: true });
    panel.addEventListener("wheel", (e) => e.stopPropagation());
    panel.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        panel.hidden = true;
        toggle.setAttribute("aria-expanded", "false");
        toggle.focus();
      } else {
        e.stopPropagation();
      }
    });

    if (modeSelect) {
      modeSelect.addEventListener("change", () => {
        const { setting } = getActiveState();
        if (!setting) {
          return;
        }
        setting.mode = modeSelect.value;
        if (fieldset) {
          fieldset.disabled = setting.mode !== "sphere";
        }
      });
    }

    function handleCenterInput() {
      const { setting } = getActiveState();
      if (!setting || !cx || !cy || !cz) {
        return;
      }
      const inputs = [cx, cy, cz];
      const center = inputs.map((input) => input.valueAsNumber);
      inputs.forEach((input, index) => {
        input.setCustomValidity(
          Number.isFinite(center[index]) ? "" : "Please enter a valid finite number"
        );
      });
      if (center.every(Number.isFinite)) {
        setting.center = center;
      }
    }

    if (cx) {
      cx.addEventListener("input", handleCenterInput);
    }
    if (cy) {
      cy.addEventListener("input", handleCenterInput);
    }
    if (cz) {
      cz.addEventListener("input", handleCenterInput);
    }

    if (radiusRange) {
      radiusRange.addEventListener("input", () => {
        const { setting, bounds } = getActiveState();
        if (!setting || !bounds) {
          return;
        }
        const percent = Number(radiusRange.value);
        setting.radius = (percent / 100) * bounds.radius;
        if (radiusVal) {
          radiusVal.textContent = `${setting.radius.toFixed(3)} (${percent}%)`;
        }
      });
    }

    if (transRange) {
      transRange.addEventListener("input", () => {
        const { setting, bounds } = getActiveState();
        if (!setting || !bounds) {
          return;
        }
        const percent = Number(transRange.value);
        setting.transition = (percent / 100) * bounds.radius;
        if (transVal) {
          transVal.textContent = `${setting.transition.toFixed(3)} (${percent}%)`;
        }
      });
    }

    if (strengthRange) {
      strengthRange.addEventListener("input", () => {
        const { setting } = getActiveState();
        if (!setting) {
          return;
        }
        setting.strength = Number(strengthRange.value);
        if (strengthVal) {
          strengthVal.textContent = `${setting.strength.toFixed(1)} px`;
        }
      });
    }

    if (guideCheck) {
      guideCheck.addEventListener("change", () => {
        const { setting } = getActiveState();
        if (!setting) {
          return;
        }
        setting.showGuide = guideCheck.checked;
      });
    }

    if (resetBtn) {
      resetBtn.addEventListener("click", () => {
        const { setting, bounds, entry } = getActiveState();
        if (!setting || !setting.defaults || !bounds) {
          return;
        }
        setting.center = [...setting.defaults.center];
        setting.radius = setting.defaults.radius;
        setting.transition = setting.defaults.transition;
        setting.strength = setting.defaults.strength;
        setting.showGuide = setting.defaults.showGuide;
        const hasFloat = entry ? supportsFloatColorBuffer(entry.renderer) : true;
        syncBlurPanelUI(viewer, setting, bounds, hasFloat);
      });
    }
  }
  // One splat downloads and compiles at a time: fifteen figures entering the
  // viewport together would otherwise contend for bandwidth and GPU memory.
  const loadQueue = [];
  const onScreen = new WeakSet();
  let pumping = false;

  function disposeViewer(viewer) {
    const entry = resources.get(viewer);
    resources.delete(viewer);
    if (entry) {
      entry.renderer.setAnimationLoop(null);
      if (entry.resizeObserver) {
        entry.resizeObserver.disconnect();
      }
      entry.controls.dispose();
      entry.splat.dispose();
      entry.spark.dispose();
      if (entry.innerGuide) {
        entry.innerGuide.material.dispose();
      }
      if (entry.outerGuide) {
        entry.outerGuide.material.dispose();
      }
      if (entry.circleGeometry) {
        entry.circleGeometry.dispose();
      }
      if (entry.liveBlurTarget && entry.liveBlurTarget.current) {
        entry.liveBlurTarget.current.dispose();
        entry.liveBlurTarget.current = null;
      }
      entry.renderer.dispose();
      if (entry.renderer.forceContextLoss) {
        entry.renderer.forceContextLoss();
      }
      entry.canvas.remove();
    }
    const panel = viewer.querySelector("[data-spark-blur-panel]");
    if (panel) {
      panel.hidden = true;
    }
    const toggle = viewer.querySelector("[data-spark-blur-toggle]");
    if (toggle) {
      toggle.setAttribute("aria-expanded", "false");
    }
    setState(viewer, "idle", "");
  }

  function enqueueLoad(viewer) {
    if (!loadQueue.includes(viewer)) {
      loadQueue.push(viewer);
    }
    pumpQueue();
  }

  function pumpQueue() {
    if (pumping || loadQueue.length === 0) {
      return;
    }
    pumping = true;
    const viewer = loadQueue.shift();
    void loadViewer(viewer).finally(() => {
      pumping = false;
      pumpQueue();
    });
  }

  async function loadViewer(viewer) {
    const currentState = viewer.dataset.sparkState;
    if (currentState === "loading" || currentState === "ready") {
      return;
    }

    if (!supportsWebGL2()) {
      setState(
        viewer,
        "fallback",
        "WebGL 2 is unavailable in this browser. The poster is shown instead."
      );
      return;
    }

    const url = viewer.dataset.sparkUrl;
    if (!url) {
      setState(viewer, "error", "No splat URL was configured for this viewer.");
      return;
    }

    setState(viewer, "loading", "Loading interactive view…");

    let renderer;
    let controls;
    let spark;
    let splat;
    let resizeObserver;
    let canvas;
    let innerGuide;
    let outerGuide;
    let circleGeometry;
    const liveBlurTarget = { current: null };

    try {
      const { OrbitControls, SparkRenderer, SplatMesh, THREE } = await loadModules();
      const stage = viewer.querySelector(".spark-viewer__stage");
      const alt = viewer.dataset.sparkAlt || "Interactive 3D Gaussian splat";
      const focalDistance = Math.max(
        readNumber(viewer.dataset.sparkFocalDistance, DEFAULT_FOCAL_DISTANCE),
        0.1
      );
      const apertureAngle = Math.max(
        readNumber(viewer.dataset.sparkApertureAngle, DEFAULT_APERTURE_ANGLE),
        0
      );

      canvas = createCanvas(viewer, alt);
      renderer = new THREE.WebGLRenderer({
        antialias: false,
        canvas,
        powerPreference: "high-performance",
      });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setClearColor(0x171614, 1);

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
      spark = new SparkRenderer({
        apertureAngle,
        focalDistance,
        renderer,
      });
      scene.add(spark);

      const sphereUniforms = installSphereBlur(THREE, spark);

      circleGeometry = createCircleGeometry(THREE, 64);
      innerGuide = createWireSphereGuide(THREE, circleGeometry, 0xe6b86a, 0.85);
      outerGuide = createWireSphereGuide(THREE, circleGeometry, 0xf2eadc, 0.4);
      innerGuide.group.visible = false;
      outerGuide.group.visible = false;
      scene.add(innerGuide.group);
      scene.add(outerGuide.group);

      splat = new SplatMesh({
        onProgress: (event) => updateProgress(viewer, event),
        url,
      });
      scene.add(splat);

      controls = new OrbitControls(camera, canvas);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;

      await splat.initialized;
      const bounds = frameSplat(THREE, splat, camera, controls, focalDistance);

      const hasFloatTarget = supportsFloatColorBuffer(renderer);
      const blurSetting = resolveBlurSetting(viewer, url, bounds);

      const sourceCenterVec = new THREE.Vector3();
      const worldCenterVec = new THREE.Vector3();
      const viewCenterVec = new THREE.Vector3();

      syncBlurPanelUI(viewer, blurSetting, bounds, hasFloatTarget);

      const resize = () => {
        const width = Math.max(stage.clientWidth, 1);
        const height = Math.max(stage.clientHeight, 1);
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();

        if (liveBlurTarget.current) {
          const drawingSize = new THREE.Vector2();
          renderer.getDrawingBufferSize(drawingSize);
          const valid = liveBlurTarget.current.setSize(
            Math.max(Math.floor(drawingSize.x), 1),
            Math.max(Math.floor(drawingSize.y), 1)
          );
          if (!valid) {
            liveBlurTarget.current.dispose();
            liveBlurTarget.current = null;
            blurSetting.mode = "camera";
            syncBlurPanelUI(viewer, blurSetting, bounds, false);
          }
        }
      };

      resize();
      if (window.ResizeObserver) {
        resizeObserver = new ResizeObserver(resize);
        resizeObserver.observe(stage);
      } else {
        window.addEventListener("resize", resize);
      }

      const panel = viewer.querySelector("[data-spark-blur-panel]");

      renderer.setAnimationLoop(function animate() {
        controls.update();

        const map = blurSettings.get(viewer);
        const currentSetting = (map && map.get(url)) || blurSetting;

        if (currentSetting.mode === "camera") {
          spark.apertureAngle = apertureAngle;
          spark.focalDistance = camera.position.distanceTo(controls.target);
          sphereUniforms.sphereBlurEnabled.value = false;
        } else if (currentSetting.mode === "sphere") {
          spark.apertureAngle = 0;
          sphereUniforms.sphereBlurEnabled.value = currentSetting.strength > 0;
        } else {
          // "off"
          spark.apertureAngle = 0;
          sphereUniforms.sphereBlurEnabled.value = false;
        }

        if (currentSetting.mode === "sphere") {
          splat.updateMatrixWorld(true);
          sourceCenterVec.set(
            currentSetting.center[0],
            currentSetting.center[1],
            currentSetting.center[2]
          );
          worldCenterVec.copy(sourceCenterVec).applyMatrix4(splat.matrixWorld);

          const worldRadius = currentSetting.radius * splat.scale.x;
          const worldTransition = currentSetting.transition * splat.scale.x;

          camera.updateMatrixWorld();
          viewCenterVec.copy(worldCenterVec).applyMatrix4(camera.matrixWorldInverse);

          sphereUniforms.sphereBlurCenterView.value.copy(viewCenterVec);
          sphereUniforms.sphereBlurRadius.value = worldRadius;
          sphereUniforms.sphereBlurTransition.value = Math.max(worldTransition, 1e-5);
          sphereUniforms.sphereBlurSigma.value =
            currentSetting.strength * renderer.getPixelRatio() * (spark.focalAdjustment || 1.0);

          const panelOpen = panel && !panel.hidden;
          const showGuides = Boolean(panelOpen && currentSetting.showGuide);
          innerGuide.group.visible = showGuides;
          outerGuide.group.visible = showGuides;
          if (showGuides) {
            innerGuide.group.position.copy(worldCenterVec);
            innerGuide.group.scale.setScalar(Math.max(worldRadius, 1e-4));

            outerGuide.group.position.copy(worldCenterVec);
            outerGuide.group.scale.setScalar(Math.max(worldRadius + worldTransition, 1e-4));
          }
        } else {
          innerGuide.group.visible = false;
          outerGuide.group.visible = false;
        }

        const useFloatTarget = Boolean(
          currentSetting.mode === "sphere" && currentSetting.strength > 0
        );

        if (useFloatTarget) {
          if (!liveBlurTarget.current) {
            liveBlurTarget.current = createBlurTarget(THREE, renderer);
            if (!liveBlurTarget.current) {
              currentSetting.mode = "camera";
              syncBlurPanelUI(viewer, currentSetting, bounds, false);
              renderer.setRenderTarget(null);
              renderer.render(scene, camera);
              return;
            }
          }
          renderer.setRenderTarget(liveBlurTarget.current.target);
          renderer.render(scene, camera);
          renderer.setRenderTarget(null);
          renderer.render(liveBlurTarget.current.copyScene, liveBlurTarget.current.copyCamera);
        } else {
          renderer.setRenderTarget(null);
          renderer.render(scene, camera);
        }
      });

      resources.set(viewer, {
        renderer,
        controls,
        spark,
        splat,
        resizeObserver,
        canvas,
        bounds,
        innerGuide,
        outerGuide,
        circleGeometry,
        liveBlurTarget,
      });
      setState(viewer, "ready", "");
      // The viewer may have scrolled out of the frame while its splat was
      // still downloading; discard it instead of rendering offscreen.
      if (viewer.dataset.sparkUnload === "true" || !onScreen.has(viewer)) {
        disposeViewer(viewer);
      }
    } catch (error) {
      console.error("spark-viewer.js: failed to load splat viewer", error);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
      if (controls) {
        controls.dispose();
      }
      if (splat) {
        splat.dispose();
      }
      if (spark) {
        spark.dispose();
      }
      if (innerGuide) {
        innerGuide.material.dispose();
      }
      if (outerGuide) {
        outerGuide.material.dispose();
      }
      if (circleGeometry) {
        circleGeometry.dispose();
      }
      if (liveBlurTarget.current) {
        liveBlurTarget.current.dispose();
        liveBlurTarget.current = null;
      }
      if (renderer) {
        renderer.setAnimationLoop(null);
        renderer.dispose();
      }
      if (canvas) {
        canvas.remove();
      }
      setState(viewer, "error", "The interactive view could not be loaded.");
    }
  }

  function wireFullscreen(viewer) {
    const button = viewer.querySelector("[data-spark-fullscreen]");
    if (!button) {
      return;
    }
    button.addEventListener("click", () => {
      const stage = viewer.querySelector(".spark-viewer__stage");
      const current = document.fullscreenElement || document.webkitFullscreenElement;
      if (current === stage) {
        const exit = document.exitFullscreen || document.webkitExitFullscreen;
        if (exit) {
          exit.call(document);
        }
        return;
      }
      const request = stage.requestFullscreen || stage.webkitRequestFullscreen;
      if (request) {
        request.call(stage);
      }
    });
  }

  function wireToggle(viewer) {
    const button = viewer.querySelector("[data-spark-toggle]");
    if (!button || !viewer.dataset.sparkUrlUnfiltered) {
      return;
    }
    button.hidden = false;
    button.addEventListener("click", () => {
      const showing = viewer.dataset.sparkShowing === "unfiltered";
      const nextUrl = showing ? viewer.dataset.sparkFilteredUrl : viewer.dataset.sparkUrlUnfiltered;
      viewer.dataset.sparkShowing = showing ? "filtered" : "unfiltered";
      viewer.dataset.sparkUrl = nextUrl;
      button.textContent = showing ? "Show unfiltered" : "Show filtered";
      viewer.dataset.sparkUnload = "false";
      disposeViewer(viewer);
      enqueueLoad(viewer);
    });
  }

  function init() {
    document.querySelectorAll("[data-spark-viewer]").forEach((viewer) => {
      if (viewer.dataset.sparkInitialized === "true") {
        return;
      }
      viewer.dataset.sparkInitialized = "true";
      viewer.dataset.sparkFilteredUrl = viewer.dataset.sparkUrl;
      viewer.dataset.sparkShowing = "filtered";

      const observer =
        "IntersectionObserver" in window
          ? new IntersectionObserver(
              (entries) => {
                entries.forEach((entry) => {
                  if (entry.isIntersecting) {
                    onScreen.add(viewer);
                    viewer.dataset.sparkUnload = "false";
                    if (
                      viewer.dataset.sparkState === "idle" ||
                      viewer.dataset.sparkState === "error"
                    ) {
                      enqueueLoad(viewer);
                    }
                  } else {
                    onScreen.delete(viewer);
                    if (viewer.dataset.sparkState === "ready") {
                      disposeViewer(viewer);
                    } else if (viewer.dataset.sparkState === "loading") {
                      // Finish the download bookkeeping, then discard: the
                      // ready branch checks this flag.
                      viewer.dataset.sparkUnload = "true";
                    }
                  }
                });
              },
              { rootMargin: "240px" }
            )
          : null;

      const loadButton = viewer.querySelector("[data-spark-load]");
      if (loadButton) {
        loadButton.addEventListener("click", () => {
          onScreen.add(viewer);
          viewer.dataset.sparkUnload = "false";
          enqueueLoad(viewer);
        });
      }

      wireFullscreen(viewer);
      wireToggle(viewer);

      wireBlurPanel(viewer);
      if (observer) {
        observer.observe(viewer);
      } else {
        // No IntersectionObserver: load on demand only, never unload.
        onScreen.add(viewer);
        enqueueLoad(viewer);
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
