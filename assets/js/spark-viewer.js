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
      entry.renderer.dispose();
      if (entry.renderer.forceContextLoss) {
        entry.renderer.forceContextLoss();
      }
      entry.canvas.remove();
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

      splat = new SplatMesh({
        onProgress: (event) => updateProgress(viewer, event),
        url,
      });
      scene.add(splat);

      controls = new OrbitControls(camera, canvas);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;

      await splat.initialized;
      frameSplat(THREE, splat, camera, controls, focalDistance);

      const resize = () => {
        const width = Math.max(stage.clientWidth, 1);
        const height = Math.max(stage.clientHeight, 1);
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      };

      resize();
      if (window.ResizeObserver) {
        resizeObserver = new ResizeObserver(resize);
        resizeObserver.observe(stage);
      } else {
        window.addEventListener("resize", resize);
      }

      renderer.setAnimationLoop(function animate() {
        controls.update();
        // Keep the depth-of-field focal plane on the orbit target so the
        // subject stays sharp while zooming; the background stays blurred.
        spark.focalDistance = camera.position.distanceTo(controls.target);
        renderer.render(scene, camera);
      });

      resources.set(viewer, { renderer, controls, spark, splat, resizeObserver, canvas });
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
          viewer.dataset.sparkUnload = "false";
          enqueueLoad(viewer);
        });
      }

      wireFullscreen(viewer);
      wireToggle(viewer);

      if (observer) {
        observer.observe(viewer);
      } else {
        // No IntersectionObserver: load on demand only, never unload.
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
