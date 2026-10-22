(() => {
  "use strict";

  const root = document.documentElement;
  const body = document.body;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)");
  const loader = document.querySelector(".loading-screen");
  const header = document.querySelector("[data-header]");
  const progress = document.querySelector(".scroll-progress span");
  const parallaxNodes = [...document.querySelectorAll("[data-parallax]")];
  const lightSections = [...document.querySelectorAll(".manifesto, .numbers, .services")];

  document.querySelectorAll("[data-reveal-delay]").forEach((element) => {
    element.style.setProperty("--reveal-delay", `${element.dataset.revealDelay}ms`);
  });

  if (!reducedMotion.matches) {
    root.classList.add("js-motion-ready");
  }

  const showTitle = () => {
    document.querySelector(".hero-title")?.classList.add("is-visible");
  };

  window.addEventListener(
    "load",
    () => {
      window.setTimeout(() => {
        loader?.classList.add("is-done");
        showTitle();
      }, reducedMotion.matches ? 0 : 1500);
    },
    { once: true },
  );

  if (document.readyState === "complete") {
    window.setTimeout(showTitle, reducedMotion.matches ? 0 : 1500);
  }

  const revealTargets = [
    ...document.querySelectorAll("[data-reveal]"),
    ...document.querySelectorAll(".manifesto h2, .contact-title"),
  ];

  if (!reducedMotion.matches && "IntersectionObserver" in window) {
    const revealObserver = new IntersectionObserver(
      (entries, observer) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -7%" },
    );
    revealTargets.forEach((element) => revealObserver.observe(element));
  } else {
    revealTargets.forEach((element) => element.classList.add("is-visible"));
  }

  const counterObserver = "IntersectionObserver" in window
    ? new IntersectionObserver(
      (entries, observer) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          animateCounter(entry.target);
          observer.unobserve(entry.target);
        });
      },
      { threshold: 0.5 },
    )
    : null;

  function animateCounter(element) {
    const target = Number(element.dataset.count || 0);
    if (reducedMotion.matches) {
      element.textContent = String(target);
      return;
    }
    const duration = 1280;
    const startAt = performance.now();
    const update = (now) => {
      const elapsed = Math.min((now - startAt) / duration, 1);
      const eased = 1 - Math.pow(1 - elapsed, 4);
      element.textContent = String(Math.round(target * eased));
      if (elapsed < 1) requestAnimationFrame(update);
    };
    requestAnimationFrame(update);
  }

  document.querySelectorAll("[data-count]").forEach((counter) => {
    if (counterObserver) counterObserver.observe(counter);
    else animateCounter(counter);
  });

  let frameQueued = false;
  const updateViewportState = () => {
    const scrollY = window.scrollY || window.pageYOffset;
    const total = document.documentElement.scrollHeight - window.innerHeight;
    if (progress) progress.style.width = `${Math.min(100, Math.max(0, (scrollY / total) * 100))}%`;

    header?.classList.toggle("is-scrolled", scrollY > 22);
    const headerLine = 74;
    const overLight = lightSections.some((section) => {
      const bounds = section.getBoundingClientRect();
      return bounds.top <= headerLine && bounds.bottom > headerLine;
    });
    header?.classList.toggle("is-light", overLight);

    if (!reducedMotion.matches) {
      const viewportCenter = window.innerHeight / 2;
      parallaxNodes.forEach((node) => {
        const rect = node.getBoundingClientRect();
        if (rect.bottom < -150 || rect.top > window.innerHeight + 150) return;
        const center = rect.top + rect.height / 2;
        const normalized = (center - viewportCenter) / window.innerHeight;
        const strength = Number(node.dataset.parallax || 0.05);
        const y = -normalized * strength * 270;
        node.style.setProperty("--parallax-y", `${y.toFixed(2)}px`);
        if (node.classList.contains("horizon-pane")) {
          node.style.setProperty("--parallax-scale", (1.02 - Math.min(Math.abs(normalized), 1) * 0.035).toFixed(3));
        }
      });
    }
    frameQueued = false;
  };

  const requestViewportUpdate = () => {
    if (frameQueued) return;
    frameQueued = true;
    requestAnimationFrame(updateViewportState);
  };

  window.addEventListener("scroll", requestViewportUpdate, { passive: true });
  window.addEventListener("resize", requestViewportUpdate, { passive: true });
  requestViewportUpdate();

  const setPointerFeatures = () => {
    const enabled = finePointer.matches && !reducedMotion.matches;
    body.classList.toggle("has-pointer", enabled);
    if (!enabled) {
      root.style.removeProperty("--pointer-x");
      root.style.removeProperty("--pointer-y");
    }
  };
  setPointerFeatures();
  finePointer.addEventListener?.("change", setPointerFeatures);
  reducedMotion.addEventListener?.("change", () => {
    setPointerFeatures();
    requestViewportUpdate();
  });

  let pointerFrame = false;
  let pointerX = 0;
  let pointerY = 0;
  window.addEventListener(
    "pointermove",
    (event) => {
      if (!finePointer.matches || reducedMotion.matches) return;
      pointerX = event.clientX;
      pointerY = event.clientY;
      if (pointerFrame) return;
      pointerFrame = true;
      requestAnimationFrame(() => {
        root.style.setProperty("--pointer-x", `${pointerX}px`);
        root.style.setProperty("--pointer-y", `${pointerY}px`);
        pointerFrame = false;
      });
    },
    { passive: true },
  );

  const magneticItems = [...document.querySelectorAll("[data-magnetic]")];
  magneticItems.forEach((item) => {
    item.addEventListener("pointermove", (event) => {
      if (!finePointer.matches || reducedMotion.matches) return;
      const rect = item.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width - 0.5) * 11;
      const y = ((event.clientY - rect.top) / rect.height - 0.5) * 9;
      item.style.setProperty("--mag-x", `${x.toFixed(2)}px`);
      item.style.setProperty("--mag-y", `${y.toFixed(2)}px`);
    });
    item.addEventListener("pointerleave", () => {
      item.style.setProperty("--mag-x", "0px");
      item.style.setProperty("--mag-y", "0px");
    });
  });

  document.querySelectorAll("[data-tilt]").forEach((card) => {
    card.addEventListener("pointermove", (event) => {
      if (!finePointer.matches || reducedMotion.matches) return;
      const rect = card.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width;
      const y = (event.clientY - rect.top) / rect.height;
      const rotateY = (x - 0.5) * 3.6;
      const rotateX = (0.5 - y) * 3.1;
      card.style.setProperty("--card-x", `${(x * 100).toFixed(1)}%`);
      card.style.setProperty("--card-y", `${(y * 100).toFixed(1)}%`);
      card.style.willChange = "transform";
      card.style.transform = `perspective(950px) rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg) translate3d(0, -3px, 0)`;
    });
    card.addEventListener("pointerleave", () => {
      card.style.transform = "";
      card.style.willChange = "";
    });
  });

  const menuButton = document.querySelector("[data-menu-button]");
  const mobileMenu = document.querySelector("[data-mobile-menu]");
  const toggleMenu = (force) => {
    if (!menuButton || !mobileMenu) return;
    const open = typeof force === "boolean" ? force : !mobileMenu.classList.contains("is-open");
    mobileMenu.classList.toggle("is-open", open);
    menuButton.classList.toggle("is-open", open);
    menuButton.setAttribute("aria-expanded", String(open));
    menuButton.setAttribute("aria-label", open ? "Fechar menu" : "Abrir menu");
    mobileMenu.setAttribute("aria-hidden", String(!open));
    body.classList.toggle("menu-open", open);
  };
  menuButton?.addEventListener("click", () => toggleMenu());
  mobileMenu?.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => toggleMenu(false)));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") toggleMenu(false);
  });

  document.querySelectorAll("[data-service]").forEach((service) => {
    service.setAttribute("aria-pressed", "false");
    service.addEventListener("click", () => {
      const wasActive = service.classList.contains("is-active");
      document.querySelectorAll("[data-service]").forEach((row) => {
        row.classList.remove("is-active");
        row.setAttribute("aria-pressed", "false");
      });
      service.classList.toggle("is-active", !wasActive);
      service.setAttribute("aria-pressed", String(!wasActive));
    });
  });

  function initSmoothScroll() {
    if (reducedMotion.matches || !finePointer.matches) return;
    root.style.scrollBehavior = "auto";
    let current = window.scrollY;
    let target = current;
    let rafId = 0;
    let last = 0;
    const maxScroll = () => Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
    const clamp = (value) => Math.max(0, Math.min(value, maxScroll()));

    const tick = (now) => {
      const dt = Math.min(now - last, 64);
      last = now;
      const diff = target - current;
      if (Math.abs(diff) < 0.35) {
        current = target;
        window.scrollTo(0, current);
        rafId = 0;
        return;
      }
      current += diff * (1 - Math.exp(-dt / 130));
      window.scrollTo(0, current);
      rafId = requestAnimationFrame(tick);
    };
    const run = () => {
      if (rafId) return;
      last = performance.now();
      rafId = requestAnimationFrame(tick);
    };
    const resync = () => {
      if (!rafId) current = target = window.scrollY;
    };

    const scrollsInside = (node, deltaY) => {
      for (let el = node; el && el !== document.body && el !== root; el = el.parentElement) {
        const style = getComputedStyle(el);
        if (!/(auto|scroll)/.test(style.overflowY) || el.scrollHeight <= el.clientHeight) continue;
        const atTop = el.scrollTop <= 0;
        const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 1;
        if ((deltaY < 0 && !atTop) || (deltaY > 0 && !atBottom)) return true;
      }
      return false;
    };

    window.addEventListener(
      "wheel",
      (event) => {
        if (event.ctrlKey || event.defaultPrevented || body.classList.contains("menu-open")) return;
        if (scrollsInside(event.target, event.deltaY)) return;
        event.preventDefault();
        resync();
        const unit = event.deltaMode === 1 ? 32 : event.deltaMode === 2 ? window.innerHeight : 1;
        target = clamp(target + event.deltaY * unit);
        run();
      },
      { passive: false },
    );

    document.addEventListener("click", (event) => {
      const link = event.target.closest?.('a[href^="#"]');
      if (!link || event.defaultPrevented || event.metaKey || event.ctrlKey || event.shiftKey) return;
      const hash = link.getAttribute("href");
      if (hash === "#") return;
      const section = document.querySelector(hash);
      if (!section) return;
      event.preventDefault();
      resync();
      target = clamp(section.getBoundingClientRect().top + window.scrollY);
      history.replaceState(null, "", hash);
      run();
    });

    window.addEventListener("keydown", (event) => {
      if (["Home", "End", "PageUp", "PageDown", " ", "ArrowUp", "ArrowDown"].includes(event.key)) {
        if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
      }
    });
  }

  // Desfoque direcional: alonga os detalhes no eixo vertical, acompanhando o scroll.
  // Um único GaussianBlur (sem turbulence/displacement) mantém o custo bem menor.
  function initScrollDistortion() {
    if (reducedMotion.matches || !finePointer.matches || root.classList.contains("effect-blur")) return;
    const sections = [...document.querySelectorAll("main > section")];
    const blurNode = document.querySelector("#scroll-directional-blur");
    if (!sections.length || !blurNode) return;

    const MAX_STRETCH = 10.5;      // mantém o rastro forte em scroll rápido
    const SPEED_DEADZONE = 155;    // ignora micro-scrolls e ruído do trackpad
    const SPEED_FOR_MAX = 1050;    // só acelera bastante quando o scroll é real
    let lastY = window.scrollY;
    let lastT = performance.now();
    let lastScroll = 0;
    let intensity = 0;
    let rafId = 0;

    const clearAll = () => {
      blurNode.setAttribute("stdDeviation", "0 0");
      sections.forEach((section) => {
        section.style.filter = "";
        section.style.willChange = "";
      });
    };

    const step = (now) => {
      if (root.classList.contains("effect-blur")) { clearAll(); rafId = 0; return; }
      const dt = Math.max(0.008, Math.min((now - lastT) / 1000, 0.05));
      const y = window.scrollY;
      const speed = Math.abs(y - lastY) / dt;
      lastY = y;
      lastT = now;

      // Abaixo do limite, não há blur. Acima dele, a intensidade cresce
      // progressivamente; scroll rápido continua chegando ao efeito máximo.
      const speedAmount = Math.max(0, Math.min((speed - SPEED_DEADZONE) / (SPEED_FOR_MAX - SPEED_DEADZONE), 1));
      const target = speedAmount <= 0 ? 0 : Math.pow(speedAmount, 0.72);
      intensity += (target - intensity) * (target > intensity ? 0.88 : 0.13);
      const stretch = intensity * MAX_STRETCH;

      if (now - lastScroll > 170 && intensity < 0.012 && speed < 5) {
        clearAll();
        rafId = 0;
        return;
      }

      // Blur quase exclusivamente vertical: cria rastros na direção do movimento,
      // em vez de simplesmente desfocar igualmente todos os lados.
      blurNode.setAttribute("stdDeviation", `0 ${stretch.toFixed(2)}`);
      sections.forEach((section) => {
        const rect = section.getBoundingClientRect();
        const visible = rect.bottom > -80 && rect.top < window.innerHeight + 80;
        if (!visible || stretch < 0.08) {
          section.style.filter = "";
          section.style.willChange = "";
          return;
        }
        section.style.willChange = "filter";
        section.style.filter = "url(#scroll-directional-filter)";
      });
      rafId = requestAnimationFrame(step);
    };

    window.addEventListener("scroll", () => {
      lastScroll = performance.now();
      if (rafId) return;
      lastT = lastScroll;
      lastY = window.scrollY;
      rafId = requestAnimationFrame(step);
    }, { passive: true });
  }

  function initNebula() {
    const canvas = document.querySelector("#nebula-canvas");
    if (!canvas || reducedMotion.matches) return;
    const context = canvas.getContext("2d", { alpha: true });
    if (!context) return;

    let width = 0;
    let height = 0;
    let particles = [];
    const maxDpr = Math.min(window.devicePixelRatio || 1, 1.4);

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = Math.floor(width * maxDpr);
      canvas.height = Math.floor(height * maxDpr);
      context.setTransform(maxDpr, 0, 0, maxDpr, 0, 0);
      const amount = width < 620 ? 17 : 31;
      particles = Array.from({ length: amount }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        radius: Math.random() * 1.35 + 0.2,
        drift: Math.random() * 0.16 + 0.03,
        phase: Math.random() * Math.PI * 2,
        tint: Math.random() > 0.71 ? "166,213,255" : "180,140,255",
      }));
    };

    const draw = (time) => {
      context.clearRect(0, 0, width, height);
      const seconds = time * 0.001;
      particles.forEach((particle, index) => {
        const px = particle.x + Math.sin(seconds * particle.drift + particle.phase) * 14;
        const py = particle.y + Math.cos(seconds * particle.drift * .7 + particle.phase) * 20;
        context.beginPath();
        context.fillStyle = `rgba(${particle.tint}, ${0.13 + Math.sin(seconds + particle.phase) * 0.05})`;
        context.arc(px, py, particle.radius, 0, Math.PI * 2);
        context.fill();

        const next = particles[(index + 3) % particles.length];
        const nx = next.x + Math.sin(seconds * next.drift + next.phase) * 14;
        const ny = next.y + Math.cos(seconds * next.drift * .7 + next.phase) * 20;
        const distance = Math.hypot(px - nx, py - ny);
        if (distance < 160) {
          context.beginPath();
          context.strokeStyle = `rgba(205, 175, 255, ${0.028 * (1 - distance / 160)})`;
          context.lineWidth = .5;
          context.moveTo(px, py);
          context.lineTo(nx, ny);
          context.stroke();
        }
      });
      requestAnimationFrame(draw);
    };

    resize();
    window.addEventListener("resize", resize, { passive: true });
    requestAnimationFrame(draw);
  }

  function initGalaxies() {
    document.querySelectorAll("canvas[data-galaxy]").forEach((canvas) => {
      const size = 900;
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const c = size / 2;
      const arms = 2;
      const draw = (count, spread, sizeMax, alpha) => {
        for (let i = 0; i < count; i++) {
          const r = Math.pow(Math.random(), 0.62) * c * 0.94;
          const arm = (i % arms) * Math.PI;
          const angle = arm + (r / c) * 6.3 + (Math.random() - 0.5) * spread * (1.2 - r / c * 0.5);
          const jitter = (Math.random() - 0.5) * 16 * (1 - r / c * 0.5);
          const x = c + Math.cos(angle) * r + jitter;
          const y = c + Math.sin(angle) * r + jitter;
          const t = r / c;
          const col = t < 0.25 ? "255,244,225" : Math.random() > 0.55 ? "166,213,255" : "190,150,255";
          ctx.fillStyle = `rgba(${col},${(alpha * (1 - t * 0.55) * (0.5 + Math.random() * 0.5)).toFixed(3)})`;
          ctx.beginPath();
          ctx.arc(x, y, Math.random() * sizeMax + 0.35, 0, Math.PI * 2);
          ctx.fill();
        }
      };
      ctx.globalCompositeOperation = "lighter";
      draw(2600, 1.1, 5, 0.1);
      draw(5200, 0.55, 1.8, 0.7);
      draw(900, 2.4, 1.2, 0.5);
    });
  }
  document.querySelectorAll("img.photo[data-hides]").forEach((photo) => {
    const show = () => {
      if (!photo.naturalWidth) return;
      photo.classList.add("is-loaded");
      const scope = photo.parentElement;
      photo.dataset.hides.split(",").forEach((selector) => {
        scope.querySelectorAll(selector.trim()).forEach((node) => { node.style.display = "none"; });
      });
    };
    if (photo.complete) show();
    else photo.addEventListener("load", show, { once: true });
    photo.addEventListener("error", () => photo.remove(), { once: true });
  });

  function initPerformanceControls() {
    const tools = document.querySelector("[data-performance-tools]");
    if (!tools) return;

    const toggle = tools.querySelector(".performance-toggle");
    const panel = tools.querySelector(".performance-panel");
    const reset = tools.querySelector("[data-effects-reset]");
    const controls = [...tools.querySelectorAll("input[data-effect]")];
    const storageKey = "abrobagpt-performance-effects-v1";
    const effects = ["nebula", "galaxy", "noise", "glow", "blur"];

    const apply = (effect, enabled) => {
      root.classList.toggle(`effect-${effect}`, !enabled);
      if (effect === "blur" && !enabled) {
        const blurNode = document.querySelector("#scroll-directional-blur");
        blurNode?.setAttribute("stdDeviation", "0 0");
        document.querySelectorAll("main > section").forEach((section) => {
          section.style.filter = "";
          section.style.willChange = "";
        });
      }
    };

    const save = () => {
      const settings = {};
      controls.forEach((control) => {
        settings[control.dataset.effect] = control.checked;
      });
      try { localStorage.setItem(storageKey, JSON.stringify(settings)); } catch (_) { /* armazenamento indisponível */ }
    };

    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(storageKey) || "{}"); } catch (_) { saved = {}; }
    controls.forEach((control) => {
      const effect = control.dataset.effect;
      if (effects.includes(effect) && typeof saved[effect] === "boolean") control.checked = saved[effect];
      apply(effect, control.checked);
      control.addEventListener("change", () => {
        apply(effect, control.checked);
        save();
      });
    });

    toggle?.addEventListener("click", () => {
      const opening = toggle.getAttribute("aria-expanded") !== "true";
      toggle.setAttribute("aria-expanded", String(opening));
      if (panel) panel.hidden = !opening;
      const symbol = toggle.querySelector("span");
      if (symbol) symbol.textContent = opening ? "−" : "＋";
    });

    reset?.addEventListener("click", () => {
      controls.forEach((control) => {
        control.checked = true;
        apply(control.dataset.effect, true);
      });
      save();
    });
  }

  initPerformanceControls();
  initSmoothScroll();
  initScrollDistortion();

  initGalaxies();

  initNebula();
})();
