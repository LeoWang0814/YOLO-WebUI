(() => {
  const preferenceKey = "yolov10-workbench.theme";
  const root = document.documentElement;
  let feedbackTimer;
  const pendingLogScroll = new Map();
  const t = (value) => window.WorkbenchI18n?.t(value) || value;
  const actualTheme = (preference) => preference === "system"
    ? (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
    : preference;

  const applyTheme = (preference) => {
    const theme = actualTheme(preference);
    root.dataset.theme = theme;
    root.dataset.themePreference = preference;
    updateThemeToggle(theme);
    document.querySelectorAll("[data-plot]").forEach((chart) => { delete chart.dataset.plotReady; });
    const results = document.querySelector("#run-results[hx-get]");
    if (results && window.htmx) window.htmx.trigger(results, "load");
  };

  const renderCharts = (scope = document) => {
    if (!window.Plotly) return;
    scope.querySelectorAll("[data-plot]").forEach((chart) => {
      if (chart.dataset.plotReady === "true") return;
      try {
        const figure = JSON.parse(chart.dataset.plot);
        window.Plotly.react(chart, figure.data || [], figure.layout || {}, {
          displayModeBar: false,
          responsive: true,
        });
        chart.dataset.plotReady = "true";
      } catch (_) {
        chart.textContent = t("Unable to render metrics.");
      }
    });
  };

  const initializeIcons = () => {
    if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
  };

  const updateThemeToggle = (theme) => {
    const toggle = document.querySelector("[data-theme-toggle]");
    if (!toggle) return;
    const nextTheme = theme === "dark" ? "light" : "dark";
    toggle.setAttribute("aria-label", t(`Switch to ${nextTheme} theme`));
    toggle.setAttribute("title", t(`Switch to ${nextTheme} theme`));
    toggle.innerHTML = `<i data-lucide="${theme === "dark" ? "sun" : "moon"}"></i>`;
    initializeIcons();
  };

  const refreshConditionals = (scope = document) => {
    scope.querySelectorAll("[data-conditional]").forEach((element) => {
      const [name, expected] = element.dataset.conditional.split(":");
      const selected = document.querySelector(`[name="${name}"]:checked`) || document.querySelector(`[name="${name}"]`);
      const visible = Boolean(selected && selected.value === expected);
      element.hidden = !visible;
      const controls = element.matches("input, select, textarea, fieldset")
        ? [element]
        : element.querySelectorAll("input, select, textarea, fieldset");
      controls.forEach((control) => { control.disabled = !visible; });
    });
  };

  const initializeSearch = (scope = document) => {
    scope.querySelectorAll("[data-parameter-search]").forEach((search) => {
      if (search.dataset.bound) return;
      search.dataset.bound = "true";
      search.addEventListener("input", () => {
        const query = search.value.trim().toLowerCase();
        document.querySelectorAll("[data-parameter-name]").forEach((field) => {
          field.hidden = Boolean(query && !field.dataset.parameterName.toLowerCase().includes(query));
        });
      });
    });
  };

  const initializeFileInputs = (scope = document) => {
    scope.querySelectorAll(".upload-drop input[type='file']").forEach((input) => {
      if (input.dataset.bound) return;
      input.dataset.bound = "true";
      input.addEventListener("change", () => {
        const summary = input.closest(".upload-drop")?.querySelector("[data-file-summary]");
        if (!summary) return;
        if (!input.files?.length) summary.textContent = t(input.multiple ? "No files selected" : "No file selected");
        else if (input.files.length === 1) summary.textContent = input.files[0].name;
        else summary.textContent = t(`${input.files.length} files selected`);
      });
    });
  };

  const isLogAtBottom = (output) => output.scrollHeight - output.scrollTop - output.clientHeight <= 8;

  const updateLogFollowStatus = (output, following) => {
    output.dataset.followOutput = String(following);
    const section = output.closest("[data-log-fragment]");
    const status = section?.querySelector("[data-log-follow-status]");
    const label = status?.querySelector("span");
    if (!status || !label) return;
    status.classList.toggle("is-paused", !following);
    label.textContent = following
      ? t(section.hasAttribute("hx-get") ? "Following output" : "At latest output")
      : t("Auto-follow paused");
  };

  const captureLogScroll = (target) => {
    const section = target?.matches?.("[data-log-fragment]") ? target : target?.querySelector?.("[data-log-fragment]");
    const output = section?.querySelector("[data-log-output]");
    if (!section?.id || !output) return;
    pendingLogScroll.set(section.id, {
      following: isLogAtBottom(output),
      scrollTop: output.scrollTop,
    });
  };

  const initializeLogTerminals = () => {
    document.querySelectorAll("[data-log-fragment]").forEach((section) => {
      const output = section.querySelector("[data-log-output]");
      if (!output || output.dataset.bound) return;
      const saved = pendingLogScroll.get(section.id);
      if (saved) pendingLogScroll.delete(section.id);
      const following = saved ? saved.following : true;
      output.scrollTop = following ? output.scrollHeight : saved.scrollTop;
      updateLogFollowStatus(output, following);
      output.dataset.bound = "true";
      output.addEventListener("scroll", () => updateLogFollowStatus(output, isLogAtBottom(output)), { passive: true });
    });
  };

  const updateRunFilter = () => {
    const search = document.querySelector("[data-run-search]");
    const selected = document.querySelector("[name='run_filter']:checked");
    if (!search || !selected) return;
    const query = search.value.trim().toLowerCase();
    let visible = 0;
    document.querySelectorAll("[data-run-row]").forEach((row) => {
      const matchesKind = selected.value === "all" || row.dataset.runKind === selected.value;
      const matchesName = !query || row.dataset.runName.includes(query);
      row.hidden = !(matchesKind && matchesName);
      if (!row.hidden) visible += 1;
    });
    const empty = document.querySelector("[data-run-empty]");
    if (empty) empty.hidden = visible > 0;
  };

  const initializeRunFilters = (scope = document) => {
    const search = scope.querySelector("[data-run-search]");
    if (search && !search.dataset.bound) {
      search.dataset.bound = "true";
      search.addEventListener("input", updateRunFilter);
    }
  };

  const viewerState = {
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    dragging: false,
    pointerId: null,
    startX: 0,
    startY: 0,
    baseWidth: 0,
    baseHeight: 0,
    restoreFocus: null,
  };

  const getViewer = () => document.querySelector("#media-viewer");

  const getViewerMedia = () => {
    const viewer = getViewer();
    if (!viewer) return null;
    const image = viewer.querySelector("[data-viewer-image]");
    const video = viewer.querySelector("[data-viewer-video]");
    return image && !image.hidden ? image : (video && !video.hidden ? video : null);
  };

  const clamp = (value, minimum, maximum) => Math.min(Math.max(value, minimum), maximum);

  const getViewerStage = () => getViewer()?.querySelector("[data-viewer-stage]");

  const updateViewerInteractionState = () => {
    const stage = getViewerStage();
    if (!stage) return;
    stage.classList.toggle("is-pannable", viewerState.scale > 1 && Boolean(getViewerMedia()?.matches("img")));
  };

  const clampViewerOffset = () => {
    const stage = getViewerStage();
    if (!stage || viewerState.scale <= 1) {
      viewerState.offsetX = 0;
      viewerState.offsetY = 0;
      return;
    }
    const maxX = Math.max(0, (viewerState.baseWidth * viewerState.scale - stage.clientWidth) / 2);
    const maxY = Math.max(0, (viewerState.baseHeight * viewerState.scale - stage.clientHeight) / 2);
    viewerState.offsetX = clamp(viewerState.offsetX, -maxX, maxX);
    viewerState.offsetY = clamp(viewerState.offsetY, -maxY, maxY);
  };

  const applyViewerTransform = () => {
    const viewer = getViewer();
    const media = getViewerMedia();
    if (!viewer || !media) return;
    clampViewerOffset();
    media.style.transform = `translate3d(${viewerState.offsetX}px, ${viewerState.offsetY}px, 0) scale(${viewerState.scale})`;
    const caption = viewer.querySelector("[data-viewer-caption]");
    if (caption) caption.textContent = viewerState.scale === 1 ? t("Fit to view") : `${Math.round(viewerState.scale * 100)}%`;
    updateViewerInteractionState();
  };

  const resetViewerTransform = () => {
    viewerState.scale = 1;
    viewerState.offsetX = 0;
    viewerState.offsetY = 0;
    applyViewerTransform();
  };

  const fitViewerMedia = () => {
    const stage = getViewerStage();
    const media = getViewerMedia();
    if (!stage || !media || !stage.clientWidth || !stage.clientHeight) return;
    const sourceWidth = media.matches("img") ? media.naturalWidth : media.videoWidth;
    const sourceHeight = media.matches("img") ? media.naturalHeight : media.videoHeight;
    if (!sourceWidth || !sourceHeight) return;
    const scale = Math.min(stage.clientWidth / sourceWidth, stage.clientHeight / sourceHeight);
    viewerState.baseWidth = Math.max(1, Math.floor(sourceWidth * scale));
    viewerState.baseHeight = Math.max(1, Math.floor(sourceHeight * scale));
    media.style.width = `${viewerState.baseWidth}px`;
    media.style.height = `${viewerState.baseHeight}px`;
    applyViewerTransform();
  };

  const setViewerScale = (nextScale, point) => {
    const stage = getViewerStage();
    const previousScale = viewerState.scale;
    viewerState.scale = clamp(Number(nextScale.toFixed(2)), 1, 5);
    if (point && stage && viewerState.scale !== previousScale) {
      const relativeX = point.clientX - stage.getBoundingClientRect().left - stage.clientWidth / 2;
      const relativeY = point.clientY - stage.getBoundingClientRect().top - stage.clientHeight / 2;
      const ratio = viewerState.scale / previousScale;
      viewerState.offsetX = relativeX - (relativeX - viewerState.offsetX) * ratio;
      viewerState.offsetY = relativeY - (relativeY - viewerState.offsetY) * ratio;
    }
    applyViewerTransform();
  };

  const adjustViewerZoom = (delta, point) => setViewerScale(viewerState.scale + delta, point);

  const stopViewerDrag = (event) => {
    const stage = getViewerStage();
    if (!viewerState.dragging || (event && event.pointerId !== viewerState.pointerId)) return;
    viewerState.dragging = false;
    viewerState.pointerId = null;
    stage?.classList.remove("is-dragging");
  };

  const closeViewer = () => {
    const viewer = getViewer();
    if (!viewer || viewer.hidden) return;
    const video = viewer.querySelector("[data-viewer-video]");
    video?.pause();
    if (video) {
      video.removeAttribute("src");
      video.load();
    }
    const image = viewer.querySelector("[data-viewer-image]");
    if (image) image.removeAttribute("src");
    image?.style.removeProperty("width");
    image?.style.removeProperty("height");
    resetViewerTransform();
    stopViewerDrag();
    viewer.hidden = true;
    viewer.setAttribute("aria-hidden", "true");
    document.body.classList.remove("is-viewer-open");
    viewerState.restoreFocus?.focus();
    viewerState.restoreFocus = null;
  };

  const openViewer = (trigger) => {
    const viewer = getViewer();
    if (!viewer) return;
    const kind = trigger.dataset.mediaKind || "image";
    const src = trigger.dataset.mediaSrc || trigger.getAttribute("href");
    if (!src) return;
    const image = viewer.querySelector("[data-viewer-image]");
    const video = viewer.querySelector("[data-viewer-video]");
    const empty = viewer.querySelector("[data-viewer-empty]");
    const title = viewer.querySelector("[data-viewer-title]");
    const original = viewer.querySelector("[data-viewer-open]");
    viewerState.restoreFocus = trigger;
    viewerState.scale = 1;
    viewerState.offsetX = 0;
    viewerState.offsetY = 0;
    empty.hidden = true;
    image.hidden = kind !== "image";
    video.hidden = kind !== "video";
    if (kind === "image") {
      image.src = src;
      image.onload = () => {
        resetViewerTransform();
        fitViewerMedia();
      };
      image.onerror = () => { image.hidden = true; empty.hidden = false; };
    } else {
      video.src = src;
      video.load();
    }
    if (title) title.textContent = trigger.dataset.mediaLabel || t(kind === "video" ? "Video" : "Image");
    if (original) original.href = src;
    viewer.hidden = false;
    viewer.setAttribute("aria-hidden", "false");
    document.body.classList.add("is-viewer-open");
    resetViewerTransform();
    viewer.querySelector("[data-viewer-close]")?.focus();
  };

  const initializeViewer = () => {
    const viewer = getViewer();
    const stage = viewer?.querySelector("[data-viewer-stage]");
    if (!viewer || !stage || viewer.dataset.bound) return;
    viewer.dataset.bound = "true";
    stage.addEventListener("wheel", (event) => {
      if (viewer.hidden) return;
      event.preventDefault();
      adjustViewerZoom(event.deltaY < 0 ? 0.15 : -0.15, event);
    }, { passive: false });
    stage.addEventListener("pointerdown", (event) => {
      const image = viewer.querySelector("[data-viewer-image]");
      if (viewer.hidden || image?.hidden || viewerState.scale <= 1 || !event.isPrimary || (event.pointerType === "mouse" && event.button !== 0)) return;
      event.preventDefault();
      viewerState.dragging = true;
      viewerState.pointerId = event.pointerId;
      viewerState.startX = event.clientX - viewerState.offsetX;
      viewerState.startY = event.clientY - viewerState.offsetY;
      stage.classList.add("is-dragging");
      stage.setPointerCapture?.(event.pointerId);
    });
    stage.addEventListener("pointermove", (event) => {
      if (!viewerState.dragging || event.pointerId !== viewerState.pointerId) return;
      event.preventDefault();
      viewerState.offsetX = event.clientX - viewerState.startX;
      viewerState.offsetY = event.clientY - viewerState.startY;
      applyViewerTransform();
    });
    stage.addEventListener("pointerup", stopViewerDrag);
    stage.addEventListener("pointercancel", stopViewerDrag);
    stage.addEventListener("lostpointercapture", stopViewerDrag);
    stage.addEventListener("dblclick", (event) => {
      if (!viewer.hidden) {
        event.preventDefault();
        if (viewerState.scale === 1) setViewerScale(2, event);
        else resetViewerTransform();
      }
    });
    window.addEventListener("resize", () => {
      if (!viewer.hidden) window.requestAnimationFrame(fitViewerMedia);
    });
  };

  const showFeedback = (message, tone = "error") => {
    const region = document.querySelector("#app-feedback");
    if (!region) return;
    window.clearTimeout(feedbackTimer);
    const alert = document.createElement("div");
    alert.className = `feedback-alert is-${tone}`;
    alert.textContent = message;
    region.replaceChildren(alert);
    feedbackTimer = window.setTimeout(() => region.replaceChildren(), 6000);
  };

  const responseMessage = (xhr) => {
    try {
      const payload = JSON.parse(xhr.responseText || "{}");
      if (payload.detail) return String(payload.detail);
    } catch (_) {
      // The route returned HTML or plain text.
    }
    return `Request failed (${xhr.status || "network error"}).`;
  };

  const initializeDatasetProgress = () => {
    document.querySelectorAll("[data-dataset-job]").forEach((element) => {
      if (element.dataset.pollBound) return;
      element.dataset.pollBound = "true";
      const poll = async () => {
        if (!element.isConnected) return;
        try {
          const response = await fetch(`/fragments/dataset/prepare/${encodeURIComponent(element.dataset.datasetJob)}`, {
            headers: { "HX-Request": "true", "X-Theme": root.dataset.theme || "light" },
          });
          if (!response.ok) throw new Error("Dataset status request failed.");
          const replacement = await response.text();
          if (!element.isConnected) return;
          element.outerHTML = replacement;
          initializeScope();
        } catch (_) {
          const status = element.querySelector(".dataset-progress-copy span");
          if (status) status.textContent = t("Reconnecting to dataset preparation status…");
          window.setTimeout(poll, 900);
        }
      };
      window.setTimeout(poll, 300);
    });
  };

  const initializeDocsSearch = (scope = document) => {
    scope.querySelectorAll("[data-doc-search-root]").forEach((root) => {
      if (root.dataset.bound) return;
      root.dataset.bound = "true";

      const input = root.querySelector("[data-doc-search]");
      const results = root.querySelector("[data-doc-search-results]");
      const indexNode = root.querySelector("[data-doc-search-index]");
      if (!input || !results || !indexNode) return;

      let index = [];
      try {
        index = JSON.parse(indexNode.textContent || "[]");
      } catch (_) {
        return;
      }

      let matches = [];
      let activeIndex = -1;
      const resultId = (position) => `docs-search-result-${position}`;

      const scoreEntry = (entry, query, terms) => {
        const localizedTitle = t(entry.title);
        const localizedPageTitle = t(entry.page_title);
        const title = localizedTitle.toLowerCase();
        const pageTitle = localizedPageTitle.toLowerCase();
        const searchable = `${entry.terms} ${localizedTitle} ${localizedPageTitle}`.toLowerCase();
        if (!terms.every((term) => searchable.includes(term))) return -1;

        let score = 100;
        if (title === query) score += 900;
        else if (title.startsWith(query)) score += 700;
        else if (title.includes(query)) score += 500;
        if (pageTitle === query) score += 300;
        else if (pageTitle.startsWith(query)) score += 180;
        if (searchable.startsWith(query)) score += 80;
        return score;
      };

      const setActiveResult = (position) => {
        activeIndex = position;
        results.querySelectorAll("[data-doc-search-result]").forEach((result, indexPosition) => {
          const selected = indexPosition === activeIndex;
          result.classList.toggle("is-active", selected);
          result.setAttribute("aria-selected", String(selected));
        });
        if (activeIndex >= 0) input.setAttribute("aria-activedescendant", resultId(activeIndex));
        else input.removeAttribute("aria-activedescendant");
      };

      const renderResults = () => {
        const query = input.value.trim().toLowerCase();
        results.replaceChildren();
        activeIndex = -1;
        matches = [];
        input.setAttribute("aria-expanded", "false");
        input.removeAttribute("aria-activedescendant");
        if (!query) {
          results.hidden = true;
          return;
        }

        const terms = query.split(/\s+/).filter(Boolean);
        matches = index
          .map((entry) => ({ entry, score: scoreEntry(entry, query, terms) }))
          .filter(({ score }) => score >= 0)
          .sort((left, right) => right.score - left.score || left.entry.kind.localeCompare(right.entry.kind) || left.entry.title.localeCompare(right.entry.title))
          .slice(0, 8)
          .map(({ entry }) => entry);

        results.hidden = false;
        input.setAttribute("aria-expanded", "true");
        if (!matches.length) {
          const empty = document.createElement("p");
          empty.className = "docs-search-empty";
          empty.textContent = window.WorkbenchI18n?.isChinese()
            ? `没有与“${input.value.trim()}”匹配的文档。`
            : `No documentation matches “${input.value.trim()}”.`;
          results.append(empty);
          return;
        }

        matches.forEach((entry, position) => {
          const result = document.createElement("a");
          result.id = resultId(position);
          result.href = entry.url;
          result.className = "docs-search-result";
          result.dataset.docSearchResult = "true";
          result.setAttribute("role", "option");
          result.setAttribute("aria-selected", "false");

          const context = document.createElement("small");
          context.textContent = entry.kind === "Page" ? t("Page") : t(entry.page_title);
          const title = document.createElement("strong");
          title.textContent = t(entry.title);
          result.append(context, title);
          result.addEventListener("pointermove", () => setActiveResult(position));
          result.addEventListener("focus", () => setActiveResult(position));
          results.append(result);
        });
      };

      input.addEventListener("input", renderResults);
      input.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
          if (!input.value) return;
          event.preventDefault();
          input.value = "";
          renderResults();
          return;
        }
        if (!matches.length) return;
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault();
          const direction = event.key === "ArrowDown" ? 1 : -1;
          setActiveResult((activeIndex + direction + matches.length) % matches.length);
          return;
        }
        if (event.key === "Enter") {
          event.preventDefault();
          window.location.assign(matches[activeIndex >= 0 ? activeIndex : 0].url);
        }
      });
    });
  };

  const initializeScope = (scope = document) => {
    window.WorkbenchI18n?.apply(scope);
    initializeIcons();
    refreshConditionals(scope);
    initializeSearch(scope);
    initializeFileInputs(scope);
    initializeLogTerminals();
    initializeRunFilters(scope);
    initializeViewer();
    renderCharts(scope);
    initializeDatasetProgress();
    initializeDocsSearch(scope);
  };

  document.addEventListener("DOMContentLoaded", () => {
    window.WorkbenchI18n?.initialize();
    applyTheme(localStorage.getItem(preferenceKey) || "system");
    initializeScope();

    document.addEventListener("change", (event) => {
      if (event.target.matches("[name='model_source'], [name='source_type'], [name='device_mode']")) refreshConditionals();
      if (event.target.matches("[name='run_filter']")) updateRunFilter();
    });

    document.addEventListener("click", (event) => {
      const media = event.target.closest("[data-viewer-media]");
      if (media) {
        event.preventDefault();
        openViewer(media);
        return;
      }

      const toggle = event.target.closest("[data-theme-toggle]");
      if (toggle) {
        const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
        localStorage.setItem(preferenceKey, nextTheme);
        applyTheme(nextTheme);
        return;
      }

      if (event.target.closest("[data-viewer-close]")) {
        closeViewer();
        return;
      }

      if (event.target.closest("[data-viewer-zoom-in]")) {
        adjustViewerZoom(0.25);
        return;
      }

      if (event.target.closest("[data-viewer-zoom-out]")) {
        adjustViewerZoom(-0.25);
        return;
      }

      if (event.target.closest("[data-viewer-reset]")) {
        resetViewerTransform();
        return;
      }

      const runName = event.target.closest("[data-use-run-name]");
      if (runName) {
        const field = document.querySelector("#run-name");
        if (field) {
          field.value = runName.dataset.useRunName;
          field.dispatchEvent(new Event("input", { bubbles: true }));
          field.focus();
        }
      }
    });

    document.addEventListener("keydown", (event) => {
      const viewer = getViewer();
      if (!viewer?.hidden) {
        if (event.key === "Escape") closeViewer();
        if (event.key === "+" || event.key === "=") adjustViewerZoom(0.25);
        if (event.key === "-") adjustViewerZoom(-0.25);
        if (event.key === "0") resetViewerTransform();
      }
    });

    document.querySelector("#run-form")?.addEventListener("submit", (event) => {
      event.preventDefault();
      document.querySelector("[data-start-run]")?.click();
    });

    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      if ((localStorage.getItem(preferenceKey) || "system") === "system") applyTheme("system");
    });
  });

  document.body.addEventListener("htmx:configRequest", (event) => {
    event.detail.headers["X-Theme"] = root.dataset.theme || "light";
  });
  document.body.addEventListener("htmx:beforeRequest", (event) => {
    if (event.detail.pathInfo.requestPath === "/fragments/models/upload") {
      event.detail.elt.closest(".upload-drop")?.classList.add("is-uploading");
    }
  });
  document.body.addEventListener("htmx:beforeSwap", (event) => {
    captureLogScroll(event.detail.target);
    const status = event.detail.xhr.status;
    const response = String(event.detail.serverResponse || "").trim();
    if (status >= 400 && response.startsWith("<")) {
      event.detail.shouldSwap = true;
      event.detail.isError = false;
    }
  });
  document.body.addEventListener("htmx:afterSwap", (event) => initializeScope(event.target));
  document.body.addEventListener("htmx:oobBeforeSwap", (event) => captureLogScroll(event.detail.target));
  document.body.addEventListener("htmx:oobAfterSwap", () => initializeScope());
  document.body.addEventListener("htmx:afterRequest", (event) => {
    if (event.detail.pathInfo.requestPath === "/fragments/models/upload") {
      const input = document.querySelector("[name='model_upload']");
      input?.closest(".upload-drop")?.classList.remove("is-uploading");
      if (input && event.detail.xhr.status < 400) input.value = "";
    }
  });
  document.body.addEventListener("htmx:responseError", (event) => showFeedback(responseMessage(event.detail.xhr)));
  document.body.addEventListener("htmx:sendError", () => showFeedback("The service could not be reached."));
  document.body.addEventListener("htmx:timeout", () => showFeedback("The request timed out."));
  document.body.addEventListener("htmx:swapError", () => showFeedback("The response could not be displayed."));
})();
