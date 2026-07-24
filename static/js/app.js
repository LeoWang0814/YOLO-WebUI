(() => {
  const preferenceKey = "yolov10-workbench.theme";
  const root = document.documentElement;
  let feedbackTimer;
  const pendingLogScroll = new Map();
  const actualTheme = (preference) => preference === "system"
    ? (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
    : preference;

  const applyTheme = (preference) => {
    const theme = actualTheme(preference);
    root.dataset.theme = theme;
    root.dataset.themePreference = preference;
    document.querySelectorAll("[data-theme-choice]").forEach((button) => {
      button.setAttribute("aria-checked", String(button.dataset.themeChoice === preference));
    });
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
        chart.textContent = "Unable to render metrics.";
      }
    });
  };

  const initializeIcons = () => {
    if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
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
        if (!input.files?.length) summary.textContent = input.multiple ? "No files selected" : "No file selected";
        else if (input.files.length === 1) summary.textContent = input.files[0].name;
        else summary.textContent = `${input.files.length} files selected`;
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
      ? (section.hasAttribute("hx-get") ? "Following output" : "At latest output")
      : "Auto-follow paused";
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

  const closeThemeMenu = () => {
    document.querySelector(".theme-popover")?.classList.remove("is-open");
    document.querySelector("[data-theme-toggle]")?.setAttribute("aria-expanded", "false");
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

  const initializeScope = (scope = document) => {
    initializeIcons();
    refreshConditionals(scope);
    initializeSearch(scope);
    initializeFileInputs(scope);
    initializeLogTerminals();
    initializeRunFilters(scope);
    renderCharts(scope);
  };

  document.addEventListener("DOMContentLoaded", () => {
    applyTheme(localStorage.getItem(preferenceKey) || "system");
    initializeScope();

    document.addEventListener("change", (event) => {
      if (event.target.matches("[name='model_source'], [name='source_type'], [name='device_mode']")) refreshConditionals();
      if (event.target.matches("[name='run_filter']")) updateRunFilter();
    });

    document.addEventListener("click", (event) => {
      const choice = event.target.closest("[data-theme-choice]");
      if (choice) {
        const preference = choice.dataset.themeChoice;
        localStorage.setItem(preferenceKey, preference);
        applyTheme(preference);
        closeThemeMenu();
        return;
      }

      const toggle = event.target.closest("[data-theme-toggle]");
      if (toggle) {
        const popover = toggle.closest(".theme-menu")?.querySelector(".theme-popover");
        const open = !popover?.classList.contains("is-open");
        popover?.classList.toggle("is-open", open);
        toggle.setAttribute("aria-expanded", String(open));
        return;
      }

      if (!event.target.closest(".theme-menu")) closeThemeMenu();

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
      if (event.key === "Escape") closeThemeMenu();
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
