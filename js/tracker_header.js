(function() {
  var cachedStatus = null;
  var isFetching = false;

  function formatCompactDate(dateStr) {
    if (!dateStr || dateStr === "N/A") return "N/A";
    try {
      // Input is like "2026-06-12 01:39:01" -> output "2606120139"
      var yr = dateStr.substring(2, 4);
      var mo = dateStr.substring(5, 7);
      var dy = dateStr.substring(8, 10);
      var hr = dateStr.substring(11, 13);
      var mi = dateStr.substring(14, 16);
      return yr + mo + dy + hr + mi;
    } catch (e) {
      return "N/A";
    }
  }

  function formatMetricVal(val, key) {
    if (typeof val !== "number") return val;
    var name = key.toLowerCase();
    if (name.indexOf("faithfulness") !== -1 || name.indexOf("recall") !== -1) {
      return (val * 100).toFixed(0) + "%";
    }
    if (Math.abs(val) >= 1000000) {
      return (val / 1000000).toFixed(2) + "M";
    }
    if (Math.abs(val) >= 1000) {
      return (val / 1000).toFixed(1) + "k";
    }
    if (val !== 0 && Math.abs(val) < 0.01) {
      return val.toFixed(5);
    }
    if (val !== 0 && Math.abs(val) < 0.1) {
      return val.toFixed(4);
    }
    return val.toFixed(3);
  }

  function formatKeyName(key) {
    if (key === "r2_score") return "r2";
    if (key === "total_latency_sec") return "latency";
    if (key === "total_cost_usd") return "cost";
    if (key === "overlap_faithfulness") return "faith";
    if (key === "context_recall") return "recall";
    if (key === "llm_judge_grounding") return "judge";
    if (key === "n_estimators") return "n_est";
    if (key === "learning_rate") return "lr";
    if (key === "model_type") return "type";
    if (key === "max_depth") return "depth";
    return key;
  }

  function updateDom(data) {
    var dataValEl = document.getElementById("tracker-data-val");
    if (dataValEl) {
      var dVer = data.data_version || "N/A";
      if (dVer !== "N/A" && !dVer.startsWith("v")) {
        dVer = "v-" + dVer.substring(0, 8);
      }
      var dTime = formatCompactDate(data.data_last_updated);
      var newText = dVer + (dTime !== "N/A" ? " - " + dTime : "");
      if (dataValEl.textContent !== newText) {
        dataValEl.textContent = newText;
      }
    }

    var modelValEl = document.getElementById("tracker-model-val");
    if (modelValEl) {
      var mVer = data.model_version || "N/A";
      var mTime = formatCompactDate(data.model_last_updated);
      var newText = mVer + (mTime !== "N/A" ? " - " + mTime : "");
      if (modelValEl.textContent !== newText) {
        modelValEl.textContent = newText;
      }
    }

    var promptValEl = document.getElementById("tracker-prompt-val");
    if (promptValEl) {
      var pVer = data.prompt_version || "N/A";
      if (promptValEl.textContent !== pVer) {
        promptValEl.textContent = pVer;
      }
    }

    var runValEl = document.getElementById("tracker-run-val");
    if (runValEl) {
      var runText = "N/A";
      if (data.latest_runs && data.latest_runs.length > 0) {
        var run = data.latest_runs[0];
        var runId = run.run_id || "N/A";
        var status = run.status || "";
        var statusIndicator = "";
        if (status === "FINISHED" || status === "SUCCESS") {
          statusIndicator = "🟢";
        } else if (status === "FAILED") {
          statusIndicator = "🔴";
        } else if (status === "RUNNING") {
          statusIndicator = "🟡";
        }
        runText = runId + (statusIndicator ? " " + statusIndicator : "");
      }
      if (runValEl.textContent !== runText) {
        runValEl.textContent = runText;
      }
    }

    var metricValEl = document.getElementById("tracker-metric-val");
    if (metricValEl) {
      var metricText = "N/A";
      if (data.latest_runs && data.latest_runs.length > 0) {
        var run = data.latest_runs[0];
        if (run.metrics && Object.keys(run.metrics).length > 0) {
          var priority = ["r2_score", "r2", "rmse", "train_mae", "overlap_faithfulness", "context_recall", "llm_judge_grounding", "loss", "accuracy"];
          var keys = Object.keys(run.metrics);
          keys.sort(function(a, b) {
            var idxA = priority.indexOf(a);
            var idxB = priority.indexOf(b);
            if (idxA === -1 && idxB === -1) return a.localeCompare(b);
            if (idxA === -1) return 1;
            if (idxB === -1) return -1;
            return idxA - idxB;
          });
          if (keys.length > 0) {
            var k = keys[0];
            var v = run.metrics[k];
            metricText = formatKeyName(k) + ": " + formatMetricVal(v, k);
          }
        }
      }
      if (metricValEl.textContent !== metricText) {
        metricValEl.textContent = metricText;
      }
    }

    var paramValEl = document.getElementById("tracker-param-val");
    if (paramValEl) {
      var paramText = "N/A";
      if (data.latest_runs && data.latest_runs.length > 0) {
        var run = data.latest_runs[0];
        if (run.params && Object.keys(run.params).length > 0) {
          var priority = ["model_type", "n_estimators", "max_depth", "learning_rate", "epochs", "prompt_version"];
          var keys = Object.keys(run.params);
          keys.sort(function(a, b) {
            var idxA = priority.indexOf(a);
            var idxB = priority.indexOf(b);
            if (idxA === -1 && idxB === -1) return a.localeCompare(b);
            if (idxA === -1) return 1;
            if (idxB === -1) return -1;
            return idxA - idxB;
          });
          if (keys.length > 0) {
            var k = keys[0];
            var v = run.params[k];
            paramText = formatKeyName(k) + ": " + v;
          }
        }
      }
      if (paramValEl.textContent !== paramText) {
        paramValEl.textContent = paramText;
      }
    }
  }

  function injectTracker() {
    // 1. If tracker box already exists in the header, do nothing.
    if (document.querySelector(".header-tracker-box")) {
      return;
    }

    var search = document.querySelector(".md-search");
    if (!search) return;

    // Find the dashboard link from the navigation to get the correct relative URL
    var navLink = document.querySelector('a[href*="tracking_dashboard"]');
    var targetHref = navLink ? navLink.getAttribute("href") : "/src/tracking_dashboard/";

    // Create the tracker button element
    var tracker = document.createElement("a");
    tracker.href = targetHref;
    tracker.className = "header-tracker-box";
    tracker.innerHTML = 
      '<div class="tracker-cell"><span class="tracker-icon">📦</span> <span class="tracker-label">data:</span> <span class="tracker-val" id="tracker-data-val">Loading...</span></div>' +
      '<div class="tracker-cell"><span class="tracker-icon">🧠</span> <span class="tracker-label">model:</span> <span class="tracker-val" id="tracker-model-val">Loading...</span></div>' +
      '<div class="tracker-cell"><span class="tracker-icon">💬</span> <span class="tracker-label">prompt:</span> <span class="tracker-val" id="tracker-prompt-val">Loading...</span></div>' +
      '<div class="tracker-cell"><span class="tracker-icon">🧪</span> <span class="tracker-label">run:</span> <span class="tracker-val" id="tracker-run-val">Loading...</span></div>' +
      '<div class="tracker-cell"><span class="tracker-icon">📈</span> <span class="tracker-label">metric:</span> <span class="tracker-val" id="tracker-metric-val">Loading...</span></div>' +
      '<div class="tracker-cell"><span class="tracker-icon">🔧</span> <span class="tracker-label">param:</span> <span class="tracker-val" id="tracker-param-val">Loading...</span></div>';

    // Insert directly after the search bar container
    search.parentNode.insertBefore(tracker, search.nextSibling);

    // 2. Use cached tracker data if available
    if (cachedStatus) {
      updateDom(cachedStatus);
      return;
    }

    // 3. Prevent duplicate active fetches
    if (isFetching) {
      return;
    }

    isFetching = true;
    var statusUrl = "/tracker_status.json";
    var cssLink = document.querySelector('link[href*="extra.css"]');
    if (cssLink) {
      var href = cssLink.getAttribute("href");
      statusUrl = href.replace("stylesheets/extra.css", "tracker_status.json");
    }

    fetch(statusUrl)
      .then(function(res) {
        if (!res.ok) throw new Error("Status file not found");
        return res.json();
      })
      .then(function(data) {
        cachedStatus = data;
        updateDom(data);
        isFetching = false;
      })
      .catch(function(err) {
        console.warn("Could not load tracker status: ", err);
        isFetching = false;
      });
  }

  // Run immediately and on page load events
  injectTracker();
  document.addEventListener("DOMContentLoaded", injectTracker);

  // Monitor DOM insertions to handle fast SPA transitions in Material theme
  var observer = new MutationObserver(function(mutations) {
    // Only check mutations if the tracker box isn't already present in the DOM
    if (!document.querySelector(".header-tracker-box")) {
      for (var i = 0; i < mutations.length; i++) {
        if (mutations[i].addedNodes.length) {
          injectTracker();
          break;
        }
      }
    }
  });

  observer.observe(document.documentElement, {
    childList: true,
    subtree: true
  });
})();
