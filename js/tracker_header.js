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
      '<div class="tracker-row"><span class="tracker-icon">📦</span> <span class="tracker-label">data:</span> <span class="tracker-val" id="tracker-data-val">Loading...</span></div>' +
      '<div class="tracker-row"><span class="tracker-icon">🧠</span> <span class="tracker-label">model:</span> <span class="tracker-val" id="tracker-model-val">Loading...</span></div>';

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
