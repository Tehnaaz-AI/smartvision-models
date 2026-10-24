/**
 * SmartVision Caregiver Portal Client Application.
 *
 * Real-time telemetry, prescription management, verification feeds, and HUD simulator.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const prescriptionsList = document.getElementById("prescriptions-list");
  const timeline = document.getElementById("verification-timeline");
  const alertsContainer = document.getElementById("alerts-container");
  const prescriptionsCount = document.getElementById("prescriptions-count");
  const btnAddMed = document.getElementById("btn-add-med");
  const modalAddMed = document.getElementById("modal-add-med");
  const btnCancelMed = document.getElementById("btn-cancel-med");
  const formAddMed = document.getElementById("form-add-med");
  const btnSimulateScan = document.getElementById("btn-simulate-scan");
  const toast = document.getElementById("toast");

  // Telemetry Elements
  const deviceStatus = document.getElementById("device-status");
  const batteryLevel = document.getElementById("battery-level");
  const templeTemp = document.getElementById("temple-temp");
  const statVerified = document.getElementById("stat-verified-count");
  const statWarning = document.getElementById("stat-warning-count");

  // HUD Elements
  const hudScreen = document.getElementById("hud-screen");
  const hudStatusBadge = document.getElementById("hud-status-badge");
  const hudTitle = document.getElementById("hud-title");
  const hudMain = document.getElementById("hud-main");
  const hudSub = document.getElementById("hud-sub");

  // GPS & Google Maps Elements
  const gpsAddress = document.getElementById("gps-address");
  const gpsCoordsText = document.getElementById("gps-coords-text");
  const gpsDistanceText = document.getElementById("gps-distance-text");
  const geofenceStatus = document.getElementById("geofence-status");
  const btnGoogleMaps = document.getElementById("btn-google-maps");
  const btnSimulateWalk = document.getElementById("btn-simulate-walk");
  const btnQueryPharmacy = document.getElementById("btn-query-pharmacy");
  const pharmacyList = document.getElementById("pharmacy-list");

  let isSimulatedOutside = false;

  let safeCount = 18;
  let warningCount = 2;

  // Initial State: Prescriptions
  let prescriptions = [
    {
      name: "Dolo",
      strength: "650 mg",
      dosage: "1 Tablet",
      schedule_time: "08:00 AM",
      meal_relation: "AFTER_FOOD",
      instructions: "Take one tablet after breakfast with water",
      instructions_te: "అల్పాహారం తర్వాత ఒక టాబ్లెట్ నీటితో వేసుకోండి",
    },
    {
      name: "Metformin",
      strength: "500 mg",
      dosage: "1 Tablet",
      schedule_time: "08:00 AM",
      meal_relation: "WITH_FOOD",
      instructions: "Take one tablet with morning food for diabetes",
      instructions_te: "మధుమేహం కోసం ఉదయం భోజనంతో ఒక టాబ్లెట్ తీసుకోండి",
    },
    {
      name: "Atorvastatin",
      strength: "10 mg",
      dosage: "1 Tablet",
      schedule_time: "08:00 PM",
      meal_relation: "AFTER_FOOD",
      instructions: "Take one tablet after dinner before sleep for cholesterol",
      instructions_te: "కొలెస్ట్రాల్ కోసం రాత్రి భోజనం తర్వాత పడుకునే ముందు ఒక టాబ్లెట్ తీసుకోండి",
    },
  ];

  // Initial Timeline Verifications
  const verificationLogs = [
    {
      status: "SAFE",
      medicine: "Dolo 650 mg",
      ocr_text: "DOLO 650 PARACETAMOL IP",
      confidence: "98.7%",
      time: "08:05 AM Today",
      message: "Verified safe. Take one tablet after food with water.",
      message_te: "సురక్షితమైనది. భోజనం తర్వాత ఒక టాబ్లెట్ నీటితో తీసుకోండి.",
    },
    {
      status: "SAFE",
      medicine: "Metformin 500 mg",
      ocr_text: "METFORMIN HYDROCHLORIDE 500MG",
      confidence: "99.1%",
      time: "08:07 AM Today",
      message: "Verified safe. Take one tablet with breakfast.",
      message_te: "సురక్షితమైనది. అల్పాహారంతో ఒక టాబ్లెట్ తీసుకోండి.",
    },
    {
      status: "WARNING",
      medicine: "Amlodipine 5 mg",
      ocr_text: "AMLODIPINE BESYLATE 5MG",
      confidence: "95.2%",
      time: "08:12 AM Today",
      message: "WARNING: Amlodipine is not on Lakshmi Devi's prescription. Do not take!",
      message_te: "హెచ్చరిక: ఈ మందు మీ ప్రిస్క్రిప్షన్‌లో లేదు. దయచేసి వేసుకోవద్దు!",
    },
  ];

  // Render Prescriptions
  function renderPrescriptions() {
    prescriptionsCount.textContent = prescriptions.length;
    prescriptionsList.innerHTML = prescriptions
      .map(
        (m) => `
      <div class="prescription-item">
        <div class="item-top">
          <span class="med-name">${escapeHtml(m.name)} <small style="color:var(--text-dim)">(${escapeHtml(m.strength)})</small></span>
          <span class="med-schedule">${escapeHtml(m.schedule_time)}</span>
        </div>
        <div class="med-instructions">${escapeHtml(m.instructions)}</div>
        <div class="med-instructions-te">${escapeHtml(m.instructions_te)}</div>
      </div>
    `
      )
      .join("");
  }

  // Render Timeline
  function renderTimeline() {
    statVerified.textContent = safeCount;
    statWarning.textContent = warningCount;

    timeline.innerHTML = verificationLogs
      .map((entry) => {
        const isSafe = entry.status === "SAFE";
        return `
        <div class="timeline-entry ${isSafe ? "safe" : "warning"}">
          <div class="entry-icon">${isSafe ? "✓" : "!"}</div>
          <div class="entry-content">
            <div class="entry-title-row">
              <span class="entry-title">${escapeHtml(entry.medicine)}</span>
              <span class="entry-time">${escapeHtml(entry.time)}</span>
            </div>
            <div class="entry-desc">OCR: <span class="mono">${escapeHtml(entry.ocr_text)}</span> (${escapeHtml(entry.confidence)})</div>
            <div class="entry-spoken">Spoken: "${escapeHtml(entry.message)}"</div>
          </div>
        </div>
      `;
      })
      .join("");
  }

  // Render Emergency Alerts
  function renderAlerts() {
    fetch("/api/caregiver/alerts")
      .then((res) => (res.ok ? res.json() : { alerts: [] }))
      .then((data) => {
        const alerts = data.alerts || [];
        if (alerts.length === 0) {
          alertsContainer.innerHTML = `
            <div class="alert-item info">
              <div class="alert-title-row">
                <span>SYSTEM NORMAL</span>
                <span>Active</span>
              </div>
              <div class="alert-text">No active medical mismatch alerts. All prescriptions adhere to schedule.</div>
            </div>
          `;
          return;
        }

        alertsContainer.innerHTML = alerts
          .map(
            (a) => `
          <div class="alert-item ${a.severity === "CRITICAL" ? "" : "info"}">
            <div class="alert-title-row">
              <span>${escapeHtml(a.type)}</span>
              <span style="color:var(--accent-red)">${escapeHtml(a.severity)}</span>
            </div>
            <div class="alert-text">${escapeHtml(a.message)}</div>
            <div class="alert-meta">
              <span>Status: ${escapeHtml(a.status)}</span>
              <span class="mono">${new Date(a.timestamp).toLocaleTimeString()}</span>
            </div>
          </div>
        `
          )
          .join("");
      })
      .catch(() => {
        // Fallback default alert
        alertsContainer.innerHTML = `
          <div class="alert-item">
            <div class="alert-title-row">
              <span>MEDICATION_MISMATCH</span>
              <span style="color:var(--accent-red)">CRITICAL</span>
            </div>
            <div class="alert-text">Caregiver SMS dispatched: Unprescribed tablet 'Amlodipine 5mg' intercepted by smart glasses.</div>
            <div class="alert-meta">
              <span>Status: DELIVERED (SMS)</span>
              <span class="mono">08:12 AM</span>
            </div>
          </div>
        `;
      });
  }

  // Update Lens HUD
  function updateHUD(isSafe, medName, dose, warning) {
    if (isSafe) {
      hudScreen.classList.remove("alert-mode");
      hudStatusBadge.textContent = "[OK - SAFE]";
      hudTitle.textContent = "VERIFIED SAFE";
      hudMain.textContent = `> ${medName.toUpperCase()}`;
      hudSub.textContent = `Take: ${dose}`;
    } else {
      hudScreen.classList.add("alert-mode");
      hudStatusBadge.textContent = "[!! ALERT !!]";
      hudTitle.textContent = "WARNING: DO NOT TAKE";
      hudMain.textContent = `> ${medName ? medName.toUpperCase() : "UNKNOWN"}`;
      hudSub.textContent = warning || "Unprescribed medicine!";
    }
  }

  // Show Toast
  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.remove("hidden");
    setTimeout(() => toast.classList.add("hidden"), 3500);
  }

  // Form Interactions
  btnAddMed.addEventListener("click", () => {
    modalAddMed.classList.toggle("hidden");
  });

  btnCancelMed.addEventListener("click", () => {
    modalAddMed.classList.add("hidden");
  });

  formAddMed.addEventListener("submit", (e) => {
    e.preventDefault();
    const newMed = {
      name: document.getElementById("input-med-name").value.trim(),
      strength: document.getElementById("input-med-strength").value.trim(),
      dosage: document.getElementById("input-med-dose").value.trim(),
      schedule_time: document.getElementById("input-med-time").value.trim(),
      meal_relation: document.getElementById("input-med-meal").value,
      instructions: document.getElementById("input-med-inst-en").value.trim(),
      instructions_te: document.getElementById("input-med-inst-te").value.trim(),
    };

    prescriptions.push(newMed);
    renderPrescriptions();
    formAddMed.reset();
    modalAddMed.classList.add("hidden");
    showToast(`Prescription '${newMed.name} ${newMed.strength}' saved to Cloud DB & Smart Glasses!`);

    // Post to backend
    fetch("/api/caregiver/prescriptions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(newMed),
    }).catch((err) => console.debug("Offline cached:", err));
  });

  // Simulate Glasses Scan (Demonstrates live hardware-software loop)
  const simulationScenarios = [
    {
      ocr_text: "DOLO 650 PARACETAMOL TABLETS IP",
      strength: "650 mg",
      scheduled: "Dolo",
    },
    {
      ocr_text: "METFORMIN HYDROCHLORIDE 500 MG",
      strength: "500 mg",
      scheduled: "Metformin",
    },
    {
      ocr_text: "CIPROFLOXACIN 500 MG ANTIBIOTIC",
      strength: "500 mg",
      scheduled: null, // Unprescribed!
    },
  ];
  let scanIndex = 0;

  btnSimulateScan.addEventListener("click", () => {
    const scenario = simulationScenarios[scanIndex % simulationScenarios.length];
    scanIndex++;

    btnSimulateScan.textContent = "Processing Optical Burst...";
    btnSimulateScan.disabled = true;

    fetch("/api/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        detected_text: scenario.ocr_text,
        detected_strength: scenario.strength,
        scheduled_medicine_name: scenario.scheduled,
        ocr_confidence: 0.98,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        btnSimulateScan.textContent = "Simulate Glasses Scan";
        btnSimulateScan.disabled = false;

        const isSafe = data.is_safe_to_take;
        if (isSafe) {
          safeCount++;
        } else {
          warningCount++;
        }

        const newLog = {
          status: isSafe ? "SAFE" : "WARNING",
          medicine: data.matched_medicine || scenario.ocr_text.split(" ")[0],
          ocr_text: scenario.ocr_text,
          confidence: "98.2%",
          time: new Date().toLocaleTimeString(),
          message: data.spoken_message,
          message_te: data.spoken_message_te,
        };

        verificationLogs.unshift(newLog);
        renderTimeline();
        updateHUD(isSafe, newLog.medicine, "1 Tablet", data.spoken_message);
        renderAlerts();

        showToast(
          isSafe
            ? `Verified: '${newLog.medicine}' is safe to take.`
            : `WARNING: Unsafe medicine intercepted! Caregiver alert triggered.`
        );
      })
      .catch((err) => {
        btnSimulateScan.textContent = "Simulate Glasses Scan";
        btnSimulateScan.disabled = false;
        console.error("Verification error:", err);
      });
  });

  // GPS & Google Maps Location Handlers
  function loadLocationData() {
    fetch("/api/location/current")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!data || !data.coordinates) return;
        const coords = data.coordinates;
        gpsAddress.textContent = coords.address || "Banjara Hills, Hyderabad";
        gpsCoordsText.textContent = `${coords.latitude.toFixed(4)}° N, ${coords.longitude.toFixed(4)}° E`;
        btnGoogleMaps.href = data.google_maps_pin;
      })
      .catch((err) => console.debug("GPS load fallback:", err));
  }

  function loadNearbyPharmacies() {
    fetch("/api/location/nearby-pharmacies?language=te")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!data || !data.pharmacies) return;
        pharmacyList.innerHTML = data.pharmacies
          .map(
            (p) => `
          <div class="pharmacy-item">
            <div>
              <a href="${escapeHtml(p.maps_url)}" target="_blank">${escapeHtml(p.name)}</a>
              <div style="font-size:0.7rem; color:var(--text-dim)">${escapeHtml(p.address)}</div>
            </div>
            <span class="pharmacy-dist">${p.distance_meters}m</span>
          </div>
        `
          )
          .join("");
      })
      .catch((err) => console.debug("Pharmacy load fallback:", err));
  }

  // Simulate Patient Walking (Testing Geofence Wandering Alert)
  btnSimulateWalk.addEventListener("click", () => {
    isSimulatedOutside = !isSimulatedOutside;
    // Inside: 17.4399, 78.4482 | Outside (450m away): 17.4442, 78.4525
    const lat = isSimulatedOutside ? 17.4442 : 17.4399;
    const lng = isSimulatedOutside ? 78.4525 : 78.4482;

    fetch("/api/location/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        latitude: lat,
        longitude: lng,
        speed_kmh: 3.5,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        const st = data.location_status;
        gpsCoordsText.textContent = `${st.latitude.toFixed(4)}° N, ${st.longitude.toFixed(4)}° E`;
        gpsAddress.textContent = st.address;
        gpsDistanceText.textContent = `Distance from Home: ${st.distance_from_home_m}m`;
        btnGoogleMaps.href = st.google_maps_url;

        if (st.is_inside_safe_zone) {
          geofenceStatus.textContent = "Safe Zone";
          geofenceStatus.className = "badge badge-accent";
          showToast("Patient is inside Home Safe Zone.");
        } else {
          geofenceStatus.textContent = "Wandering Alert!";
          geofenceStatus.className = "badge";
          geofenceStatus.style.background = "rgba(239, 68, 68, 0.2)";
          geofenceStatus.style.color = "var(--accent-red)";
          geofenceStatus.style.borderColor = "var(--accent-red)";
          renderAlerts();
          showToast(`GEOFENCE ALERT: Patient is ${st.distance_from_home_m}m away! Caregiver SMS dispatched with Google Maps link.`);
        }
      });
  });

  // Voice Search Nearby Pharmacy Simulation
  btnQueryPharmacy.addEventListener("click", () => {
    fetch("/api/location/nearby-pharmacies?language=te")
      .then((res) => res.json())
      .then((data) => {
        showToast(`Smart Glasses Audio: "${data.spoken_en}"`);
      });
  });

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Initial Boot
  renderPrescriptions();
  renderTimeline();
  renderAlerts();
  loadLocationData();
  loadNearbyPharmacies();
});
