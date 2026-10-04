// Simple Smartphone Profiles for Non-Tech Users
const profiles = {
    budget: {
        name: "Basic Daily Phone",
        company_name: "Xiaomi",
        ram_gb: "3",
        storage_gb: "32",
        battery_mah: "5000",
        primary_camera_mp: "13",
        front_camera_mp: "5",
        cpu_speed_ghz: "2.0",
        cpu_cores: "4",
        screen_size_inch: "6.5",
        has_5g: false
    },
    midrange: {
        name: "All-Rounder Phone",
        company_name: "OnePlus",
        ram_gb: "6",
        storage_gb: "128",
        battery_mah: "5000",
        primary_camera_mp: "50",
        front_camera_mp: "16",
        cpu_speed_ghz: "2.4",
        cpu_cores: "8",
        screen_size_inch: "6.5",
        has_5g: true
    },
    gaming: {
        name: "Gamer / Creator Phone",
        company_name: "iQOO",
        ram_gb: "8",
        storage_gb: "256",
        battery_mah: "5000",
        primary_camera_mp: "64",
        front_camera_mp: "16",
        cpu_speed_ghz: "2.8",
        cpu_cores: "8",
        screen_size_inch: "6.7",
        has_5g: true
    },
    flagship: {
        name: "Ultra Flagship Phone",
        company_name: "Apple",
        ram_gb: "12",
        storage_gb: "256",
        battery_mah: "5000",
        primary_camera_mp: "108",
        front_camera_mp: "32",
        cpu_speed_ghz: "3.2",
        cpu_cores: "8",
        screen_size_inch: "6.8",
        has_5g: true
    }
};

// Formatter for Indian Rupees with commas (e.g., ₹24,999)
const inrFormatter = new Intl.NumberFormat('en-IN', {
    maximumFractionDigits: 0
});

document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    loadHistory();
    setupForm();
});

// Check Server & System Status
async function checkHealth() {
    const statusPill = document.getElementById("system-status");
    if (!statusPill) return;

    try {
        const res = await fetch("/api/health");
        const data = await res.json();

        if (data.model_loaded) {
            statusPill.className = "status-indicator online";
            statusPill.innerHTML = '<span class="status-dot"></span> System Ready';
        } else {
            statusPill.className = "status-indicator offline";
            statusPill.innerHTML = '<span class="status-dot"></span> System Connecting';
        }
    } catch (err) {
        statusPill.className = "status-indicator";
        statusPill.innerHTML = '<span class="status-dot"></span> Offline';
    }
}

// Preset Profile Loader
function applyPreset(type) {
    const p = profiles[type];
    if (!p) return;

    if (p.company_name && document.getElementById("company_name")) {
        document.getElementById("company_name").value = p.company_name;
    }
    document.getElementById("ram_gb").value = p.ram_gb;
    document.getElementById("storage_gb").value = p.storage_gb;
    document.getElementById("battery_mah").value = p.battery_mah;
    document.getElementById("primary_camera_mp").value = p.primary_camera_mp;
    document.getElementById("front_camera_mp").value = p.front_camera_mp;
    document.getElementById("cpu_speed_ghz").value = p.cpu_speed_ghz;
    document.getElementById("cpu_cores").value = p.cpu_cores;
    document.getElementById("screen_size_inch").value = p.screen_size_inch;
    document.getElementById("has_5g").checked = p.has_5g;

    // Highlight selected preset button
    document.querySelectorAll(".btn-preset").forEach(btn => btn.classList.remove("active-preset"));
    event.currentTarget.classList.add("active-preset");

    showToast(`Loaded ${p.name} settings!`, "info");
}

// Setup Form Submission
function setupForm() {
    const form = document.getElementById("prediction-form");
    const submitBtn = document.getElementById("submit-btn");
    const btnLabel = submitBtn.querySelector(".btn-label");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const companyElem = document.getElementById("company_name");
        const payload = {
            company_name: companyElem ? companyElem.value : "Samsung",
            ram_gb: parseInt(document.getElementById("ram_gb").value),
            storage_gb: parseInt(document.getElementById("storage_gb").value),
            battery_mah: parseInt(document.getElementById("battery_mah").value),
            primary_camera_mp: parseInt(document.getElementById("primary_camera_mp").value),
            front_camera_mp: parseInt(document.getElementById("front_camera_mp").value),
            cpu_speed_ghz: parseFloat(document.getElementById("cpu_speed_ghz").value),
            cpu_cores: parseInt(document.getElementById("cpu_cores").value),
            screen_size_inch: parseFloat(document.getElementById("screen_size_inch").value),
            has_5g: document.getElementById("has_5g").checked
        };

        submitBtn.disabled = true;
        btnLabel.textContent = "Checking Market Prices...";

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await res.json();

            if (data.success) {
                displayResult(data, payload);
                showToast("Fair price calculated and saved!", "success");
                loadHistory();
            } else {
                showToast(data.error || "Price check failed.", "error");
            }
        } catch (err) {
            console.error("Valuation request failed:", err);
            showToast("Network error. Please try again.", "error");
        } finally {
            submitBtn.disabled = false;
            btnLabel.textContent = "🔍 Calculate Fair Market Price";
        }
    });
}

// Generate friendly user advice based on hardware
function getFriendlyVerdict(specs, price) {
    if (specs.ram_gb >= 12 || price >= 55000) {
        return "🌟 Ultra Luxury Tier: Exceptional power for 4K video recording, high-end professional photography, ultra-smooth 3D gaming, and heavy multitasking.";
    } else if (specs.ram_gb >= 8) {
        return "🎮 Creator & Gamer Grade: Perfect for high-graphics games (BGMI, FreeFire), fast video editing, and running heavy apps simultaneously without slowdowns.";
    } else if (specs.ram_gb >= 6) {
        return "✨ Ideal Everyday All-Rounder: Smooth app switching, great for YouTube & Instagram, sharp family photos, and fast future-proof 5G internet.";
    } else {
        return "📱 Practical Daily Phone: Reliable for phone calls, WhatsApp chats, Google Maps, online payments, and basic daily utility.";
    }
}

// Display Valuation Output
function displayResult(result, specs) {
    document.getElementById("result-placeholder").style.display = "none";
    const resultContent = document.getElementById("result-content");
    resultContent.style.display = "block";

    // Format Indian Rupees
    document.getElementById("price-inr").textContent = inrFormatter.format(result.price_inr);

    // Friendly Market segment tag
    const tierBadge = document.getElementById("tier-badge");
    tierBadge.textContent = result.price_tier;
    tierBadge.className = "tier-tag";

    if (result.price_tier.includes("Budget")) {
        tierBadge.classList.add("tier-budget");
        tierBadge.textContent = "Budget Friendly Segment";
    } else if (result.price_tier.includes("Mid-Range")) {
        tierBadge.classList.add("tier-midrange");
        tierBadge.textContent = "Mid-Range (Best Value For Money)";
    } else if (result.price_tier.includes("Premium") || result.price_tier.includes("Upper")) {
        tierBadge.classList.add("tier-premium");
        tierBadge.textContent = "Premium Performance Segment";
    } else {
        tierBadge.classList.add("tier-flagship");
        tierBadge.textContent = "Flagship Luxury Segment";
    }

    // Display Mobile Company Information
    if (result.company_name) {
        const companyBox = document.getElementById("company-display-box");
        if (companyBox) companyBox.style.display = "block";

        const nameElem = document.getElementById("predicted-company-name");
        if (nameElem) nameElem.textContent = result.company_name;

        const taglineElem = document.getElementById("predicted-company-tagline");
        if (taglineElem) taglineElem.textContent = result.company_tagline || "Verified Market Brand";

        const originElem = document.getElementById("predicted-company-origin");
        if (originElem) originElem.textContent = result.company_origin ? `Origin: ${result.company_origin}` : "Verified Match";

        const modelsContainer = document.getElementById("predicted-matching-models");
        if (modelsContainer) {
            if (result.matching_models && result.matching_models.length > 0) {
                modelsContainer.innerHTML = result.matching_models.map(m => `<span class="model-tag">📱 ${m}</span>`).join("");
            } else {
                modelsContainer.innerHTML = `<span class="model-tag">📱 Standard Series (${specs.ram_gb}GB + ${specs.storage_gb}GB)</span>`;
            }
        }

        const altWrapper = document.getElementById("alt-brands-wrapper");
        const altNames = document.getElementById("predicted-alt-brands");
        if (altWrapper && altNames) {
            if (result.alternative_companies && result.alternative_companies.length > 0) {
                altNames.textContent = result.alternative_companies.join(", ");
                altWrapper.style.display = "flex";
            } else {
                altWrapper.style.display = "none";
            }
        }
    }

    // Set Plain English Recommendation
    const verdict = getFriendlyVerdict(specs, result.price_inr);
    document.getElementById("use-case-desc").textContent = verdict;

    // Database record indicator
    const syncText = document.getElementById("db-sync-text");
    syncText.textContent = "Result saved to your history table below.";

    // Specifications summary chips in friendly plain language
    const chipsContainer = document.getElementById("summary-chips");
    chipsContainer.innerHTML = `
        <div class="spec-chip">🏢 <strong>${result.company_name || 'Samsung'}</strong></div>
        <div class="spec-chip">🧠 <strong>${specs.ram_gb} GB RAM</strong> (Smoothness)</div>
        <div class="spec-chip">💾 <strong>${specs.storage_gb} GB Storage</strong> (Photos/Apps)</div>
        <div class="spec-chip">🔋 <strong>${specs.battery_mah} mAh</strong> (All-Day Battery)</div>
        <div class="spec-chip">📸 <strong>${specs.primary_camera_mp} MP</strong> Back / <strong>${specs.front_camera_mp} MP</strong> Selfie</div>
        <div class="spec-chip">⚡ <strong>${specs.cpu_cores} Cores</strong> @ ${specs.cpu_speed_ghz} GHz</div>
        <div class="spec-chip">📱 <strong>${specs.screen_size_inch}"</strong> Screen Display</div>
        <div class="spec-chip">📶 <strong>${specs.has_5g ? "5G Internet Ready" : "4G LTE Network"}</strong></div>
    `;

    // Smoothly scroll to results on mobile / tablet screens
    if (window.innerWidth <= 920) {
        setTimeout(() => {
            resultContent.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }, 100);
    }
}

// Load Saved History for Non-Tech Users
async function loadHistory() {
    const tbody = document.getElementById("history-tbody");

    try {
        const res = await fetch("/api/history");
        const data = await res.json();

        if (!data.success || !data.records || data.records.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="11" class="empty-table-cell">No saved checks yet. Use the form above to check your first phone!</td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = data.records.map(rec => {
            const tierClass = rec.price_tier.includes("Budget") ? "tier-budget" :
                             rec.price_tier.includes("Mid-Range") ? "tier-midrange" :
                             (rec.price_tier.includes("Premium") || rec.price_tier.includes("Upper")) ? "tier-premium" : "tier-flagship";

            const formattedPrice = "₹" + inrFormatter.format(rec.predicted_price);

            return `
                <tr>
                    <td><strong>#${rec.id}</strong></td>
                    <td><strong style="color: #0f172a; font-size: 0.88rem;">${rec.company_name || 'Samsung'}</strong></td>
                    <td><strong>${rec.ram_gb} GB</strong> / ${rec.storage_gb} GB Space</td>
                    <td>${rec.battery_mah} mAh</td>
                    <td>${rec.primary_camera_mp} MP + ${rec.front_camera_mp} MP</td>
                    <td>${rec.cpu_cores} Cores (${rec.cpu_speed_ghz} GHz)</td>
                    <td>${rec.has_5g ? "✅ 5G" : "4G"}</td>
                    <td><strong style="font-size: 0.95rem; color: #c2410c;">${formattedPrice}</strong></td>
                    <td><span class="tier-tag ${tierClass}" style="margin:0; font-size:0.75rem;">${rec.price_tier}</span></td>
                    <td style="color:#78716a; font-size:0.8rem;">${rec.created_at || "Recent"}</td>
                    <td>
                        <button class="btn-delete-row" title="Remove this record" onclick="deleteHistoryItem(${rec.id})">Delete</button>
                    </td>
                </tr>
            `;
        }).join("");

    } catch (err) {
        tbody.innerHTML = `
            <tr>
                <td colspan="11" class="empty-table-cell" style="color: #b91c1c;">
                    Unable to load previous checks right now.
                </td>
            </tr>
        `;
    }
}

// Delete History Record
async function deleteHistoryItem(id) {
    if (!confirm(`Do you want to remove check #${id} from your saved list?`)) {
        return;
    }

    try {
        const res = await fetch(`/api/history/${id}`, { method: "DELETE" });
        const data = await res.json();

        if (data.success) {
            showToast(`Record #${id} removed.`, "success");
            loadHistory();
        } else {
            showToast(data.message || "Failed to remove record.", "error");
        }
    } catch (err) {
        showToast("Network error removing record.", "error");
    }
}

// Simple Toast Notification
function showToast(message, type = "info") {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.className = `toast-message ${type}`;
    toast.style.display = "block";

    setTimeout(() => {
        toast.style.display = "none";
    }, 3200);
}
