
"use strict";

// ==========================================
// EPISCOPE - DASHBOARD APPLICATION
// ==========================================

const DATA_PATH = "static/data/";
const API_PATH = "http://127.0.0.1:8001/api/";

const state = {
    weekly: [],
    districtSummary: [],
    originalAlerts: [],
    improvedAlerts: [],
    forecasts: [],
    resourceSummary: [],
    facilities: [],
    selectedDistrict: null,
    trendChart: null
};

const numberFormat = new Intl.NumberFormat("en-US");

function formatNumber(value) {
    return numberFormat.format(Math.round(Number(value) || 0));
}

function formatDecimal(value) {
    return Number(value || 0).toLocaleString("en-US", {
        maximumFractionDigits: 2
    });
}

function districtKey(name) {
    return String(name || "")
        .toLowerCase()
        .replace(/&/g, "and")
        .replace(/[^a-z0-9]/g, "");
}

function showError(message) {
    console.error(message);

    const container = document.querySelector(".main");

    if (!container) return;

    const notice = document.createElement("div");

    notice.style.cssText =
        "background:#fee2e2;color:#991b1b;padding:15px;margin-bottom:20px;border-radius:8px;";

    notice.textContent = message;
    container.prepend(notice);
}


// ==========================================
// API DATA LOADING
// ==========================================

async function fetchAllAPIRecords(endpoint, pageSize = 1000) {
    let allRecords = [];
    let offset = 0;
    let total = null;

    while (total === null || offset < total) {
        const response = await fetch(
            `${API_PATH}${endpoint}?limit=${pageSize}&offset=${offset}`
        );

        if (!response.ok) {
            throw new Error(
                `API error for ${endpoint}: HTTP ${response.status}`
            );
        }

        const result = await response.json();

        if (!Array.isArray(result.data)) {
            throw new Error(
                `Invalid API response from ${endpoint}`
            );
        }

        allRecords = allRecords.concat(result.data);
        total = Number(result.total);

        if (!Number.isFinite(total)) {
            throw new Error(
                `Invalid total record count from ${endpoint}`
            );
        }

        offset += pageSize;
    }

    return allRecords;
}


// ==========================================
// STATIC GEOJSON LOADING
// ==========================================

async function fetchGeoJSON() {
    const response = await fetch(
        DATA_PATH + "bangladesh_districts.geojson"
    );

    if (!response.ok) {
        throw new Error(
            `Could not load Bangladesh GeoJSON: HTTP ${response.status}`
        );
    }

    return response.json();
}


// ==========================================
// LOAD ALL DASHBOARD DATA
// ==========================================

async function loadData() {
    console.log("Loading EpiScope datasets through FastAPI...");

    const [
        weekly,
        districtSummary,
        originalAlerts,
        improvedAlerts,
        forecasts,
        resourceSummary,
        facilities,
        geojson
    ] = await Promise.all([
        fetchAllAPIRecords("trends", 5000),
        fetchAllAPIRecords("districts"),
        fetchAllAPIRecords("alerts/original"),
        fetchAllAPIRecords("alerts"),
        fetchAllAPIRecords("forecasts"),
        fetchAllAPIRecords("resource-summary"),
        fetchAllAPIRecords("facility-allocations"),
        fetchGeoJSON()
    ]);

    state.weekly = weekly;
    state.districtSummary = districtSummary;
    state.originalAlerts = originalAlerts;
    state.improvedAlerts = improvedAlerts;
    state.forecasts = forecasts;
    state.resourceSummary = resourceSummary;
    state.facilities = facilities;

    console.log("EpiScope datasets loaded successfully:", {
        source: "FastAPI",
        weekly: state.weekly.length,
        districtSummary: state.districtSummary.length,
        originalAlerts: state.originalAlerts.length,
        improvedAlerts: state.improvedAlerts.length,
        forecasts: state.forecasts.length,
        resourceSummary: state.resourceSummary.length,
        facilityAllocations: state.facilities.length
    });

    return geojson;
}


// ==========================================
// NATIONAL OVERVIEW
// ==========================================

function updateOverview() {
    const disease = document.getElementById("disease-filter").value;

    const filtered = disease === "All"
        ? state.weekly
        : state.weekly.filter(row => row.disease === disease);

    const totalCases = d3.sum(
        filtered,
        row => Number(row.total_cases)
    );

    document.getElementById("total-cases").textContent =
        formatNumber(totalCases);

    document.getElementById("total-districts").textContent = "64";

    document.getElementById("total-alerts").textContent =
        formatNumber(state.improvedAlerts.length);

    document.getElementById("total-forecasts").textContent =
        formatNumber(state.forecasts.length);
}


// ==========================================
// BANGLADESH DISTRICT MAP
// ==========================================

async function createMap(geojson) {
    const container = document.getElementById("bangladesh-map");

    if (!container) {
        throw new Error("Map container #bangladesh-map was not found.");
    }

    if (!geojson || !Array.isArray(geojson.features)) {
        throw new Error("Invalid Bangladesh GeoJSON.");
    }

    container.innerHTML = "";

    const width = 650;
    const height = 460;

    const projection = d3.geoIdentity()
        .reflectY(true)
        .fitExtent(
            [[20, 20], [width - 20, height - 20]],
            geojson
        );

    const path = d3.geoPath().projection(projection);

    const svg = d3.select(container)
        .append("svg")
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet")
        .style("width", "100%")
        .style("height", "460px");

    const mapGroup = svg.append("g");

    mapGroup.selectAll("path")
        .data(geojson.features)
        .enter()
        .append("path")
        .attr("class", "district-shape")
        .attr("d", path)
        .attr("fill", "#dbeafe")
        .attr("stroke", "#ffffff")
        .attr("stroke-width", 1)
        .attr("data-name", feature =>
            feature.properties.adm2_en ||
            feature.properties.ADM2_EN ||
            feature.properties.name ||
            "Unknown"
        )
        .on("click", function(event, feature) {
            const name =
                feature.properties.adm2_en ||
                feature.properties.ADM2_EN ||
                feature.properties.name ||
                "Unknown";

            state.selectedDistrict = name;

            updateMap();
            showDistrictDetails(name);
        })
        .append("title")
        .text(feature =>
            feature.properties.adm2_en ||
            feature.properties.ADM2_EN ||
            feature.properties.name ||
            "District"
        );

    updateMap();

    console.log("Bangladesh map rendered:", {
        districts: geojson.features.length
    });
}


// ==========================================
// DISTRICT CASES AND POPULATION
// ==========================================

function getDistrictCases() {
    const disease = document.getElementById("map-disease-filter").value;

    const filtered = disease === "All"
        ? state.districtSummary
        : state.districtSummary.filter(
            row => row.disease === disease
        );

    const populationByDistrict = new Map();

    state.weekly.forEach(row => {
        const key = districtKey(row.district);
        const population = Number(row.total_population || 0);

        if (
            key &&
            population > 0 &&
            !populationByDistrict.has(key)
        ) {
            populationByDistrict.set(key, population);
        }
    });

    const result = new Map();

    filtered.forEach(row => {
        const key = districtKey(row.district);

        const current = result.get(key) || {
            cases: 0,
            admissions: 0,
            population:
                Number(row.total_population || 0) ||
                populationByDistrict.get(key) ||
                0,
            district: row.district
        };

        current.cases += Number(row.total_cases || 0);
        current.admissions += Number(row.admissions || 0);

        result.set(key, current);
    });

    return result;
}


// ==========================================
// UPDATE DISTRICT MAP
// ==========================================

function updateMap() {
    const values = getDistrictCases();

    const maximum = d3.max(
        Array.from(values.values()),
        item => item.cases
    ) || 1;

    const color = d3.scaleSequential()
        .domain([0, maximum])
        .interpolator(d3.interpolateBlues);

    d3.selectAll("#bangladesh-map .district-shape")
        .attr("fill", feature => {
            const name =
                feature.properties.adm2_en ||
                feature.properties.ADM2_EN ||
                feature.properties.name;

            const item = values.get(districtKey(name));

            return item ? color(item.cases) : "#e2e8f0";
        })
        .classed("selected", feature => {
            const name =
                feature.properties.adm2_en ||
                feature.properties.ADM2_EN ||
                feature.properties.name;

            return districtKey(name) ===
                districtKey(state.selectedDistrict);
        })
        .select("title")
        .text(function() {
            const name = this.parentNode.getAttribute("data-name");
            const item = values.get(districtKey(name));

            return `${name}: ${formatNumber(item?.cases || 0)} cases`;
        });
}


// ==========================================
// DISTRICT DETAILS
// ==========================================

function showDistrictDetails(name) {
    const values = getDistrictCases();
    const item = values.get(districtKey(name));

    document.getElementById("selected-district").textContent = name;

    if (!item) {
        document.getElementById("district-hint").textContent =
            "No matching district statistics found.";

        document.getElementById("district-details").innerHTML = "";
        return;
    }

    document.getElementById("district-hint").textContent =
        "Aggregated historical surveillance statistics.";

    const casesPer100k = item.population > 0
        ? item.cases / item.population * 100000
        : 0;

    document.getElementById("district-details").innerHTML = `
        <div class="district-metric">
            <span>Total recorded cases</span>
            <strong>${formatNumber(item.cases)}</strong>
        </div>

        <div class="district-metric">
            <span>Admissions</span>
            <strong>${formatNumber(item.admissions)}</strong>
        </div>

        <div class="district-metric">
            <span>Cases per 100,000 population</span>
            <strong>${formatDecimal(casesPer100k)}</strong>
        </div>

        <div class="district-metric">
            <span>Population reference</span>
            <strong>WorldPop 2022 estimate</strong>
        </div>
    `;
}


// ==========================================
// DISEASE TREND CHART
// ==========================================

function updateTrendChart() {
    const disease = document.getElementById("disease-filter").value;

    const filtered = disease === "All"
        ? state.weekly
        : state.weekly.filter(row => row.disease === disease);

    const grouped = d3.rollup(
        filtered,
        rows => d3.sum(
            rows,
            row => Number(row.total_cases)
        ),
        row => row.week_start
    );

    const chartData = Array.from(
        grouped,
        ([week, cases]) => ({ week, cases })
    ).sort((a, b) => a.week.localeCompare(b.week));

    const labels = chartData.map(row => row.week);
    const values = chartData.map(row => row.cases);

    const ctx = document.getElementById("trend-chart");

    if (state.trendChart) {
        state.trendChart.destroy();
    }

    state.trendChart = new Chart(ctx, {
        type: "line",

        data: {
            labels,

            datasets: [{
                label: disease === "All"
                    ? "All disease cases"
                    : disease + " cases",

                data: values,
                borderColor: "#2563eb",
                backgroundColor: "rgba(37, 99, 235, 0.10)",
                borderWidth: 2,
                fill: true,
                tension: 0.25,
                pointRadius: 0,
                pointHoverRadius: 4
            }]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,

            interaction: {
                intersect: false,
                mode: "index"
            },

            plugins: {
                legend: {
                    display: true
                }
            },

            scales: {
                x: {
                    ticks: {
                        maxTicksLimit: 12,
                        maxRotation: 0
                    }
                },

                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: "Recorded cases"
                    }
                }
            }
        }
    });
}


// ==========================================
// OUTBREAK ALERTS
// ==========================================

function updateAlerts() {
    const detector = document.getElementById("detector-filter").value;

    const alerts = detector === "original"
        ? state.originalAlerts
        : state.improvedAlerts;

    const sorted = [...alerts].sort((a, b) =>
        String(b.week_start).localeCompare(String(a.week_start))
    );

    const affectedDistricts = new Set(
        alerts.map(row => row.district)
    );

    document.getElementById("alert-count").textContent =
        formatNumber(alerts.length);

    document.getElementById("alert-districts").textContent =
        formatNumber(affectedDistricts.size);

    document.getElementById("detector-name").textContent =
        detector === "original" ? "Original" : "Improved";

    const table = document.getElementById("alert-table");

    table.innerHTML = "";

    sorted.slice(0, 100).forEach(row => {
        const tr = document.createElement("tr");

        const reason = detector === "improved"
            ? row.alert_reason || "Recent or seasonal anomaly"
            : row.detection_method || "Rolling baseline anomaly";

        [
            row.district,
            row.disease,
            row.week_start,
            formatNumber(row.total_cases),
            reason
        ].forEach(value => {
            const td = document.createElement("td");
            td.textContent = value;
            tr.appendChild(td);
        });

        table.appendChild(tr);
    });
}


// ==========================================
// FORECASTS
// ==========================================

function updateForecasts() {
    const disease = document.getElementById("forecast-disease").value;

    const filtered = disease === "All"
        ? state.forecasts
        : state.forecasts.filter(row => row.disease === disease);

    const sorted = [...filtered].sort((a, b) =>
        String(a.forecast_week).localeCompare(String(b.forecast_week))
    );

    const table = document.getElementById("forecast-table");

    table.innerHTML = "";

    sorted.forEach(row => {
        const tr = document.createElement("tr");

        [
            row.district,
            row.disease,
            row.forecast_week,
            formatNumber(row.predicted_cases),
            formatNumber(row.lower_bound),
            formatNumber(row.upper_bound)
        ].forEach(value => {
            const td = document.createElement("td");
            td.textContent = value;
            tr.appendChild(td);
        });

        table.appendChild(tr);
    });
}


// ==========================================
// RESOURCE PLANNING
// ==========================================

function updateResources() {
    const rows = state.resourceSummary;

    const totalDemand = d3.sum(rows, row =>
        Number(row.total_demand)
    );

    const totalAllocated = d3.sum(rows, row =>
        Number(row.total_allocated)
    );

    const totalUnmet = d3.sum(rows, row =>
        Number(row.total_unmet)
    );

    const fulfilment = totalDemand > 0
        ? totalAllocated / totalDemand * 100
        : 0;

    document.getElementById("resource-summary").innerHTML = `
        <div>
            <span>Total simulated demand</span>
            <strong>${formatNumber(totalDemand)}</strong>
        </div>

        <div>
            <span>Total allocated quantity</span>
            <strong>${formatNumber(totalAllocated)}</strong>
        </div>

        <div>
            <span>Aggregate fulfilment ratio</span>
            <strong>${formatDecimal(fulfilment)}%</strong>
        </div>
    `;

    const sorted = [...rows].sort((a, b) =>
        Number(a.fulfilment_rate) - Number(b.fulfilment_rate)
    );

    const table = document.getElementById("resource-table");

    table.innerHTML = "";

    sorted.forEach(row => {
        const tr = document.createElement("tr");

        const percentage =
            Number(row.fulfilment_rate) * 100;

        [
            row.district,
            row.resource,
            formatNumber(row.total_demand),
            formatNumber(row.total_allocated),
            formatNumber(row.total_unmet),
            formatDecimal(percentage) + "%"
        ].forEach(value => {
            const td = document.createElement("td");
            td.textContent = value;
            tr.appendChild(td);
        });

        table.appendChild(tr);
    });
}


// ==========================================
// FILTER EVENTS
// ==========================================

function setupFilters() {
    document.getElementById("disease-filter")
        .addEventListener("change", () => {
            updateOverview();
            updateTrendChart();
        });

    document.getElementById("map-disease-filter")
        .addEventListener("change", () => {
            updateMap();

            if (state.selectedDistrict) {
                showDistrictDetails(state.selectedDistrict);
            }
        });

    document.getElementById("detector-filter")
        .addEventListener("change", updateAlerts);

    document.getElementById("forecast-disease")
        .addEventListener("change", updateForecasts);
}


// ==========================================
// APPLICATION STARTUP
// ==========================================

async function startDashboard() {
    try {
        document.getElementById("current-date").textContent =
            new Date().toLocaleDateString("en-GB", {
                day: "2-digit",
                month: "short",
                year: "numeric"
            });

        console.log("Starting EpiScope dashboard...");

        const geojson = await loadData();

        await createMap(geojson);

        updateOverview();
        updateTrendChart();
        updateAlerts();
        updateForecasts();
        updateResources();

        setupFilters();

        console.log("EpiScope dashboard initialized successfully.");

    } catch (error) {
        showError(
            "Dashboard could not load its data. Please check the local server and browser console. " +
            error.message
        );
    }
}

document.addEventListener("DOMContentLoaded", startDashboard);
