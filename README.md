\# EpiScope

\## Big Data Surveillance, Forecasting and Response-Planning Platform for Bangladesh



EpiScope is a big data analytics platform designed to demonstrate how large-scale health encounter data can support disease surveillance, outbreak detection, historical forecasting, and healthcare resource planning across Bangladesh.



The platform combines real geographic, population, and weather data with synthetic health encounter, facility, and stock datasets. It uses Apache Spark for distributed data processing, a Bronze-Silver-Gold data lake architecture, analytical models, FastAPI, and a D3.js dashboard.



> \*\*Important:\*\* EpiScope is an academic demonstration platform. Encounter, facility, and stock records are synthetic. Forecasts are retrospective historical forecasts, not live predictions. This platform is not an operational public-health surveillance service or clinical decision-support system.



\---



\## 1. Project Objectives



\- Process more than 25 million synthetic health encounter records.

\- Build a scalable Bronze, Silver, and Gold data lake.

\- Integrate geographic, population, and weather information.

\- Detect unusual disease activity at district level.

\- Generate four-week historical forecasts.

\- Demonstrate healthcare resource allocation.

\- Provide an interactive dashboard.

\- Expose processed results through a REST API.



\## 2. Technology Stack



| Component | Technology |

|---|---|

| Programming | Python, JavaScript |

| Big data processing | Apache Spark, PySpark |

| Storage | Parquet, JSON, CSV |

| Data architecture | Bronze, Silver, Gold |

| API | FastAPI, Uvicorn |

| Dashboard | HTML, CSS, JavaScript, D3.js |

| Geographic data | GeoJSON |

| Optimization | PuLP |

| Development environment | Windows, Python 3.11 |



\## 3. System Architecture



```text

&#x20;                DATA SOURCES

&#x20;                     |

&#x20;       +-------------+-------------+

&#x20;       |             |             |

&#x20;   Synthetic       Real          Real

&#x20;   Encounters    Population     Weather

&#x20;       |             |             |

&#x20;       +-------------+-------------+

&#x20;                     |

&#x20;                BRONZE LAYER

&#x20;               Raw data (Parquet)

&#x20;                     |

&#x20;                     v

&#x20;                SILVER LAYER

&#x20;         Cleaning, validation, deduplication,

&#x20;         quarantine and privacy handling

&#x20;                     |

&#x20;                     v

&#x20;                 GOLD LAYER

&#x20;         Weekly district-disease aggregates

&#x20;                     |

&#x20;          +----------+-----------+

&#x20;          |          |           |

&#x20;          v          v           v

&#x20;       Outbreak  Forecasting   Resource

&#x20;       Detection Analysis     Optimization

&#x20;          |          |           |

&#x20;          +----------+-----------+

&#x20;                     |

&#x20;              PROCESSED OUTPUTS

&#x20;                     |

&#x20;                FASTAPI SERVICE

&#x20;                     |

&#x20;                D3.JS DASHBOARD

```



\## 4. Data Sources and Classification



EpiScope combines real contextual datasets with synthetic operational datasets.



| Dataset | Classification | Purpose |

|---|---|---|

| Bangladesh district boundaries | Real | Geographic visualization |

| WorldPop 2022 population estimates | Real | Population context |

| Open-Meteo ERA5 historical weather | Real | Weather context |

| Health encounters | Synthetic | Disease analysis |

| Healthcare facilities | Synthetic | Facility planning |

| Facility resource stock | Synthetic | Allocation simulation |



\### Data Scale



\- Raw encounter records: \*\*25,500,966\*\*

\- Cleaned encounter records: \*\*24,379,792\*\*

\- Quarantined records: \*\*632,636\*\*

\- Duplicate valid records removed: \*\*488,538\*\*

\- Bangladesh districts: \*\*64\*\*

\- Synthetic healthcare facilities: \*\*640\*\*

\- Facility stock snapshots: \*\*1,168,640\*\*

\- Weekly district-disease aggregates: \*\*83,840\*\*



\## 5. Data Processing Pipeline



\### Bronze Layer



\- Stores raw encounter data in Parquet format.

\- Preserves the initial ingestion layer.



\### Silver Layer



\- Validates records.

\- Handles missing and invalid values.

\- Removes duplicate valid encounters.

\- Separates invalid records into quarantine.

\- Hashes and removes raw patient identifiers to reduce privacy risk.



\### Gold Layer



\- Aggregates cleaned records by district, disease, and week.

\- Integrates population and weather context.

\- Produces datasets for analytics and visualization.



\## 6. Analytical Modules



\### 6.1 Outbreak Detection



The outbreak detection module identifies unusual disease activity using historical patterns and recent observations.



The project includes original and improved alert configurations. The improved configuration generates more candidate alerts, but evaluation indicates that alert precision remains limited. Alerts should therefore be interpreted as exploratory signals, not confirmed outbreaks.



\### 6.2 Disease Forecasting



The forecasting module generates four-week historical forecasts for district-disease combinations.



\- Forecast origin: \*\*2025-12-22\*\*

\- Historical data period: \*\*2021–2025\*\*

\- Forecast horizon: \*\*4 weeks\*\*

\- Forecast records: \*\*1,024\*\*



These are retrospective forecasts based on the project dataset. They are not current forecasts for 2026.



\### 6.3 Resource Allocation



The optimization module demonstrates resource distribution using simulated demand and facility stock.



| Metric | Result |

|---|---:|

| Facilities | 640 |

| Allocation records | 6,400 |

| Simulated demand | 93,556.73 |

| Allocated quantity | 89,254.19 |

| Unmet demand | 4,302.54 |

| Quantity allocation ratio | Approximately 95.4% |



The allocation ratio represents allocated quantity divided by simulated demand. It does not represent the percentage of patients covered. The model uses illustrative assumptions and simulated inventory.



\## 7. Interactive Dashboard



The interactive dashboard provides:



\- Bangladesh district map.

\- District-level disease information.

\- Disease trend visualization.

\- Outbreak alert exploration.

\- Historical forecast visualization.

\- Resource allocation summaries.

\- Facility allocation information.



The dashboard uses D3.js for data visualization and GeoJSON for geographic boundaries.



\## 8. FastAPI Service



The FastAPI application provides REST endpoints for accessing processed project data.



\*\*Base URL:\*\*



`http://127.0.0.1:8001`



\*\*Interactive API documentation:\*\*



`http://127.0.0.1:8001/docs`



| Endpoint | Description |

|---|---|

| `/api/health` | API and data-file availability |

| `/api/overview` | Project summary |

| `/api/districts` | District-disease summaries |

| `/api/alerts` | Improved outbreak alerts |

| `/api/alerts/original` | Original outbreak alerts |

| `/api/forecasts` | Historical forecasts |

| `/api/trends` | Weekly disease trends |

| `/api/resources` | Facility resource allocations |

| `/api/resource-summary` | District resource summaries |

| `/api/facility-allocations` | Facility-level allocations |

| `/api/diseases` | Disease categories |

| `/api/district-names` | District names |



\## 9. Running the Project on Windows



\### Prerequisites



\- Python 3.11

\- Java configured for the Spark environment

\- Apache Spark

\- Project dependencies installed



\### Step 1: Activate the Environment



Open a terminal:



```cmd

D:

cd \\EpiScope

venv311\\Scripts\\activate

```



\### Step 2: Start FastAPI



Open the first terminal:



```cmd

D:

cd \\EpiScope

venv311\\Scripts\\activate

uvicorn serving.app:app --reload --host 127.0.0.1 --port 8001

```



Open:



`http://127.0.0.1:8001/docs`



\### Step 3: Start the Dashboard



Open a second terminal:



```cmd

D:

cd \\EpiScope

python -m http.server 8000 --directory dashboard

```



Open:



`http://localhost:8000`



Keep both terminal windows running while using the dashboard.



The API expects processed output files at the locations configured in `serving/app.py`. The dashboard loads its geographic boundary file from its static data directory.



\## 10. Limitations and Responsible Use



\- Most encounter, facility, and stock records are synthetic.

\- Population and weather datasets provide contextual information.

\- Outbreak alerts are exploratory and require further validation.

\- Forecasts are retrospective, not current operational predictions.

\- Resource allocation uses illustrative assumptions and simulated inventory.

\- The platform is intended for academic demonstration and research.



\## 11. Future Improvements



\- Improve outbreak alert precision and evaluation.

\- Enhance forecasting accuracy and model calibration.

\- Optimize data retrieval and API performance.

\- Improve resource allocation assumptions.

\- Extend dashboard functionality and reporting.



\---



\*\*Project Name:\*\* EpiScope  

\*\*Domain:\*\* Big Data Analytics and Public Health  

\*\*Geographic Focus:\*\* Bangladesh  

\*\*Purpose:\*\* Academic Project and Demonstration Platform

