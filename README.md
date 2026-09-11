# AI Digital Health Twin

## System Architecture

The project follows a layered architecture connecting IoMT data sources, a mobile client, backend services, machine-learning components, and cloud storage.

```mermaid
flowchart TB
    subgraph Acquisition["1. IoMT Data Acquisition Layer"]
        HR[Heart Rate Sensor]
        SPO2[SpO2 Sensor]
        BP[Blood Pressure Monitor]
        GLU[Glucose Monitor]
        SIM[Sensor Data Simulator]
        ESP[ESP32 Gateway]
    end

    subgraph Client["2. Client Layer"]
        APP[Mobile Application]
        AUTH[Firebase Authentication]
        NOTIFY[Push Notification Handler]
    end

    subgraph Backend["3. Backend Service Layer"]
        API[FastAPI REST API]
        VALIDATE[Data Validation]
        VITALS[Vitals Processing Service]
        TWIN[Digital Twin Service]
        ALERT[Alert Detection Service]
    end

    subgraph Intelligence["4. AI and Simulation Layer"]
        PRE[Data Preprocessing]
        ML[Machine Learning Models]
        XAI[SHAP Explainability]
        WHATIF[What-If Simulation Engine]
    end

    subgraph Cloud["5. Cloud and Storage Layer"]
        DB[(MongoDB Atlas)]
        FIREBASE[Firebase Auth and FCM]
    end

    HR --> ESP
    SPO2 --> ESP
    BP --> ESP
    GLU --> ESP
    SIM --> API
    ESP -->|HTTP or MQTT| API

    APP -->|Authentication| AUTH
    AUTH --> FIREBASE
    APP -->|REST API Requests| API

    API --> VALIDATE
    VALIDATE --> VITALS
    VITALS --> PRE
    PRE --> ML
    ML --> XAI
    ML --> TWIN
    XAI --> TWIN

    APP -->|Lifestyle Parameters| WHATIF
    WHATIF --> ML
    WHATIF -->|Simulation Result| APP

    VITALS --> DB
    TWIN --> DB
    API --> DB

    VITALS --> ALERT
    ALERT -->|Critical Event| FIREBASE
    FIREBASE -->|FCM Notification| NOTIFY
