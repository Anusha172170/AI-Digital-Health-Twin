# 🧬 AI Digital Health Twin for Predictive Healthcare

> **Department of CSE - Artificial Intelligence & Machine Learning**  
> **Vasireddy Venkatadri Institute of Technology (VVIT)**  
> **Major Project | Batch ID: AIML-C-13**

---

## 📌 Project Overview

An **AI-Based Digital Health Twin** that creates a dynamic, personalized virtual replica of a patient's health by continuously collecting physiological data from IoMT devices (Heart Rate, SpO₂, Blood Pressure, Glucose), applying Machine Learning to predict multi-disease risks (Cardiovascular Disease, Type-2 Diabetes, Hypertension), and enabling "what-if" lifestyle trajectory simulations for preventive healthcare.

---

## 🏗️ System Architecture

The system follows a **4-Layer Distributed Architecture**:

```mermaid
flowchart TB
    subgraph L1[" Layer 1: Data Acquisition (IoMT Layer) "]
        direction LR
        S1[Pulse Oximeter]
        S2[Heart Rate Monitor]
        S3[Glucose Sensor]
        S4[ESP32 / Vitals Simulator]
    end

    subgraph L2[" Layer 2: Presentation Layer (Mobile App) "]
        direction TB
        UI1[Twin Avatar Dashboard]
        UI2[AI Disease Risk & XAI]
        UI3[What-If Trajectory Simulator]
    end

    subgraph L3[" Layer 3: Backend & AI Engine (FastAPI) "]
        direction TB
        API[FastAPI REST Services]
        ML[XGBoost & Scikit-Learn Models]
        XAI[SHAP Explainable AI Engine]
        SIM[Biological Age Engine]
    end

    subgraph L4[" Layer 4: Cloud & Database Layer "]
        direction LR
        DB[(MongoDB Atlas)]
        FB[(Firebase Auth & FCM)]
    end

    S1 & S2 & S3 --> S4
    S4 -->|HTTP / Vitals Stream| API
    UI1 & UI2 & UI3 <-->|REST API Calls| API
    API --> ML --> XAI
    API --> SIM
    API <--> DB
    API --> FB
