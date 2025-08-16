# 🧵 GenFashion Supply Chain Copilot

An **agentic AI assistant** that helps fashion designers and brands find the best suppliers based on:
- **Product descriptions**
- **Cost, delivery time, and sustainability priorities**
- **Required certifications**
- **Destination Country of Import**

Built with:
- **LangChain / CrewAI** for agent reasoning
- **Google Sheets** for supplier database
- **Gradio** for an interactive UI

---

## 🚀 Features
- Extracts **structured specs** from natural language product descriptions
- Matches suppliers based on **certifications** and **capabilities**
- Estimates **shipping costs and delivery times**
- Ranks suppliers according to your priorities
- User-friendly **Gradio interface**

---

## 🛠 Architecture
User → Gradio UI → run_pipeline_with_agent()

   ├─> Spec Extraction Agent (LLM)
   
   ├─> Google Sheets Supplier Match
   
   ├─> Logistics Estimator
   
   └─> Ranking & Output
   
---

## Agent Design & Prompt Engineering

The GenFashion Supply Chain Copilot leverages a specialized agent layer to extract structured product specifications from free-text user input. This allows downstream supplier matching, logistics estimation, and ranking to be fully data-driven.

#### 1. Product Spec Extraction Agent
Agent Role: Product Spec Extraction Expert

Goal: Extract a clean JSON object with the following keys:
product_type, material, color, weight, season

Backstory / Instructions:
- Convert noisy product descriptions into a normalized JSON spec.
- Return only valid JSON with exactly the five keys.
- If a field is missing, set it to an empty string ("").
- No markdown, no commentary, no extra text.


This ensures consistency for downstream deterministic steps: material-based supplier matching, certification filtering, shipping calculation, and ranking.

## 📦 Setup

### 1. Clone the Repository
git clone https://github.com/YOUR-USERNAME/genfashion-supply-chain-copilot.git
cd genfashion-supply-chain-copilot
### 2. Install Dependencies
pip install -r requirements.txt
### 3. Configure Environment Variables
Create a .env file:

- OPENAI_API_KEY=<your_openai_key> 
- GSHEET_KEYFILE=genfashion-backend-key.json
- SHEET_NAME=<your_google_sheet_name>
- SUPPLIERS_TAB=Suppliers
- SHIPPING_TAB=Shipping
- FEEDBACK_TAB=Feedback  

Upload .env and your Google Service Account JSON key to your environment.
### 4. Run the Application
Local environment:
python app.py
