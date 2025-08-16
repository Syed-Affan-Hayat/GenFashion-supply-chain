# =======================
# 0) Install dependencies
# =======================
!pip install -q "crewai[tools]" langchain langchain-community langchain-openai openai python-dotenv gspread oauth2client gradio pandas

# =================
# 1) Core setup
# =================
from google.colab import files
import os, json, re, traceback
from dotenv import load_dotenv

# ---- Upload .env and the Google Service Account JSON ----
# Expecting you to upload:
#  - .env  (with OPENAI_API_KEY, GSHEET_KEYFILE, SHEET_NAME, SUPPLIERS_TAB, SHIPPING_TAB, FEEDBACK_TAB)
#  - <your_service_account>.json (matches GSHEET_KEYFILE in .env, typically "genfashion-backend-key.json")
uploaded = files.upload()
for filename in uploaded.keys():
    if filename.endswith(".json"):
        # Normalize the keyfile name to what's in .env (we'll set after loading .env too)
        os.rename(filename, "genfashion-backend-key.json")
    elif filename == ".env" or filename.endswith(".env"):
        if filename != ".env":
            os.rename(filename, ".env")

# Load .env (do not print secrets)
load_dotenv(".env", override=True)

REQUIRED_VARS = [
    "OPENAI_API_KEY", "GSHEET_KEYFILE", "SHEET_NAME",
    "SUPPLIERS_TAB", "SHIPPING_TAB"
]
missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
assert not missing, f"Missing required env vars: {missing}"

# Optional but recommended
if not os.getenv("FEEDBACK_TAB"):
    os.environ["FEEDBACK_TAB"] = "Feedback"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GSHEET_KEYFILE = os.getenv("GSHEET_KEYFILE")  # should be "genfashion-backend-key.json"
SHEET_NAME     = os.getenv("SHEET_NAME")
SUPPLIERS_TAB  = os.getenv("SUPPLIERS_TAB")
SHIPPING_TAB   = os.getenv("SHIPPING_TAB")
FEEDBACK_TAB   = os.getenv("FEEDBACK_TAB")
GRADIO_SHARE   = os.getenv("GRADIO_SHARE", "True").lower() == "true"
DEBUG_MODE     = os.getenv("DEBUG", "True").lower() == "true"

# =========================
# 2) Google Sheets client
# =========================
import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(GSHEET_KEYFILE, scope)
gc = gspread.authorize(creds)

# Open worksheets
sh = gc.open(SHEET_NAME)
suppliers_ws = sh.worksheet(SUPPLIERS_TAB)
shipping_ws  = sh.worksheet(SHIPPING_TAB)

# Ensure Feedback tab exists with headers exactly once
def ensure_feedback_tab():
    try:
        ws = sh.worksheet(FEEDBACK_TAB)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=FEEDBACK_TAB, rows="1000", cols="20")
        ws.append_row([
            "timestamp", "rating", "comment", "product_desc",
            "cost", "speed", "sustain", "certifications", "destination"
        ])
    return ws

feedback_ws = ensure_feedback_tab()

# =================
# 3) Data helpers
# =================
def get_suppliers_from_sheet():
    rows = suppliers_ws.get_all_records()
    for r in rows:
        r["material"]  = str(r.get("material","")).strip().lower()
        r["supplier"]  = str(r.get("supplier","")).strip()
        r["location"]  = str(r.get("location","")).strip()
        certs = str(r.get("certifications",""))
        r["certifications"] = [c.strip() for c in certs.split(",") if c.strip()]
        # numeric coercions
        try:
            r["cost_per_unit"] = float(r.get("cost_per_unit", 0) or 0)
        except Exception:
            r["cost_per_unit"] = 0.0
        try:
            r["lead_time_days"] = int(r.get("lead_time_days", 0) or 0)
        except Exception:
            r["lead_time_days"] = 0
    return rows

def load_shipping_matrix():
    rows = shipping_ws.get_all_records()
    out = {}
    for row in rows:
        origin = str(row.get("origin","")).strip()
        dest   = str(row.get("destination","")).strip()
        try:
            days = int(row.get("days", 0) or 0)
        except Exception:
            days = 0
        out[(origin, dest)] = days
    return out

SHIPPING_MATRIX = load_shipping_matrix()

def filter_by_certification(suppliers, required_certs):
    if not required_certs:
        return suppliers
    req = set(map(str.strip, required_certs))
    return [s for s in suppliers if req.intersection(set(s.get("certifications", [])))]

def _normalize(x, lo, hi):
    return (x - lo) / (hi - lo) if hi != lo else 1.0

def get_certifications_from_sheet():
    rows = suppliers_ws.get_all_records()
    all_certs = set()
    for r in rows:
        certs = str(r.get("certifications",""))
        all_certs.update([c.strip() for c in certs.split(",") if c.strip()])
    return sorted(all_certs)

def get_destinations_from_matrix():
    rows = shipping_ws.get_all_records()
    return sorted(set(str(row.get("destination","")).strip() for row in rows if str(row.get("destination","")).strip()))

CERT_OPTIONS = get_certifications_from_sheet()
DEST_OPTIONS = get_destinations_from_matrix()

# ======================================
# 4) LLM (for spec extraction) + CrewAI
# ======================================
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)

spec_agent = Agent(
    role="Product Spec Extraction Expert",
    goal="Extract JSON spec with keys: product_type, material, color, weight, season.",
    backstory=(
        "Convert noisy product descriptions into a clean JSON spec. "
        "Never add commentary. If a field is missing, set it to ''. "
        "Return ONLY valid JSON with exactly those five keys."
    ),
    llm=llm,
    verbose=False
)

import json as _json
def _parse_json_loose(text: str):
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in agent output.")
    obj = _json.loads(m.group(0))
    for k in ["product_type","material","color","weight","season"]:
        obj.setdefault(k, "")
    obj["material"] = str(obj["material"]).strip().lower()
    return obj

def extract_spec_via_agent(description: str) -> dict:
    task = Task(
        description=(
            "Extract ONLY a JSON object with EXACT keys: "
            "product_type, material, color, weight, season. "
            "No markdown, no commentary. Missing values -> ''.\n\n"
            f"Description: {description}"
        ),
        expected_output="A single JSON object with the five keys.",
        agent=spec_agent
    )
    crew = Crew(agents=[spec_agent], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return _parse_json_loose(str(result).strip())

# ======================================================
# 5) Deterministic steps: match, logistics, ranking
# ======================================================
def supplier_match_step(product_spec: dict, required_certs):
    rows = get_suppliers_from_sheet()
    target_words = product_spec.get("material", "").lower().split()
    if not target_words:
        return []
    matches = []
    for s in rows:
        words = s["material"].split()
        if all(w in words for w in target_words):
            matches.append(s)
    return filter_by_certification(matches, required_certs)

def logistics_step(suppliers: list, destination: str):
    for s in suppliers:
        key = (s["location"], destination)
        ship_days = SHIPPING_MATRIX.get(key, 10)
        s["shipping_days"] = ship_days
        s["total_delivery_time"] = int(s["lead_time_days"]) + int(ship_days)
    return suppliers

def ranking_step(suppliers: list, weights: dict):
    if not suppliers:
        return []
    costs = [s["cost_per_unit"] for s in suppliers]
    times = [s["total_delivery_time"] for s in suppliers]
    certs = [len(s["certifications"]) for s in suppliers]
    minc, maxc = min(costs), max(costs)
    mint, maxt = min(times), max(times)
    minx, maxx = min(certs), max(certs)
    ranked=[]
    for s in suppliers:
        norm_cost = 1 - _normalize(s["cost_per_unit"], minc, maxc)
        norm_time = 1 - _normalize(s["total_delivery_time"], mint, maxt)
        norm_cert = _normalize(len(s["certifications"]), minx, maxx)
        score = (
            norm_cost * float(weights.get("cost",0.5)) +
            norm_time * float(weights.get("time",0.3)) +
            norm_cert * float(weights.get("sustainability",0.2))
        )
        s = dict(s)
        s["score"] = round(float(score), 3)
        ranked.append(s)
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked

# =================
# 6) Full pipeline
# =================
import pandas as pd
from datetime import datetime

def run_pipeline_with_agent(description, cost, speed, sustain, certifications, destination):
    trace_lines = []
    def t(line):
        trace_lines.append(line)
        if DEBUG_MODE:
            print(line)

    try:
        t("🚀 Pipeline starting...")
        specs = extract_spec_via_agent(description)
        t(f"✅ Specs: {specs}")

        suppliers = supplier_match_step(specs, certifications)
        suppliers = logistics_step(suppliers, destination)
        ranked = ranking_step(suppliers, {"cost": cost, "time": speed, "sustainability": sustain})

        if not ranked:
            return (
                pd.DataFrame(),
                "⚠️ No matching suppliers found.",
                ("### Agent Reasoning\n\n" + "\n".join(trace_lines)) if DEBUG_MODE else ""
            )

        df = pd.DataFrame([
            {
                "Rank": i+1,
                "Supplier": s["supplier"],
                "Material": s["material"],
                "Cost per Unit ($)": s["cost_per_unit"],
                "Lead Time (days)": s["lead_time_days"],
                "Certifications": ", ".join(s["certifications"]),
                "Total Delivery Time (days)": s["total_delivery_time"],
                "Score": s["score"]
            }
            for i, s in enumerate(ranked)
        ])

        # Summary stays clean, no debug info
        summary_md = f"✅ Found {len(df)} matching suppliers for your product."

        # Agent reasoning is only populated when DEBUG_MODE is True
        trace_md = ""
        if DEBUG_MODE:
            reasoning_block = (
                f"**Extracted Spec**: `{json.dumps(specs)}`\n\n"
                f"**Destination**: `{destination}`\n"
                f"**Weights**: cost={cost}, speed={speed}, sustainability={sustain}\n"
                f"**Found**: {len(df)} suppliers\n\n"
                "### Trace\n" + "\n".join(trace_lines)
            )
            trace_md = reasoning_block

        return df, summary_md, trace_md

    except Exception as e:
        return (
            pd.DataFrame([{"Error": str(e)}]),
            f"❌ Error: {e}",
            ("### Agent Reasoning\n\n" + "\n".join(trace_lines)) if DEBUG_MODE else ""
        )

# ==========================
# 7) Feedback Saving Helper
# ==========================
def handle_feedback(rating, comments, desc, cost, speed, sustain, certs, dest):
    """Append feedback to FEEDBACK_TAB in same sheet."""
    try:
        ws = ensure_feedback_tab()
        ws.append_row([
            datetime.now().isoformat(),
            rating,
            comments or "",
            desc or "",
            float(cost) if cost is not None else "",
            float(speed) if speed is not None else "",
            float(sustain) if sustain is not None else "",
            ", ".join(certs) if isinstance(certs, list) else (certs or ""),
            dest or ""
        ])
        return "✅ Feedback saved to Google Sheet. Thank you!"
    except Exception as e:
        return f"❌ Failed to save feedback: {e}"

# --------------
# 8) Gradio UI
# --------------
import gradio as gr

custom_css = """
/* polished layout */
#top-row .wrap.svelte-1ipelgc { display:flex !important; align-items:center !important; gap:20px; }
label.svelte-1ipelgc { font-weight:600; font-size:14px; }
input[type='range'] { height:6px; background:#eee; border-radius:8px; }
.gradio-container { max-width: 1150px; margin: auto; }
"""

with gr.Blocks(css=custom_css, title="GenFashion Supply Chain Copilot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align:center;'>🧥 GenFashion Supply Chain Copilot</h1>")
    gr.Markdown("<p style='text-align:center; color:gray;'>Find & rank suppliers based on product description, certifications, and priorities. The UI shows the agent trace and captures feedback.</p>")

    # ---- State (optional future use) ----
    results_state = gr.State()
    summary_state = gr.State()
    trace_state   = gr.State()

    # =============================
    # Tab 1: Product & Preferences
    # =============================
    with gr.Tab("📝 Product & Preferences"):
        with gr.Row():
            description = gr.Textbox(
                lines=3,
                value="Men's hoodie made of 300gsm organic cotton for winter",
                label="📝 Product Description",
                info="Describe product (material, weight, season, target fit). The agent extracts a normalized JSON spec."
            )
        with gr.Row(elem_id="top-row"):
            cost_slider    = gr.Slider(0, 1, value=0.5, step=0.1, label="💰 Cost Priority", info="Higher = prioritize cheaper suppliers.")
            speed_slider   = gr.Slider(0, 1, value=0.3, step=0.1, label="⚡ Speed Priority", info="Higher = prioritize lower total delivery time.")
            sustain_slider = gr.Slider(0, 1, value=0.2, step=0.1, label="🌱 Sustainability Priority", info="Higher = prioritize suppliers with certifications.")
        with gr.Row():
            certifications = gr.Dropdown(choices=CERT_OPTIONS, label="📜 Required Certifications", multiselect=True, info="Select one or more certifications required.")
            destination    = gr.Dropdown(choices=DEST_OPTIONS, label="🌍 Destination Country of Import", info="Destination used to compute shipping days.")

        run_btn = gr.Button("🚀 Find Suppliers", variant="primary")

        # Show results on this page too (your request)
        product_results = gr.Dataframe(
    label="📊 Matching Suppliers (Top results)",
    interactive=False,
    headers=["Supplier", "Cost", "Speed", "Sustainability", "Certifications", "Delivery Time"],
    datatype=["str", "number", "number", "number", "str", "str"]
)


        product_summary = gr.Markdown()

    # =============================
    # Tab 2: Agent Reasoning
    # =============================
    with gr.Tab("🧠 Agent Reasoning"):
        summary = gr.Markdown(label="Summary")
        trace_md = gr.Markdown(label="Agent Reasoning Trace (step-by-step)")
        reasoning_results = gr.Dataframe(
    label="📊 Matching Suppliers",
    interactive=False,
    headers=["Supplier", "Cost", "Speed", "Sustainability", "Certifications", "Delivery Time"],
    datatype=["str", "number", "number", "number", "str", "str"]
)


    # =============================
    # Tab 3: Feedback
    # =============================
    with gr.Tab("💬 Feedback"):
        gr.Markdown("Help improve the model and rankings — your feedback is saved to the project's Google Sheet.")
        with gr.Row():
            thumbs_up = gr.Button("👍 Helpful", elem_id="thumbs-up", variant="primary")
            thumbs_down = gr.Button("👎 Not helpful", elem_id="thumbs-down", variant="secondary")
        comments_box = gr.Textbox(label="Comments (optional)", lines=3, placeholder="What worked / what didn't?")
        feedback_status = gr.Markdown()

    # ---- Wiring: main pipeline ----
    def _run_and_fanout(desc, cost, speed, sustain, certs, dest):
        df, summ, trace = run_pipeline_with_agent(desc, cost, speed, sustain, certs, dest)
        # Return for: product tab (df + summary), reasoning tab (summary + trace + df), and states
        return (
            df,                                 # product_results
            summ,                               # product_summary
            summ,                               # summary (reasoning tab)
            trace,                              # trace_md (reasoning tab)
            df,                                 # reasoning_results
            df, summ, trace                     # update states
        )

    run_btn.click(
        fn=_run_and_fanout,
        inputs=[description, cost_slider, speed_slider, sustain_slider, certifications, destination],
        outputs=[product_results, product_summary, summary, trace_md, reasoning_results,
                 results_state, summary_state, trace_state]
    )

    # ---- Wiring: feedback ----
    thumbs_up.click(
        lambda comments, desc, cost, speed, sustain, certs, dest:
            handle_feedback("👍", comments, desc, cost, speed, sustain, certs, dest),
        inputs=[comments_box, description, cost_slider, speed_slider, sustain_slider, certifications, destination],
        outputs=[feedback_status]
    )

    thumbs_down.click(
        lambda comments, desc, cost, speed, sustain, certs, dest:
            handle_feedback("👎", comments, desc, cost, speed, sustain, certs, dest),
        inputs=[comments_box, description, cost_slider, speed_slider, sustain_slider, certifications, destination],
        outputs=[feedback_status]
    )

# Launch (Colab-friendly)
demo.launch(share=GRADIO_SHARE, debug=DEBUG_MODE)
