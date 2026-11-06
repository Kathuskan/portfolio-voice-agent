import os
import time
import base64
from io import BytesIO
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import edge_tts
import httpx

app = FastAPI(title="Kathuskan VoxFolio MCP — $0 RAG & Voice Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Knowledge Base (Grounded in your Portfolio & CV)
KNOWLEDGE_CHUNKS = [
    {
        "id": "autovalue",
        "source": "AutoValue_LK_README.md",
        "text": "AutoValue LK is an end-to-end machine learning web application estimating vehicle asking prices in Sri Lanka across 11,750 cleaned records. It uses engineered log-mileage and vehicle-age features with a tuned Random Forest regression pipeline served via FastAPI and React. Metrics: 84.95% predictions within ±20% asking price and held-out R-squared of 0.739.",
        "action": "OPEN_MODAL",
        "target": "autovalue"
    },
    {
        "id": "shelv",
        "source": "Shelv_MERN_README.md",
        "text": "Shelv is a full-stack MERN e-commerce marketplace exclusively for buying and selling new and used books. Built with React.js, Node.js, Express.js, MongoDB, and Stripe. Features 3-role RBAC (Buyer, Seller, Admin), Stripe card and Cash-on-Delivery checkout, courier tracking dispatch, and buyer refund/dispute workflows.",
        "action": "OPEN_MODAL",
        "target": "shelv"
    },
    {
        "id": "ibm",
        "source": "IBM_Databricks_Pipeline.py",
        "text": "IBM Stock Forecasting Platform is a time-series big data pipeline built on Azure Databricks and PySpark analyzing 16,272 historical Yahoo Finance rows. It compares ARIMA(1,1,0), ARIMA(0,1,1), and ARIMA(1,1,1) models with uncertainty intervals, achieving MAE of 4.4037, RMSE of 7.2174, and 83.33% Prediction Interval coverage.",
        "action": "OPEN_MODAL",
        "target": "ibm"
    },
    {
        "id": "litpredict",
        "source": "LitPredict_NLP_Pipeline.py",
        "text": "LitPredict is an NLP Book Rating Predictor that predicts average book ratings across 4,700+ book blurbs using VADER sentiment analysis, sparse TF-IDF embeddings, and a Random Forest regressor, achieving a held-out Mean Absolute Error (MAE) of 0.17 stars.",
        "action": "OPEN_MODAL",
        "target": "litpredict"
    },
    {
        "id": "powerbi",
        "source": "SL_Auto_Market_BI.pbix",
        "text": "Sri Lanka Automotive Market BI Dashboard is a 3-page interactive Power BI report using custom DAX and Power Query over ikman.lk and Riyasewana listings. Uncovers that Toyota and Suzuki hold over 50% market share, alongside depreciation curves and pricing anomalies.",
        "action": "OPEN_MODAL",
        "target": "powerbi"
    },
    {
        "id": "pocket",
        "source": "Pocket_Manager_Spec.md",
        "text": "Pocket Manager is a full-stack personal finance web app built with React.js, Node.js, Express, Tailwind CSS, and MySQL. Supports income/expense tracking, profile photo uploads, CRUD categorization, and one-click CSV export.",
        "action": "OPEN_MODAL",
        "target": "pocket"
    },
    {
        "id": "skills",
        "source": "Kathuskan_CV.pdf#skills",
        "text": "Kathuskan's technical stack includes Python, SQL, R, JavaScript, and Java. Machine Learning & NLP: Scikit-learn, Random Forest, XGBoost, TF-IDF, VADER, Statsmodels. Big Data & BI: Databricks, PySpark, Power BI, Pandas, NumPy. Web: FastAPI, React.js, Node.js, MongoDB, MySQL. Certified by AMD AI Academy in AI Agents & MCP.",
        "action": "SCROLL",
        "target": "skills"
    },
    {
        "id": "education_intern",
        "source": "Kathuskan_CV.pdf#about",
        "text": "Kathuskan Thavarajah is a BSc (Hons) Data Science undergraduate (2024–Present) at Sabaragamuwa University of Sri Lanka and an NVQ Level 4 Certified Graphic Designer. He is actively available for Data Science, AI/ML, and Full-Stack Software Engineering internships. Contact email: TKathuskan@gmail.com.",
        "action": "SCROLL",
        "target": "contact"
    },
    {
        "id": "cv_download",
        "source": "Kathuskan_CV.pdf",
        "text": "Download Kathuskan Thavarajah's resume CV PDF document containing his education, projects, certifications, and references.",
        "action": "DOWNLOAD_CV",
        "target": "Kathuskan_Thavarajah_CV.pdf"
    }
]

# Fit TF-IDF Vectorizer in memory
corpus = [item["text"] for item in KNOWLEDGE_CHUNKS]
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(corpus)


class QueryRequest(BaseModel):
    query: str
    include_audio: bool = True


async def synthesize_edge_tts(text: str) -> str:
    communicate = edge_tts.Communicate(text, voice="en-US-ChristopherNeural", rate="+5%")
    audio_stream = BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_stream.write(chunk["data"])
    return base64.b64encode(audio_stream.getvalue()).decode("utf-8")


@app.get("/")
def health_check():
    return {"status": "online", "agent": "VoxFolio MCP"}


@app.post("/api/ask")
async def ask_portfolio_agent(req: QueryRequest):
    start_time = time.perf_counter()

    query_vec = vectorizer.transform([req.query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    matched_chunk = KNOWLEDGE_CHUNKS[best_idx]

    groq_key = os.getenv("GROQ_API_KEY")
    spoken_reply = matched_chunk["text"]

    if groq_key and best_score > 0.05:
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are Kathuskan Thavarajah's AI Portfolio Voice Assistant. Answer in 2 concise, confident sentences using ONLY the provided context."
                            },
                            {
                                "role": "user",
                                "content": f"Context: {matched_chunk['text']}\n\nQuestion: {req.query}"
                            }
                        ],
                        "temperature": 0.2,
                        "max_tokens": 90
                    }
                )
                if resp.status_code == 200:
                    spoken_reply = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    if best_score < 0.06:
        spoken_reply = (
            "I can tell you about Kathuskan's projects like AutoValue LK, Shelv, LitPredict, "
            "or IBM Stock Forecasting, navigate his skills, or download his CV. What would you like to see?"
        )
        ui_action = {"action": "NONE", "target": ""}
        source_doc = "Intent_Router"
    else:
        ui_action = {"action": matched_chunk["action"], "target": matched_chunk["target"]}
        source_doc = matched_chunk["source"]

    audio_b64 = None
    if req.include_audio:
        try:
            audio_b64 = await synthesize_edge_tts(spoken_reply)
        except Exception:
            audio_b64 = None

    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

    return {
        "reply": spoken_reply,
        "ui_tool_call": ui_action,
        "telemetry": {
            "source": source_doc,
            "tfidf_score": round(best_score, 3),
            "latency_ms": latency_ms
        },
        "audio_base64": audio_b64
    }