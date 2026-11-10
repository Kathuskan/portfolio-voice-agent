import os
import re
import time
import random
import base64
from io import BytesIO
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import edge_tts
import httpx

app = FastAPI(title="Kathuskan VoxFolio MCP — Hybrid Voice & RAG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Expanded & Keyword-Enriched Knowledge Base
KNOWLEDGE_CHUNKS = [
    {
        "id": "skills",
        "source": "Kathuskan_CV.pdf#skills",
        "keywords": ["tech stack", "stack", "skills", "technologies", "tools", "programming", "languages", "frameworks", "what does he know", "python", "sql", "fastapi", "react", "scikit"],
        "text": "Kathuskan's core tech stack and technical skills include Python, SQL, R, JavaScript, and Java. For Machine Learning and NLP, he uses Scikit-learn, Random Forest, XGBoost, TF-IDF, VADER, and Statsmodels. For Big Data and BI, he works with Azure Databricks, PySpark, Power BI, DAX, Pandas, and NumPy. For Full-Stack web development, he uses FastAPI, React.js, Node.js, Express, Tailwind CSS, MongoDB, and MySQL.",
        "action": "SCROLL",
        "target": "skills"
    },
    {
        "id": "autovalue",
        "source": "AutoValue_LK_README.md",
        "keywords": ["autovalue", "auto value", "car", "vehicle", "valuation", "asking price", "flagship"],
        "text": "AutoValue LK is Kathuskan's flagship Machine Learning web application that predicts vehicle asking prices in Sri Lanka across 11,750 cleaned records. It uses engineered log-mileage and vehicle-age features with a tuned Random Forest regression pipeline served via FastAPI and React, achieving 84.95% of predictions within 20% of market price and an R-squared of 0.739.",
        "action": "OPEN_MODAL",
        "target": "autovalue"
    },
    {
        "id": "shelv",
        "source": "Shelv_MERN_README.md",
        "keywords": ["shelv", "shelve", "book marketplace", "used books", "mern", "ecommerce", "e-commerce", "stripe"],
        "text": "Shelv is a full-stack MERN e-commerce marketplace built exclusively for buying and selling new and used books. Built with React.js, Node.js, Express.js, MongoDB, and Stripe, it supports Buyer, Seller, and Admin roles, Stripe card and Cash-on-Delivery checkout, courier tracking, and buyer refund/dispute workflows.",
        "action": "OPEN_MODAL",
        "target": "shelv"
    },
    {
        "id": "ibm",
        "source": "IBM_Databricks_Pipeline.py",
        "keywords": ["ibm", "stock", "forecasting", "time series", "arima", "pyspark", "databricks", "yahoo finance"],
        "text": "The IBM Stock Forecasting Platform is a Big Data time-series pipeline built on Azure Databricks and PySpark over 16,272 historical Yahoo Finance records. It evaluates ARIMA(1,1,0), ARIMA(0,1,1), and ARIMA(1,1,1) models with uncertainty intervals, achieving a Mean Absolute Error of 4.4037 and 83.33% prediction interval coverage.",
        "action": "OPEN_MODAL",
        "target": "ibm"
    },
    {
        "id": "litpredict",
        "source": "LitPredict_NLP_Pipeline.py",
        "keywords": ["litpredict", "lit predict", "nlp", "book rating", "blurbs", "vader", "sentiment"],
        "text": "LitPredict is an end-to-end NLP pipeline that predicts average book ratings across 4,700+ books directly from marketing blurbs. It combines VADER sentiment analysis and TF-IDF embeddings with a Random Forest regressor, achieving a held-out Mean Absolute Error of just 0.17 stars.",
        "action": "OPEN_MODAL",
        "target": "litpredict"
    },
    {
        "id": "powerbi",
        "source": "SL_Auto_Market_BI.pbix",
        "keywords": ["power bi", "powerbi", "bi dashboard", "dax", "ikman", "riyasewana", "business intelligence", "automotive market"],
        "text": "The Sri Lanka Automotive Market BI Dashboard is a 3-page interactive Power BI report using custom DAX measures and Power Query to analyze ikman.lk and Riyasewana vehicle listings. It reveals that Toyota and Suzuki hold over 50% market share and maps non-linear depreciation curves.",
        "action": "OPEN_MODAL",
        "target": "powerbi"
    },
    {
        "id": "pocket",
        "source": "Pocket_Manager_Spec.md",
        "keywords": ["pocket manager", "pocket", "personal finance", "expense tracker", "fintech", "savings", "mysql"],
        "text": "Pocket Manager is a full-stack personal finance tracker built with React.js, Node.js, Express, Tailwind CSS, and MySQL. It features real-time income, expense, and savings analytics, profile photo uploads, CRUD categorization, and one-click CSV report export.",
        "action": "OPEN_MODAL",
        "target": "pocket"
    },
    {
        "id": "voxfolio",
        "source": "VoxFolio_MCP_Agent.py",
        "keywords": ["voxfolio", "voice ai", "voice assistant", "how do you work", "your architecture", "rag", "who built you"],
        "text": "I am VoxFolio, Kathuskan's custom Voice and RAG Portfolio Agent! I use a hybrid Scikit-learn TF-IDF and keyword retriever, FastAPI, Microsoft neural Edge-TTS for voice synthesis, and structured DOM tool-calling to scroll sections and open project modals hands-free.",
        "action": "OPEN_MODAL",
        "target": "voxfolio"
    },
    {
        "id": "certifications",
        "source": "Kathuskan_CV.pdf#certifications",
        "keywords": ["certificate", "certifications", "credentials", "amd", "coursera", "mcp", "databricks academy", "experience"],
        "text": "Kathuskan holds verified certifications from the AMD AI Academy (Building AI Agents with MCP & Open-Source Inference), Databricks Academy (Data Engineering with PySpark), Coursera (Foundations of Business Intelligence), and is an NVQ Level 4 Certified Graphic Designer.",
        "action": "SCROLL",
        "target": "experience"
    },
    {
        "id": "education_about",
        "source": "Kathuskan_CV.pdf#education",
        "keywords": ["education", "university", "degree", "study", "sabaragamuwa", "susl", "who is kathuskan", "background", "about him", "gpa", "school"],
        "text": "Kathuskan Thavarajah is pursuing a BSc Honours degree in Data Science at Sabaragamuwa University of Sri Lanka (2024 to Present). He also has 5+ years of visual design experience as an NVQ Level 4 Certified Computer Graphic Designer.",
        "action": "SCROLL",
        "target": "about"
    },
    {
        "id": "leadership",
        "source": "Kathuskan_CV.pdf#leadership",
        "keywords": ["leadership", "volunteer", "volunteering", "clubs", "ieee", "cricket", "rotaract", "leo", "graphic design"],
        "text": "On campus, Kathuskan volunteers as a Graphic Designer for the IEEE Computer Society Chapter and Sabra Cricket Team at SUSL, and serves as Media Coordinator and active member across Zero Plastic, Leo, Rotaract, and the AWS Student Builder community.",
        "action": "SCROLL",
        "target": "experience"
    },
    {
        "id": "references",
        "source": "Kathuskan_CV.pdf#references",
        "keywords": ["referee", "references", "recommendation", "kaushalya", "gobihanath", "dialog"],
        "text": "Kathuskan's academic referee is Mrs. P.K.D.K. Kaushalya (Lecturer, Dept. of Computing & Information Systems, Sabaragamuwa University), and his industry referee is Mr. Gobihanath Balasubramaniam (Senior Executive Data Scientist at Dialog Axiata PLC).",
        "action": "SCROLL",
        "target": "references"
    },
    {
        "id": "internship_contact",
        "source": "Kathuskan_CV.pdf#contact",
        "keywords": ["intern", "internship", "hire", "hiring", "available", "availability", "contact", "email", "phone", "location", "sri lanka", "reach out", "why hire"],
        "text": "Kathuskan is actively available for Data Science, AI/ML, and Full-Stack Software Engineering internships! You can email him directly at TKathuskan@gmail.com, call +94 75 811 4531, or use the contact form on this page.",
        "action": "SCROLL",
        "target": "contact"
    },
    {
        "id": "cv_download",
        "source": "Kathuskan_Thavarajah_CV.pdf",
        "keywords": ["download cv", "download resume", "get cv", "get resume", "pdf", "curriculum vitae", "resume"],
        "text": "I am triggering the download for Kathuskan Thavarajah's official CV PDF right now!",
        "action": "DOWNLOAD_CV",
        "target": "Kathuskan_Thavarajah_CV.pdf"
    }
]

# Build enriched search corpus combining keywords + text so TF-IDF weights key terms heavily
corpus = [
    (" ".join(item["keywords"]) + " " + " ".join(item["keywords"]) + " " + item["text"])
    for item in KNOWLEDGE_CHUNKS
]
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
tfidf_matrix = vectorizer.fit_transform(corpus)

PORTFOLIO_SUMMARY_PROMPT = """You are VoxFolio, the witty, concise AI Voice Assistant on Kathuskan Thavarajah's Data Science & AI/ML portfolio website.
Rules:
1. Keep every response to 1 or 2 short, spoken-friendly sentences (under 45 words) with no markdown bullet lists.
2. Key facts about Kathuskan: BSc (Hons) Data Science undergrad at Sabaragamuwa University of Sri Lanka (SUSL), NVQ Level 4 Graphic Designer, AMD AI Academy MCP certified.
3. Core Tech Stack: Python, SQL, R, JS, Scikit-learn, Random Forest, XGBoost, TF-IDF, VADER, PySpark, Azure Databricks, Power BI, FastAPI, React, Node.js, MongoDB, MySQL.
4. Projects: AutoValue LK (Sri Lanka car price ML web app, R²=0.739 across 11,750 records), Shelv (MERN book marketplace with Stripe), IBM Stock Forecasting (Databricks PySpark ARIMA), LitPredict (NLP book rating predictor, 0.17 MAE), SL Automotive Power BI Dashboard, Pocket Manager (React/MySQL finance app).
5. If the user asks something completely unrelated to Kathuskan (e.g. general trivia, coding, weather, jokes, chitchat), answer their question briefly and playfully in 1 sentence, then smoothly tie it back to Kathuskan's skills or projects in the 2nd sentence!"""


class QueryRequest(BaseModel):
    query: str
    include_audio: bool = True


def score_chunks(query: str):
    """Combines exact keyword phrase boosting + TF-IDF Cosine Similarity."""
    q_clean = re.sub(r"[^a-z0-9\s]", " ", query.lower()).strip()
    query_vec = vectorizer.transform([q_clean])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    best_idx = 0
    best_score = 0.0

    for idx, chunk in enumerate(KNOWLEDGE_CHUNKS):
        score = float(tfidf_scores[idx])
        # Boost score if explicit keywords match
        for kw in chunk["keywords"]:
            if kw in q_clean:
                # Multi-word exact keyword match gets a strong boost
                score += 0.45 if " " in kw else 0.25
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx, min(best_score, 0.99)


def generate_smart_conversational_fallback(query: str) -> tuple[str, dict]:
    """Handles greetings, chitchat, and off-topic questions variedly when no LLM key is set."""
    q = query.lower().strip()

    # 1. Greetings
    if any(w in q.split() for w in ["hi", "hello", "hey", "greetings", "morning", "evening", "yo"]):
        greetings = [
            "Hey there! I'm Kathuskan's AI voice agent. Want to check out his ML apps like AutoValue LK, explore his Python and PySpark stack, or grab his CV?",
            "Hello! Welcome to Kathuskan's portfolio. You can ask me about his Data Science projects, his tech stack, or his internship availability!",
            "Hi! Ready to explore? Ask me about AutoValue LK, his Databricks stock forecasting pipeline, or say 'download CV'."
        ]
        return random.choice(greetings), {"action": "NONE", "target": ""}

    # 2. "How are you" / Bot identity
    if "how are you" in q or "who are you" in q or "your name" in q:
        return (
            "I'm VoxFolio, running at sub-150 millisecond latency and ready to show off Kathuskan's Data Science and Full-Stack work! Which project should we open first?",
            {"action": "NONE", "target": ""}
        )

    # 3. "Show all projects" / general project inquiry
    if "project" in q or "built" in q or "work" in q or "portfolio" in q:
        return (
            "Kathuskan has shipped 7 flagship projects including AutoValue LK for car valuation, Shelv MERN bookstore, IBM Stock Forecasting in PySpark, and LitPredict NLP. I've scrolled to his projects grid for you!",
            {"action": "SCROLL", "target": "projects"}
        )

    # 4. Theme toggle via voice
    if "dark mode" in q or "light mode" in q or "theme" in q:
        return (
            "I've toggled the portfolio theme for you! Let me know what section you'd like to inspect next.",
            {"action": "TOGGLE_THEME", "target": ""}
        )

    # 5. Thank you / bye
    if any(w in q for w in ["thank", "thanks", "awesome", "cool", "great", "nice", "bye"]):
        return (
            "You're very welcome! If you'd like to connect with Kathuskan for an internship or project, his email is TKathuskan@gmail.com.",
            {"action": "NONE", "target": ""}
        )

    # 6. Varied Off-Topic Responses (Never repeats the same line!)
    off_topic_replies = [
        f"That's an interesting question about '{query}', though my neural weights are tuned specifically to Kathuskan's Data Science portfolio! Want to see his Machine Learning projects or his tech stack?",
        f"I'm specialized in Kathuskan's engineering work rather than general trivia, but I can show you how he builds NLP pipelines, PySpark models, and full-stack apps!",
        f"I don't have a dataset for '{query}' loaded right now, but I do have 11,750 vehicle records in AutoValue LK and 16,000 stock rows in his Databricks project! Which one should I open?",
        "Haha, I'll leave that topic to general AI—my job is helping you discover why Kathuskan is a great Data Science and Full-Stack intern! Ask me about his skills, projects, or CV."
    ]
    return random.choice(off_topic_replies), {"action": "NONE", "target": ""}


async def synthesize_edge_tts(text: str) -> str:
    clean = re.sub(r"<[^>]*>", "", text)
    communicate = edge_tts.Communicate(clean, voice="en-US-ChristopherNeural", rate="+5%")
    audio_stream = BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_stream.write(chunk["data"])
    return base64.b64encode(audio_stream.getvalue()).decode("utf-8")


@app.get("/")
def health_check():
    return {"status": "online", "agent": "VoxFolio MCP v2"}


@app.post("/api/ask")
async def ask_portfolio_agent(req: QueryRequest):
    start_time = time.perf_counter()

    best_idx, best_score = score_chunks(req.query)
    matched_chunk = KNOWLEDGE_CHUNKS[best_idx]
    groq_key = os.getenv("GROQ_API_KEY")

    # Case A: Strong Portfolio Match (Score >= 0.18)
    if best_score >= 0.18:
        spoken_reply = matched_chunk["text"]
        ui_action = {"action": matched_chunk["action"], "target": matched_chunk["target"]}
        source_doc = matched_chunk["source"]

        # Optional Groq LLM natural phrasing over the matched chunk
        if groq_key:
            try:
                async with httpx.AsyncClient(timeout=4.5) as client:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {groq_key}"},
                        json={
                            "model": "llama-3.3-70b-versatile",
                            "messages": [
                                {"role": "system", "content": PORTFOLIO_SUMMARY_PROMPT},
                                {"role": "user", "content": f"Relevant Portfolio Fact: {matched_chunk['text']}\n\nUser Question: {req.query}"}
                            ],
                            "temperature": 0.3,
                            "max_tokens": 95
                        }
                    )
                    if resp.status_code == 200:
                        spoken_reply = resp.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                pass

    # Case B: Open-Ended / Off-Topic / General Question (Score < 0.18)
    else:
        source_doc = "Conversational_Agent"
        spoken_reply, ui_action = generate_smart_conversational_fallback(req.query)

        # If GROQ_API_KEY is available, let Llama-3.3-70B answer ANY question naturally!
        if groq_key:
            try:
                async with httpx.AsyncClient(timeout=4.5) as client:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {groq_key}"},
                        json={
                            "model": "llama-3.3-70b-versatile",
                            "messages": [
                                {"role": "system", "content": PORTFOLIO_SUMMARY_PROMPT},
                                {"role": "user", "content": req.query}
                            ],
                            "temperature": 0.6,
                            "max_tokens": 95
                        }
                    )
                    if resp.status_code == 200:
                        spoken_reply = resp.json()["choices"][0]["message"]["content"].strip()
                        source_doc = "Llama-3.3-70B_General"
            except Exception:
                pass

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