# Kathuskan Thavarajah — AI/ML & Data Science Portfolio (+ VoxFolio Voice Agent)

Live interactive portfolio showcasing end-to-end Machine Learning applications, NLP pipelines, PySpark/Databricks forecasting, Power BI analytics, and Full-Stack MERN apps.

## 🎙️ Featured Architecture: VoxFolio MCP (Voice & RAG Assistant)
Built directly into this portfolio is an agentic voice assistant that allows recruiters to query my projects and navigate the UI hands-free:
- **Retrieval (RAG):** In-memory `Scikit-learn` TF-IDF (`ngram_range=(1,2)`) + Cosine Similarity over project documentation and CV metrics (`<15ms` retrieval latency).
- **LLM Synthesis & Guardrails:** Grounded context generation via `FastAPI` + `Llama-3.1-8B-Instant` with an automatic deterministic offline fallback.
- **Voice Pipeline:** Browser `Web Speech API` (`SpeechRecognition`) for STT + Microsoft Neural `edge-tts` (`en-US-ChristopherNeural`) streamed as Base64 MP3 audio.
- **DOM Tool-Calling:** Executes real-time UI actions (`OPEN_MODAL`, `SCROLL`, `DOWNLOAD_CV`) alongside live telemetry badges.

## 🚀 Projects Highlighted
1. **AutoValue LK** — Random Forest Vehicle Valuation Engine (`FastAPI`, `Scikit-learn`, `React`, $R^2 = 0.739$ across 11,750 listings)
2. **Shelv** — Full-Stack Book Marketplace (`MongoDB`, `Express`, `React`, `Node.js`, `Stripe`, 3-Role RBAC)
3. **IBM Stock Forecasting** — Time-Series Pipeline (`Azure Databricks`, `PySpark`, `ARIMA`, 16,272 records)
4. **LitPredict** — NLP Book Rating Predictor (`TF-IDF`, `VADER`, `Random Forest`, 0.17 Star MAE)
5. **SL Automotive Market BI** — 3-Page Interactive Report (`Power BI`, `DAX`, `Power Query`)
6. **Pocket Manager** — Full-Stack Personal Finance Tracker (`React`, `Node.js`, `MySQL`)