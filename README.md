# LLM Evaluation Dashboard — Starter Scaffold

## Setup (Week 1)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then fill in your real API keys
```

## Run it
Terminal 1 (backend):
```bash
uvicorn app.main:app --reload
```
Terminal 2 (dashboard):
```bash
streamlit run streamlit_app.py
```
Visit http://localhost:8501 for the dashboard, http://127.0.0.1:8000/docs for the raw API.

## What's already built
- `app/database.py` — SQLite by default, one-line swap to Postgres later
- `app/models.py` — Dataset, Prompt, Run, Response, Score tables
- `app/llm_clients.py` — OpenAI, Gemini, Mistral wrappers behind one `call_model()` function
- `app/evaluators.py` — LLM-as-a-Judge scoring for correctness/relevance/faithfulness/coherence
- `app/main.py` — REST endpoints: create dataset, run evaluation, fetch results
- `streamlit_app.py` — 3-tab dashboard: create dataset, run eval, view results/charts

## Next steps in your roadmap
- Week 3: add hallucination, context precision/recall metrics to `METRIC_DEFINITIONS`
- Week 4: swap SQLite -> Postgres in `.env`, add Users/Auth table + Logs table
- Week 5: split `main.py` into `app/routers/*.py` (auth, datasets, runs, results) as it grows
- Week 6-7: prompt/response side-by-side viewer, filters by model/tag, PDF/CSV export (pandas `.to_csv()` / a PDF lib)
- Week 7: add response caching (hash prompt+model -> skip re-calling if already run) to cut cost
- Week 8: write a few pytest tests for `evaluators.py` and `llm_clients.py`
- Week 9: Dockerfile + docker-compose (Postgres + backend + Streamlit), deploy to Render/Railway
- Week 10+: scheduled runs (cron / APScheduler), Slack/email alerts on regression
