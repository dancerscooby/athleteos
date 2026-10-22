# Deploying AthleteOS on Render (free tier) + Supabase

## 1. Supabase (database)
1. Create a project and run `schema.sql` in the SQL editor.
2. Get the connection string from **Connect → Session pooler**. The host looks like
   `aws-0-<region>.pooler.supabase.com:5432`.
   - **Don't use "Direct connection"** (`db.<ref>.supabase.co`). That host is IPv6-only,
     and Render can't connect to IPv6, so every database call fails.
   - The transaction pooler (port 6543) also works. The backend turns off prepared
     statements so it's compatible with it.

## 2. Render (web service)
1. **New + → Blueprint**, then pick this repo. Render reads `render.yaml`.
2. Fill in the secrets it asks for:
   - `OPENROUTER_API_KEY`
   - `DATABASE_URL`: the session pooler string from step 1.
   - `GROQ_API_KEY`: only if you set `ATHLETEOS_PROVIDER=groq`.
3. Deploy, then open the service URL. The app is served at `/`, and `/health` is the
   health check.

If you set the service up by hand instead of using the Blueprint, use:

| Setting | Value |
|---|---|
| Build command | `pip install -r requirements.txt && python prefetch_models.py` |
| Start command | `uvicorn api:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` |
| Env | `PYTHON_VERSION=3.12.14`, `FASTEMBED_CACHE_PATH=/opt/render/project/src/.fastembed_cache`, `ATHLETEOS_PROVIDER=openrouter`, `ATHLETEOS_STORAGE_BACKEND=postgres` |

## Things to know
- **Keep `ATHLETEOS_STORAGE_BACKEND=postgres` in production.** With `json`, data goes to
  Render's temporary disk and is lost on every restart and redeploy.
- **Cold starts.** The free tier sleeps after 15 idle minutes, and the first request then
  takes about 30-60s. The RAG models are downloaded during the build, so a cold start
  doesn't download them again.
- **Free-model limits.**
  - OpenRouter's free tier allows about 50 requests per day without credits (1,000 per day
    after a one-time $10 credit purchase), shared between meal plans and chat.
  - A 7-day plan uses 2 requests, or up to 4 if both batches need the fallback model.
  - Groq's free tier is limited by tokens per minute, so a 7-day plan there is paced by
    waits and takes about 2-4 minutes.
  - If plans still fail often, a cheap paid model is the most reliable fix.
