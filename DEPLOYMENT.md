# 🚀 Deployment Guide — Vercel + Neon

## Overview

| Layer | Service | Why |
|-------|---------|-----|
| Frontend Dashboard | Vercel Static | CDN, zero-config, free |
| API (FastAPI) | Vercel Serverless Functions | Auto-scaling, preview URLs |
| Database | Neon PostgreSQL | Serverless Postgres, free tier, branching |
| Model Training | Local / GitHub Actions | Heavy compute, not serverless |

---

## Part 1 — Neon DB Setup

### Step 1: Create Neon Account
1. Go to [console.neon.tech](https://console.neon.tech)
2. Sign up with GitHub
3. Create new project: `sales-forecast`
4. Select region closest to your Vercel deployment region

### Step 2: Get Connection String
1. Dashboard → Connection Details
2. Copy the **Pooled connection** string (better for serverless)
3. Format: `postgresql://user:password@ep-xxx.region.aws.neon.tech/dbname?sslmode=require`

### Step 3: Initialize Schema
```bash
# In your local project
echo "DATABASE_URL=your-connection-string" >> .env
python utils/database.py
```

### Neon Free Tier Limits
- 10 GB storage
- 1 project
- Unlimited databases
- Auto-suspend after 5 min inactivity (cold start ~500ms)
- Branching for dev/staging/prod

---

## Part 2 — Vercel Deployment

### Step 1: Install Vercel CLI
```bash
npm install -g vercel
vercel login
```

### Step 2: First Deploy
```bash
cd sales-forecast
vercel
# Follow prompts:
# - Link to existing project? No
# - Project name: sales-forecast
# - Directory: ./
# - Override settings? No
```

### Step 3: Add Environment Variables
```bash
vercel env add DATABASE_URL
# Paste your Neon connection string

vercel env add APP_ENV
# Enter: production

vercel env add DEBUG
# Enter: false
```

Or via Vercel Dashboard:
`Settings → Environment Variables → Add`

### Step 4: Production Deploy
```bash
vercel --prod
# Your app: https://sales-forecast.vercel.app
```

### Step 5: Custom Domain (Optional)
`Settings → Domains → Add → your-domain.com`

---

## Part 3 — GitHub Actions CI/CD

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Vercel

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm i -g vercel
      - run: vercel --prod --token=${{ secrets.VERCEL_TOKEN }}
        env:
          VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
          VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}
```

Add secrets in GitHub: `Settings → Secrets → Actions`

---

## Part 4 — Architecture Notes

### ⚠️ Vercel Limitations for ML
- Max function duration: 60 seconds (set in vercel.json)
- Max memory: 1024 MB
- **Model training should NOT run on Vercel**

### ✅ Recommended Architecture
```
Local Machine / GitHub Actions
      ↓ Train models
      ↓ Save .h5 / .pkl to cloud storage (S3 / R2)
      
Vercel Function (api/server.py)
      ↓ Download pre-trained model on cold start
      ↓ Run inference (fast, <5s)
      ↓ Save results to Neon DB
      
Neon DB
      ↓ Store all forecast history
      ↓ Serve to dashboard via API
```

### For Heavy Training
Consider adding:
- **Railway** or **Render** for persistent Python server
- **GitHub Actions** for scheduled retraining
- **AWS Lambda** with larger memory for batch inference

---

## Part 5 — Monitoring

```bash
# Vercel logs
vercel logs sales-forecast --follow

# Neon queries
# Dashboard → Monitoring → Query Performance
```

---

## Quick Commands

```bash
# Local dev
make api                    # Start FastAPI
make train                  # Full training run

# Deploy
vercel                      # Preview deploy
vercel --prod               # Production deploy

# Database
python utils/database.py    # Init / verify schema
```
