# 🚀 LARUN API - Ready to Deploy!

## ✅ PRE-DEPLOYMENT CHECKLIST - ALL COMPLETE!

- ✅ **8 TinyML Models** copied and verified (164 KB total)
- ✅ **railway.json** created
- ✅ **Dockerfile** ready
- ✅ **cloud_endpoints.py** integrated
- ✅ **nodes/registry.yaml** updated
- ✅ **Environment variables** configured
- ✅ **API code** ready in api.py

---

## 🎯 DEPLOY NOW (3 Simple Commands)

```bash
cd ~/portfolio/larun

# 1. Login to Railway
railway login

# 2. Deploy (this will create a new project automatically)
railway up

# 3. Get your deployment URL
railway domain
```

**Expected Output:**
```
✓ Deployment successful
✓ URL: https://larun-api-production-xxxx.up.railway.app
```

---

## 📋 POST-DEPLOYMENT STEPS

### 1. Set Environment Variables in Railway Dashboard

Go to Railway dashboard → Your project → Variables

Add these:
```env
SUPABASE_URL=https://mwmbcfcvnkwegrjlauis.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im13bWJjZmN2bmt3ZWdyamxhdWlzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2OTg2MTk5MSwiZXhwIjoyMDg1NDM3OTkxfQ.AuKLxPahIKkyyzhIDVUevTw0svYo31uyEhbFOj8I7qE
FRONTEND_URL=https://larun.space
```

### 2. Configure Custom Domain (api.larun.space)

**Option A: In Railway Dashboard**
1. Go to Settings → Domains
2. Click "Add Custom Domain"
3. Enter: `api.larun.space`
4. Copy the CNAME record Railway provides

**Option B: In Your Domain Registrar**
Add CNAME record:
```
Type: CNAME
Name: api
Value: <your-railway-url>.up.railway.app
TTL: 3600
```

### 3. Test Deployment

```bash
# Get your Railway URL
RAILWAY_URL=$(railway domain)

# Test health endpoint
curl $RAILWAY_URL/health

# Test models endpoint
curl $RAILWAY_URL/api/models

# Expected: JSON response with 8 models listed
```

### 4. Update Frontend (Only if not using custom domain)

If you skip custom domain setup, update frontend files:

**File: `~/portfolio/larun-space/js/api.js`**
**File: `~/portfolio/larun-space/js/app.js`**

Replace `https://api.larun.space` with your Railway URL.

---

## 🔍 VERIFY DEPLOYMENT

All 8 models should be accessible:

```bash
curl https://api.larun.space/api/models | jq '.models[].name'
```

Expected output:
```
"Exoplanet Transit Detector"
"Variable Star Classifier"
"Stellar Flare Detector"
"Asteroseismology Analyzer"
"Supernova/Transient Detector"
"Galaxy Morphology Classifier"
"Spectral Type Classifier"
"Microlensing Event Detector"
```

---

## 📊 Monitor Deployment

```bash
# Watch logs in real-time
railway logs

# Check deployment status
railway status
```

---

## 🛠️ Troubleshooting

### Build Fails
```bash
# Check Railway logs
railway logs --build
```

Common issues:
- Missing dependencies → Check requirements.txt
- Port not set → Railway auto-sets $PORT variable

### Models Not Loading
```bash
# SSH into Railway container
railway run bash

# Check models exist
ls -lh nodes/*/model/*.tflite
```

### API Returns 500
- Check Railway logs for Python errors
- Verify environment variables are set
- Ensure Supabase is accessible

---

## 💰 Cost Estimate

**Railway Free Tier:**
- $5 credit/month
- ~500 hours runtime
- Perfect for testing and low traffic

**Railway Hobby ($5/month):**
- $5 usage included
- Good for production with moderate traffic (1000-5000 requests/day)

---

## ✨ You're Ready!

Everything is configured and ready to deploy. Just run:

```bash
cd ~/portfolio/larun
railway login
railway up
```

Then configure the custom domain `api.larun.space` and your LARUN platform will be fully operational! 🚀
