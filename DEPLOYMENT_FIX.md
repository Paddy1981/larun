# LARUN API Deployment Fix Guide

## Problem Identified
- `api.larun.space` DNS does not exist
- Backend API never deployed to Railway (or any platform)
- Frontend hardcoded to expect `https://api.larun.space`

## Solution: Deploy + Configure DNS

### Step 1: Deploy Backend to Railway

```bash
cd ~/portfolio/larun

# Login to Railway (opens browser)
railway login

# Initialize project
railway init
# Select: Create new project
# Name: larun-api

# Deploy
railway up
```

Railway will give you a URL like:
`https://larun-api-production-xxxx.up.railway.app`

### Step 2: Configure DNS for api.larun.space

**In your domain registrar (where larun.space is registered):**

Add a CNAME record:
```
Type: CNAME
Name: api
Value: larun-api-production-xxxx.up.railway.app
TTL: 3600
```

**OR in Railway Dashboard:**
1. Go to your Railway project
2. Settings → Domains
3. Click "Add Domain"
4. Enter: `api.larun.space`
5. Railway will give you CNAME records to add

### Step 3: Verify Deployment

Wait 5-10 minutes for DNS propagation, then test:

```bash
# Check DNS
nslookup api.larun.space

# Test API
curl https://api.larun.space/health

# Expected response:
# {
#   "status": "healthy",
#   "timestamp": "...",
#   "skills_available": [...]
# }
```

### Step 4: Test Frontend Integration

Visit `https://larun.space` and try:
1. Open browser console (F12)
2. Navigate to app/chat/dashboard
3. Check for API calls - should now work

---

## Alternative: Quick Fix (Use Railway Default Domain)

If you want to skip DNS configuration for now:

### 1. Deploy to Railway
```bash
cd ~/portfolio/larun
railway login
railway init
railway up
```

### 2. Get Railway URL
```bash
railway domain
# Copy the URL, e.g.: https://larun-api-production-xxxx.up.railway.app
```

### 3. Update Frontend JavaScript

Edit these files in `~/portfolio/larun-space/`:

**File: `js/api.js`** (Line 17)
```javascript
// Change from:
baseURL: 'https://api.larun.space',

// To:
baseURL: 'https://larun-api-production-xxxx.up.railway.app',
```

**File: `js/app.js`** (Line 9)
```javascript
// Change from:
apiBaseURL: 'https://api.larun.space',

// To:
apiBaseURL: 'https://larun-api-production-xxxx.up.railway.app',
```

### 4. Commit and Push (if using GitHub Pages)
```bash
cd ~/portfolio/larun-space
git add .
git commit -m "Update API endpoint to Railway"
git push
```

---

## Check Models Availability

Before deploying, verify all models exist:

```bash
cd ~/portfolio/larun
find nodes -name "*.tflite" | wc -l
# Should show: 8 (or number of models you have)

# List all models
ls -lh nodes/*/model/*.tflite
```

---

## Environment Variables for Railway

Make sure these are set in Railway dashboard:

```env
SUPABASE_URL=https://mwmbcfcvnkwegrjlauis.supabase.co
SUPABASE_SERVICE_KEY=<your-service-key>
FRONTEND_URL=https://larun.space
```

---

## Troubleshooting

### "Module not found" errors
Install dependencies:
```bash
cd ~/portfolio/larun
pip install -r requirements.txt
```

### Railway build fails
Check `requirements.txt` and ensure Dockerfile is present

### API returns 500
Check Railway logs:
```bash
railway logs
```

---

## Next Steps After Deployment

1. ✅ Test all API endpoints
2. ✅ Verify model inference works
3. ✅ Test frontend integration
4. ✅ Monitor Railway logs for errors
5. ✅ Set up custom domain properly

---

**Estimated Time**: 15-20 minutes
**Cost**: Railway free tier ($5 credit/month) is enough for testing
