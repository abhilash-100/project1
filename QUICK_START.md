# Quick Start Guide

## 🚀 Get Running in 5 Minutes

### Step 1: Install Dependencies (1 min)

```bash
# Install frontend dependencies
npm install

# Install backend dependencies
cd backend
pip install -r requirements.txt
cd ..
```

### Step 2: Configure Gemini API Key (2 min)

1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Go to [Supabase Dashboard](https://supabase.com/dashboard)
3. Open your project
4. Navigate to: **Project Settings** > **Edge Functions** > **Secrets**
5. Click **Add New Secret**:
   - Name: `GEMINI_API_KEY`
   - Value: Paste your API key
6. Click **Save**

### Step 3: Start Development Servers (2 min)

**Terminal 1 - Frontend**:
```bash
npm run dev
```

**Terminal 2 - Backend**:
```bash
cd backend
python main.py
```

### Step 4: Open Application

1. Open browser to: http://localhost:5173
2. Sign up with email and password
3. Start classifying flowers!

## 📋 Quick Checklist

Before using the app, verify:

- ✅ Frontend running on http://localhost:5173
- ✅ Backend running on http://localhost:8000
- ✅ `.env` file has Supabase credentials
- ✅ Gemini API key added to Supabase Secrets
- ✅ Browser shows login/signup form

## 🎯 Quick Test

1. **Sign Up**: Create an account
2. **Upload Image**: Choose a flower photo
3. **Wait 5-10 seconds**: First classification calls Gemini API
4. **View Results**: See comprehensive flower information
5. **Ask Question**: Type "Is this flower toxic to pets?"
6. **Check History**: Switch to History tab

## 📖 Full Documentation

- **README.md** - Complete architecture and file descriptions
- **SETUP.md** - Detailed configuration guide
- **TROUBLESHOOTING.md** - Fix blank preview and other issues
- **IMPLEMENTATION_SUMMARY.md** - What was built and how it works

## ⚡ Performance Tips

**First Classification**: 5-10 seconds (calls Gemini API)
**Second Classification (same flower)**: <1 second (cached in database)
**Common Questions**: Instant (cached answers)

The system gets faster as more users classify flowers!

## 🔑 Important Notes

1. **GEMINI_API_KEY is required** for flower information
2. **Python backend must be running** for AI classification
3. **Authentication is required** to use the app
4. **Database is pre-configured** - migrations already applied
5. **Edge Functions are deployed** - no additional setup needed

## 🆘 Having Issues?

### Blank Page?
1. Check browser console (F12) for errors
2. Verify `.env` has correct Supabase credentials
3. See **TROUBLESHOOTING.md** for detailed help

### Classification Failing?
1. Ensure Python backend is running
2. Check terminal for backend errors
3. Verify model file exists (may need training)

### No Flower Information?
1. Add `GEMINI_API_KEY` to Supabase Secrets
2. Classification works without it, but info won't be detailed

## 📞 System Requirements

- **Node.js**: 18+
- **Python**: 3.8+
- **Browser**: Chrome 90+, Firefox 88+, Safari 14+
- **Internet**: Required for Gemini API calls

## 🎓 What's Next?

After successful setup:
1. Read **README.md** to understand the architecture
2. Explore the intelligent caching system
3. Try asking questions about flowers
4. Check how history updates in real-time
5. Review the HSSAN attention mechanism in `backend/model/hssan.py`

## 💡 Pro Tips

- **First flower classification is slow** - it's fetching comprehensive info
- **Subsequent classifications are instant** - cached in database
- **Popular questions are cached** - system learns from all users
- **History updates automatically** - no refresh needed
- **Ask any question** - Gemini API will find the answer

---

Ready to classify flowers? Let's go! 🌸
