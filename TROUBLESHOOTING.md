# Troubleshooting Guide

## Blank Preview / White Screen Issues

If you're experiencing a blank preview or white screen when the application loads, follow these steps:

### 1. Check Environment Variables

The most common cause is missing or incorrect Supabase environment variables.

**Verify your `.env` file contains**:
```env
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your_anon_key_here
VITE_API_URL=http://localhost:8000
```

**Important**:
- Environment variables must start with `VITE_` to be accessible in the frontend
- After changing `.env`, restart the dev server
- The Supabase URL should be the full URL including `https://`

### 2. Check Browser Console

Open browser developer tools (F12) and check the Console tab for errors:

**Common errors and solutions**:

- **"Failed to fetch" or CORS errors**:
  - Make sure the Python backend is running on port 8000
  - Verify `VITE_API_URL` in `.env` matches your backend URL

- **"Invalid API key" or Supabase auth errors**:
  - Verify your `VITE_SUPABASE_ANON_KEY` is correct
  - Go to Supabase Dashboard > Settings > API to find your keys

- **"auth.users" relation does not exist**:
  - Database migrations may not have run
  - Check the Supabase Dashboard > Database > Tables
  - You should see: `uploads`, `predictions`, `flowers` tables

### 3. Authentication State

The application requires authentication. A blank screen might mean:

- You're not logged in yet (should show login form instead)
- Auth state is stuck in "loading"

**To fix**:
1. Clear browser cache and cookies
2. Open browser console and run: `localStorage.clear()`
3. Reload the page
4. You should see the login/signup form

### 4. Database Tables Missing

If tables are missing, reapply migrations:

The following tables should exist:
- `uploads` - Stores image upload records
- `predictions` - Stores classification results
- `flowers` - Stores flower knowledge base

Check in Supabase Dashboard > Database > Tables

### 5. Storage Bucket Missing

The `flower_images` storage bucket must exist.

Check in Supabase Dashboard > Storage:
- Bucket name: `flower_images`
- Should be private (not public)
- File size limit: 10MB

### 6. Edge Functions Not Deployed

The application requires two Edge Functions:
- `classify-flower`
- `ask-question`

Check in Supabase Dashboard > Edge Functions

If missing, they were already deployed during setup.

### 7. Python Backend Not Running

The AI classification requires the Python backend:

```bash
cd backend
python main.py
```

Should output:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**To test**:
```bash
curl http://localhost:8000/health
```

Should return: `{"status":"healthy","model_loaded":true}`

### 8. Gemini API Key Missing

For full functionality, add your Gemini API key:

1. Get key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add to Supabase Dashboard > Project Settings > Edge Functions > Secrets
3. Secret name: `GEMINI_API_KEY`
4. Secret value: Your API key

**Note**: Classification will work without this, but flower information won't be fetched.

### 9. Network Issues

Check your network connection:
- Can you access Supabase Dashboard?
- Can you ping your Supabase project URL?
- Are you behind a corporate firewall or VPN?

### 10. Browser Compatibility

The application requires a modern browser:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Step-by-Step Debug Process

1. **Open browser console** (F12)
2. **Reload the page** (Ctrl/Cmd + R)
3. **Check for red errors** in the console
4. **Look at Network tab** - are requests failing?
5. **Check Application tab** > Local Storage - is there auth data?

## Common Error Messages

### "Failed to fetch"
**Cause**: Python backend not running or wrong URL
**Solution**: Start backend with `python backend/main.py`

### "Invalid JWT" or "Unauthorized"
**Cause**: Authentication token expired or invalid
**Solution**: Sign out and sign back in

### "Bucket not found"
**Cause**: Storage bucket not created
**Solution**: Check Supabase Dashboard > Storage

### "Table does not exist"
**Cause**: Migrations not applied
**Solution**: Migrations were applied during setup, check Supabase Dashboard

### "GEMINI_API_KEY not configured"
**Cause**: Gemini API key not added to Supabase secrets
**Solution**: Add secret in Supabase Dashboard (see step 8 above)

## Still Having Issues?

1. **Check the README.md** for complete setup instructions
2. **Verify all prerequisites** are installed (Node.js 18+, Python 3.8+)
3. **Review the SETUP.md** file for configuration details
4. **Check Supabase Dashboard** for service status

## Quick Reset

If nothing works, try a complete reset:

```bash
# Clear browser data
# In browser console:
localStorage.clear()
sessionStorage.clear()

# Restart dev server
npm run dev

# Restart backend
cd backend
python main.py
```

## Verification Checklist

✅ `.env` file exists with correct variables
✅ `npm install` completed successfully
✅ `npm run dev` is running
✅ Python backend is running on port 8000
✅ Supabase tables exist (uploads, predictions, flowers)
✅ Supabase storage bucket exists (flower_images)
✅ Edge Functions are deployed (classify-flower, ask-question)
✅ Browser console shows no errors
✅ Can access login/signup form
✅ After login, can see upload interface

If all items are checked, the application should work correctly!
