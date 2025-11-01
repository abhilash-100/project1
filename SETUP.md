# Flower Classification Application Setup

## Required Configuration

### Gemini API Key Setup

This application uses Google's Gemini API to fetch comprehensive flower information. You need to configure your Gemini API key as a secret in Supabase.

#### Steps:

1. Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

2. Add the secret to your Supabase project:
   - Go to your Supabase Dashboard
   - Navigate to Project Settings > Edge Functions > Secrets
   - Add a new secret with:
     - Name: `GEMINI_API_KEY`
     - Value: Your Gemini API key

3. The application will automatically use this key for:
   - Fetching detailed flower information on first classification
   - Answering user questions about flowers
   - Building the intelligent knowledge base

## How It Works

### Intelligent Classification Workflow

1. **User uploads a flower image**

2. **AI Classification**: The image is classified using the HSSAN model

3. **Smart Information Retrieval**:
   - First checks the Supabase `flowers` table
   - If found: Returns stored information instantly
   - If new: Calls Gemini API to gather comprehensive data

4. **Knowledge Base Growth**: New flower information is automatically stored for future use

### Question & Answer Feature

Users can ask specific questions about any classified flower:
- Questions are answered using Gemini API
- Answers are cached in the database
- Future identical questions return instantly from cache

## Database Schema

### Tables Created

- `uploads`: Tracks all user image uploads
- `predictions`: Stores classification results
- `flowers`: Central knowledge base with comprehensive flower information

### Storage

- `flower_images` bucket: Stores uploaded flower images

## Edge Functions

- `classify-flower`: Handles intelligent classification workflow
- `ask-question`: Processes user questions about flowers
