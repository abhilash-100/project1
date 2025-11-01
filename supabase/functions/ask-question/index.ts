import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "npm:@supabase/supabase-js@2.57.4";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Client-Info, Apikey",
};

interface QuestionPayload {
  scientificName: string;
  question: string;
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, {
      status: 200,
      headers: corsHeaders,
    });
  }

  try {
    const supabaseClient = createClient(
      Deno.env.get("SUPABASE_URL") ?? "",
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
      {
        auth: {
          persistSession: false,
        },
      }
    );

    const authHeader = req.headers.get("Authorization");
    if (!authHeader) {
      throw new Error("Missing authorization header");
    }

    const token = authHeader.replace("Bearer ", "");
    const { data: { user }, error: authError } = await supabaseClient.auth.getUser(token);
    
    if (authError || !user) {
      throw new Error("Unauthorized");
    }

    const { scientificName, question }: QuestionPayload = await req.json();

    if (!scientificName || !question) {
      throw new Error("Missing required fields: scientificName and question");
    }

    const normalizedName = scientificName.toLowerCase().trim();

    const { data: flower, error: flowerError } = await supabaseClient
      .from("flowers")
      .select("*")
      .eq("scientific_name", normalizedName)
      .maybeSingle();

    if (flowerError) {
      throw new Error(`Database query error: ${flowerError.message}`);
    }

    if (!flower) {
      throw new Error(`Flower not found: ${scientificName}`);
    }

    const existingQA = flower.q_and_a || [];
    const existingAnswer = existingQA.find(
      (qa: any) => qa.question.toLowerCase().trim() === question.toLowerCase().trim()
    );

    if (existingAnswer) {
      return new Response(
        JSON.stringify({
          success: true,
          answer: existingAnswer.answer,
          cached: true,
        }),
        {
          headers: {
            ...corsHeaders,
            "Content-Type": "application/json",
          },
        }
      );
    }

    const geminiApiKey = Deno.env.get("GEMINI_API_KEY");
    if (!geminiApiKey) {
      throw new Error("GEMINI_API_KEY not configured");
    }

    const prompt = `You are a botanical expert. Answer the following question about the flower "${scientificName}":

Question: ${question}

Provide a clear, accurate, and concise answer. If you don't know the answer, say so.`;

    const geminiResponse = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${geminiApiKey}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          contents: [{
            parts: [{ text: prompt }]
          }],
          generationConfig: {
            temperature: 0.7,
            topK: 32,
            topP: 1,
            maxOutputTokens: 1024,
          },
        }),
      }
    );

    if (!geminiResponse.ok) {
      throw new Error(`Gemini API error: ${geminiResponse.statusText}`);
    }

    const geminiData = await geminiResponse.json();
    const answer = geminiData.candidates?.[0]?.content?.parts?.[0]?.text;
    
    if (!answer) {
      throw new Error("No response from Gemini API");
    }

    const updatedQA = [...existingQA, { question, answer }];

    const { error: updateError } = await supabaseClient
      .from("flowers")
      .update({ 
        q_and_a: updatedQA,
        updated_at: new Date().toISOString(),
      })
      .eq("scientific_name", normalizedName);

    if (updateError) {
      console.error("Failed to update Q&A:", updateError);
    }

    return new Response(
      JSON.stringify({
        success: true,
        answer,
        cached: false,
      }),
      {
        headers: {
          ...corsHeaders,
          "Content-Type": "application/json",
        },
      }
    );
  } catch (error) {
    console.error("Question answering error:", error);
    return new Response(
      JSON.stringify({
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      }),
      {
        status: 500,
        headers: {
          ...corsHeaders,
          "Content-Type": "application/json",
        },
      }
    );
  }
});