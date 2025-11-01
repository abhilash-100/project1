import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "npm:@supabase/supabase-js@2.57.4";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Client-Info, Apikey",
};

interface ClassificationPayload {
  uploadId: string;
  scientificName: string;
}

interface FlowerInfo {
  id: string;
  scientific_name: string;
  common_names: string[];
  description: string;
  botanical_properties: Record<string, any>;
  common_uses: string[];
  visual_states: Record<string, any>;
  care_instructions: string;
  toxicity_info: Record<string, any>;
  q_and_a: Array<{ question: string; answer: string }>;
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

    const { uploadId, scientificName }: ClassificationPayload = await req.json();

    if (!uploadId || !scientificName) {
      throw new Error("Missing required fields: uploadId and scientificName");
    }

    const normalizedName = scientificName.toLowerCase().trim();

    const { data: existingFlower, error: flowerQueryError } = await supabaseClient
      .from("flowers")
      .select("*")
      .eq("scientific_name", normalizedName)
      .maybeSingle();

    if (flowerQueryError) {
      throw new Error(`Database query error: ${flowerQueryError.message}`);
    }

    let flowerInfo: FlowerInfo;

    if (existingFlower) {
      flowerInfo = existingFlower;
      console.log(`Found existing flower: ${normalizedName}`);
    } else {
      console.log(`Fetching new flower info from Gemini: ${normalizedName}`);
      
      const geminiApiKey = Deno.env.get("GEMINI_API_KEY");
      if (!geminiApiKey) {
        throw new Error("GEMINI_API_KEY not configured");
      }

      const prompt = `You are a botanical expert. Provide comprehensive information about the flower "${scientificName}" in JSON format with the following structure:
{
  "common_names": ["array of common names"],
  "description": "detailed description of the flower",
  "botanical_properties": {
    "family": "plant family",
    "genus": "genus name",
    "native_region": "native region",
    "bloom_season": "blooming season",
    "growth_habit": "growth habit description"
  },
  "common_uses": ["ornamental", "medicinal", "culinary", etc],
  "visual_states": {
    "healthy": "description of healthy flower appearance",
    "wilted": "description when old/wilted",
    "damaged": "description when damaged",
    "diseased": "common diseases and their visual signs"
  },
  "care_instructions": "how to care for this flower",
  "toxicity_info": {
    "pets": "toxicity info for dogs, cats, etc",
    "humans": "toxicity info for humans"
  }
}

Provide accurate, detailed information. If unsure about any field, use an empty string or empty array.`;

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
              temperature: 0.4,
              topK: 32,
              topP: 1,
              maxOutputTokens: 2048,
            },
          }),
        }
      );

      if (!geminiResponse.ok) {
        throw new Error(`Gemini API error: ${geminiResponse.statusText}`);
      }

      const geminiData = await geminiResponse.json();
      const responseText = geminiData.candidates?.[0]?.content?.parts?.[0]?.text;
      
      if (!responseText) {
        throw new Error("No response from Gemini API");
      }

      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error("Could not extract JSON from Gemini response");
      }

      const parsedInfo = JSON.parse(jsonMatch[0]);

      const { data: newFlower, error: insertError } = await supabaseClient
        .from("flowers")
        .insert({
          scientific_name: normalizedName,
          common_names: parsedInfo.common_names || [],
          description: parsedInfo.description || "",
          botanical_properties: parsedInfo.botanical_properties || {},
          common_uses: parsedInfo.common_uses || [],
          visual_states: parsedInfo.visual_states || {},
          care_instructions: parsedInfo.care_instructions || "",
          toxicity_info: parsedInfo.toxicity_info || {},
          q_and_a: [],
          source: "gemini_api",
        })
        .select()
        .single();

      if (insertError) {
        throw new Error(`Failed to store flower info: ${insertError.message}`);
      }

      flowerInfo = newFlower;
    }

    const { error: updateError } = await supabaseClient
      .from("uploads")
      .update({ status: "completed" })
      .eq("id", uploadId);

    if (updateError) {
      console.error("Failed to update upload status:", updateError);
    }

    return new Response(
      JSON.stringify({
        success: true,
        flowerInfo,
      }),
      {
        headers: {
          ...corsHeaders,
          "Content-Type": "application/json",
        },
      }
    );
  } catch (error) {
    console.error("Classification error:", error);
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