import { Attachment, ModelType } from "../types";

/**
 * MOCK TYPES
 * We define a structure that mimics the real Gemini response 
 * to satisfy TypeScript's strict null checks.
 */
interface MockResponse {
  candidates?: Array<{
    content?: {
      parts?: Array<{
        text?: string;
      }>;
    };
  }>;
}

// Gemini API is disconnected. Using 'any' as a bridge, 
// but we will handle the data safely below.
const ai: any = {}; 

/**
 * Streams chat response for text-based models.
 * Modified to ensure no "undefined" access errors occur during build.
 */
export const streamChatResponse = async function* (
  model: ModelType,
  history: Array<{ role: string; parts: any[] }>,
  newMessage: string,
  attachments: Attachment[],
  useGrounding: boolean
) {
  console.warn("Gemini API is disconnected. Returning mock chat response.");
  
  // Safe fallback to prevent TS18048 and TS2532
  const mockResponse: MockResponse = {
    candidates: [
      {
        content: {
          parts: [{ text: "Gemini API is currently disconnected. System is operating in Standalone Mode." }]
        }
      }
    ]
  };

  // This line mimics the access pattern that was failing:
  const text = mockResponse.candidates?.[0]?.content?.parts?.[0]?.text ?? "System Offline";
  
  yield { text };
  return;
};

/**
 * Generates an image using the specific image generation model.
 */
export const generateImageContent = async (
  prompt: string,
  aspectRatio: "1:1" | "16:9" | "9:16" = "1:1"
): Promise<{ imageUrl: string | null; text: string | null }> => {
  console.warn("Gemini API is disconnected. Returning mock image response.");
  
  return {
    imageUrl: "https://via.placeholder.com/256x256?text=Standalone+Mode",
    text: "Gemini API is currently disconnected. System is operating in Standalone Mode.",
  };
};
