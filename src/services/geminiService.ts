
import { Attachment, ModelType } from "../types";

// Gemini API has been disconnected.
const ai: any = {}; // Mock object to prevent errors

/**
 * Streams chat response for text-based models.
 */
export const streamChatResponse = async function* (
  model: ModelType,
  history: Array<{ role: string; parts: any[] }>,
  newMessage: string,
  attachments: Attachment[],
  useGrounding: boolean
) {
  console.warn("Gemini API is disconnected. Returning mock chat response.");
  yield { text: "Gemini API is currently disconnected. System is operating in Standalone Mode." };
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
