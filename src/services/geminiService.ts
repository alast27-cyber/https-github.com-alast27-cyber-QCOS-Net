import { GenerateContentResponse, Chat, Part } from "@google/genai";
// import { GoogleGenAI } from "@google/genai";
import { Attachment, ModelType } from "../types";
import { generateContentWithRetry, isRetriableError } from "../utils/gemini";

// Gemini API has been disconnected.
// const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
const ai: any = {}; // Mock object to prevent errors

/**
 * Converts a file Attachment to the format expected by the SDK (Part).
 */
const attachmentToPart = (attachment: Attachment): Part => {
  // Remove the data URL prefix (e.g., "data:image/png;base64,") to get raw base64
  const base64Data = attachment.base64.split(',')[1];
  return {
    inlineData: {
      mimeType: attachment.mimeType,
      data: base64Data,
    },
  };
};

/**
 * Streams chat response for text-based models.
 */
export const streamChatResponse = async function* (
  model: ModelType,
  history: Array<{ role: string; parts: Part[] }>,
  newMessage: string,
  attachments: Attachment[],
  useGrounding: boolean
) {
  console.warn("Gemini API is disconnected. Returning mock chat response.");
  yield { text: "Gemini API is currently disconnected. Please activate Google AI Studio free tier or connect a paid API key to enable AI functionalities." };
  return;
};

/**
 * Generates an image using the specific image generation model.
 * Note: We use generateContent with the image model for this purpose.
 */
export const generateImageContent = async (
  prompt: string,
  aspectRatio: "1:1" | "16:9" | "9:16" = "1:1"
): Promise<{ imageUrl: string | null; text: string | null }> => {
  console.warn("Gemini API is disconnected. Returning mock image response.");
  return {
    imageUrl: "https://via.placeholder.com/256x256?text=Gemini+API+Disconnected",
    text: "Gemini API is currently disconnected. Please activate Google AI Studio free tier or connect a paid API key to enable AI functionalities.",
  };
};
