import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import { Attachment, ModelType } from "../../types";

// Initialize the Gemini API client
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });

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
  if (!process.env.GEMINI_API_KEY) {
    yield { text: "Error: Gemini API key is missing." };
    return;
  }

  try {
    const chat = ai.chats.create({
      model: model,
    });

    // Add history to chat if needed, though Gemini SDK handles history differently
    // This is a simplified implementation.
    
    const responseStream = await chat.sendMessageStream({ message: newMessage });
    
    for await (const chunk of responseStream) {
      const c = chunk as GenerateContentResponse;
      yield { text: c.text || "" };
    }
  } catch (error) {
    console.error("Error streaming chat response:", error);
    yield { text: "Error: Failed to stream chat response." };
  }
};

/**
 * Generates an image using the specific image generation model.
 */
export const generateImageContent = async (
  prompt: string,
  aspectRatio: "1:1" | "16:9" | "9:16" = "1:1"
): Promise<{ imageUrl: string | null; text: string | null }> => {
  if (!process.env.GEMINI_API_KEY) {
    return { imageUrl: null, text: "Error: Gemini API key is missing." };
  }

  try {
    const response = await ai.models.generateContent({
      model: ModelType.IMAGE_GEN,
      contents: { parts: [{ text: prompt }] },
      config: {
        imageConfig: { aspectRatio },
      },
    });

    let imageUrl: string | null = null;
    const text: string | null = response.text || null;

    for (const part of response.candidates[0].content.parts) {
      if (part.inlineData) {
        imageUrl = `data:image/png;base64,${part.inlineData.data}`;
      }
    }

    return { imageUrl, text };
  } catch (error) {
    console.error("Error generating image:", error);
    return { imageUrl: null, text: "Error: Failed to generate image." };
  }
};
