import { GoogleGenAI, GenerateContentResponse, Chat, Part } from "@google/genai";
import { Attachment, ModelType } from "../types";

// Initialize the client
// The API key MUST be provided via process.env.API_KEY
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

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
  try {
    const chat: Chat = ai.chats.create({
      model: model,
      history: history,
      config: {
        tools: useGrounding ? [{ googleSearch: {} }] : undefined,
      },
    });

    const parts: Part[] = [];
    
    // Add attachments if any
    if (attachments.length > 0) {
      parts.push(...attachments.map(attachmentToPart));
    }
    
    // Add text prompt
    parts.push({ text: newMessage });

    // Send message stream
    // Note: The SDK chat.sendMessageStream takes a generic input which can be a string or parts.
    // However, the typed definition for sendMessageStream primarily highlights `message: string | Part[]`.
    const result = await chat.sendMessageStream({ 
      message: parts.length === 1 && parts[0].text ? parts[0].text : parts 
    });

    for await (const chunk of result) {
      const c = chunk as GenerateContentResponse;
      yield c;
    }
  } catch (error) {
    console.error("Gemini Chat Stream Error:", error);
    throw error;
  }
};

/**
 * Generates an image using the specific image generation model.
 * Note: We use generateContent with the image model for this purpose.
 */
export const generateImageContent = async (
  prompt: string,
  aspectRatio: "1:1" | "16:9" | "9:16" = "1:1"
): Promise<{ imageUrl: string | null; text: string | null }> => {
  try {
    const response: GenerateContentResponse = await ai.models.generateContent({
      model: ModelType.IMAGE_GEN,
      contents: {
        parts: [{ text: prompt }],
      },
      config: {
        imageConfig: {
          aspectRatio: aspectRatio,
        }
      },
    });

    let imageUrl: string | null = null;
    let text: string | null = null;

    if (response.candidates && response.candidates[0].content.parts) {
        for (const part of response.candidates[0].content.parts) {
            if (part.inlineData) {
                imageUrl = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
            } else if (part.text) {
                text = part.text;
            }
        }
    }

    return { imageUrl, text };

  } catch (error) {
    console.error("Gemini Image Generation Error:", error);
    throw error;
  }
};
