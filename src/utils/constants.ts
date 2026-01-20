import { ModelType, AppConfig } from './types';

export const DEFAULT_CONFIG: AppConfig = {
  model: ModelType.FLASH,
  temperature: 0.7,
  useGrounding: false,
};

export const MODEL_LABELS: Record<ModelType, string> = {
  [ModelType.FLASH]: 'Gemini 3 Flash (Fast)',
  [ModelType.PRO]: 'Gemini 3 Pro (Reasoning)',
  [ModelType.IMAGE_GEN]: 'Gemini Flash Image (Visuals)',
};

export const WELCOME_MESSAGE = `
**Welcome to Nebula.**

I am powered by Google's latest Gemini models. 
- Use **Flash** for speed.
- Use **Pro** for complex reasoning.
- Switch to **Image Mode** to generate visuals.

How can I help you create today?
`;
