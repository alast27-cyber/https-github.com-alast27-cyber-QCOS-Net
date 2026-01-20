
import React from 'react';

// --- QCOS Types ---

export interface ChartDataPoint {
  name: string;
  [key: string]: string | number;
}

export interface LogEntry {
  id: number;
  level: 'INFO' | 'WARN' | 'ERROR' | 'CMD' | 'SUCCESS';
  msg: string;
  time: string;
}

export type IconComponent = React.FC<{ className?: string }>;

export interface UIStructure {
  component: string;
  props?: { [key: string]: any };
  children?: (UIStructure | string)[];
}

export interface AppDefinition {
  id: string;
  name: string;
  description: string;
  icon: IconComponent;
  status: 'available' | 'downloading' | 'installing' | 'installed';
  isCustom?: boolean;
  component?: React.ReactNode;
  q_uri?: string;
  https_url?: string;
  uiStructure?: UIStructure;
  code?: string;
}

export interface URIAssignment {
    appName: string;
    q_uri: string;
    https_url: string;
    timestamp: string;
}

export interface SystemHealth {
    cognitiveEfficiency: number;
    semanticIntegrity: number;
    dataThroughput: number;
    ipsThroughput: number;
    powerEfficiency: number;
    decoherenceFactor: number;
    processingSpeed: number;
    qpuTempEfficiency: number;
    qubitStability: number;
    neuralLoad: number;
    activeThreads: number;
}

export interface SavedProtocol {
  name: string;
  code: string;
  category: 'Security' | 'Communication' | 'Entanglement' | 'Command' | 'Optimization' | 'Simulation' | 'General';
}

// --- Nebula Types (Preserved for compatibility) ---

export enum ModelType {
  FLASH = 'gemini-3-flash-preview',
  PRO = 'gemini-3-pro-preview',
  IMAGE_GEN = 'gemini-2.5-flash-image',
}

export interface AppConfig {
  model: ModelType;
  temperature: number;
  useGrounding: boolean;
}

export interface Attachment {
  file: File;
  previewUrl: string;
  mimeType: string;
  base64: string;
}

export enum Role {
  USER = 'user',
  MODEL = 'model',
  SYSTEM = 'system',
}

export interface Message {
  role: Role;
  text: string;
  timestamp: number;
  isError?: boolean;
  attachments?: Attachment[];
  groundingMetadata?: {
    groundingChunks?: Array<{
      web?: { uri: string; title: string };
    }>;
  };
}
