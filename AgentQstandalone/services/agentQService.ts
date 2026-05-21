// src/services/agentQService.ts
import { safeFetch } from '../shared/utils/api';

export interface AgentQResponse {
    message: string;
    data?: any;
    error?: string;
}

export class AgentQService {
    private baseUrl: string;

    constructor(baseUrl: string = '/api/agentq') {
        this.baseUrl = baseUrl;
    }

    async sendMessage(message: string, context: string | null): Promise<AgentQResponse> {
        return safeFetch<AgentQResponse>(`${this.baseUrl}/message`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, context }),
        });
    }

    async getInsights(): Promise<AgentQResponse> {
        return safeFetch<AgentQResponse>(`${this.baseUrl}/insights`);
    }
}

export const agentQService = new AgentQService();
