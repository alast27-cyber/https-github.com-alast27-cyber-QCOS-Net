// src/services/agentQService.ts

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
        const response = await fetch(`${this.baseUrl}/message`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, context }),
        });
        return response.json();
    }

    async getInsights(): Promise<AgentQResponse> {
        const response = await fetch(`${this.baseUrl}/insights`);
        return response.json();
    }
}

export const agentQService = new AgentQService();
