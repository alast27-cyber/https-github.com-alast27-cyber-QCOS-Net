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
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`AgentQ API error: ${response.status} - ${errorText}`);
        }

        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            const responseText = await response.text();
            throw new Error(`Expected JSON response but received ${contentType || 'unknown'} content type. Response body: ${responseText.substring(0, 200)}...`);
        }

        return response.json();
    }

    async getInsights(): Promise<AgentQResponse> {
        const response = await fetch(`${this.baseUrl}/insights`);
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`AgentQ API error: ${response.status} - ${errorText}`);
        }

        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            const responseText = await response.text();
            throw new Error(`Expected JSON response but received ${contentType || 'unknown'} content type. Response body: ${responseText.substring(0, 200)}...`);
        }

        return response.json();
    }
}

export const agentQService = new AgentQService();
