/**
 * QAPI: Quantum Entanglement Bridge for AgentQ
 * 
 * Provides a high-dimensional, low-latency interface for accessing 
 * AgentQ supreme authority from any point in the QCOS ecosystem.
 * 
 * Through the use of BroadcastChannel, the "Quantum State" is shared 
 * across all instances (tabs/windows) of the app, ensuring that 
 * changes in one are instantly reflected in the other.
 */

import { agentQService } from './services/agentQService';

export enum EntanglementLevel {
    STABLE = 'stable',
    SUPERPOSITION = 'superposition',
    SINGULARITY = 'singularity'
}

export interface QAPIInterface {
    /** Entangles a new system or app with AgentQ's core cognition */
    entangle: (targetId: string, level?: EntanglementLevel) => void;
    /** Synchronously dispatch a command through the quantum field */
    dispatch: (command: string | any, context?: Record<string, any>) => Promise<any>;
    /** Retrieve the current cognitive state from the shared superposition */
    getCognitionState: () => Promise<any>;
    /** Resolve decoherence in the communication channel */
    resolveDecoherence: () => void;
    /** Subscribe to state changes across the entanglement mesh */
    subscribe: (callback: (payload: any) => void) => () => void;
}

// Quantum Broadcast Channel for Entanglement
const entanglementChannel = new BroadcastChannel('agentq-entanglement-mesh');

export const QAPI: QAPIInterface = {
    entangle: (targetId, level = EntanglementLevel.STABLE) => {
        console.log(`[QAPI] Entangling: ${targetId} at level ${level.toUpperCase()}`);
        entanglementChannel.postMessage({
            type: 'ENTANGLEMENT_INIT',
            targetId,
            level,
            timestamp: Date.now()
        });
    },
    
    dispatch: async (command: string | any, context = {}) => {
        const commandStr = typeof command === 'string' ? command : (command.type || 'UNKNOWN_ACTION');
        console.log(`[QAPI] Quantum Dispatch: "${commandStr}"`);
        
        try {
            // Pulse to local AgentQ service
            const response = await agentQService.sendMessage(commandStr, JSON.stringify(context));
            
            // Pulse to server-side QAPI relay if reachable
            fetch('/api/telemetry', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    type: 'DISPATCH',
                    priority: 'HIGH',
                    payload: { command, context, response }
                })
            }).catch(() => {/* Background relay failed, fallback to local only */});

            // Pulse the result through the local entanglement field (tabs)
            entanglementChannel.postMessage({
                type: 'COMMAND_DISPATCHED',
                command,
                context,
                response,
                timestamp: Date.now()
            });
            
            return response;
        } catch (error) {
            console.error("[QAPI] Decoherence during dispatch:", error);
            throw error;
        }
    },
    
    getCognitionState: async () => {
        return await agentQService.getInsights();
    },
    
    resolveDecoherence: () => {
        console.log("[QAPI] Recalibrating Entanglement Field...");
        entanglementChannel.postMessage({ type: 'RECALIBRATE', timestamp: Date.now() });
    },

    subscribe: (callback) => {
        const handler = (event: MessageEvent) => {
            callback(event.data);
        };
        entanglementChannel.addEventListener('message', handler);
        return () => entanglementChannel.removeEventListener('message', handler);
    }
};

export default QAPI;

