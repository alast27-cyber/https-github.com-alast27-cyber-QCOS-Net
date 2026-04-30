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
    dispatch: (command: string, context?: Record<string, any>) => Promise<any>;
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
    
    dispatch: async (command, context = {}) => {
        console.log(`[QAPI] Quantum Dispatch: "${command}"`);
        try {
            const response = await agentQService.sendMessage(command, JSON.stringify(context));
            
            // Pulse the result through the entanglement field
            entanglementChannel.postMessage({
                type: 'COMMAND_DISPATCHED',
                command,
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

