import React, { createContext, useContext, useState, useEffect } from 'react';
import { UniversalBridge } from '../bridge/UniversalBridge';

interface PersistenceContextType {
    saveWeights: (weights: any) => Promise<void>;
    loadWeights: () => Promise<any>;
    lastSaved: string | null;
}

const PersistenceContext = createContext<PersistenceContextType | undefined>(undefined);

export const PersistenceProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [lastSaved, setLastSaved] = useState<string | null>(null);

    const saveWeights = async (weights: any) => {
        try {
            const result = await UniversalBridge.saveWeights(weights);
            if (result.success) {
                setLastSaved(new Date().toISOString());
                console.log("[PERSISTENCE] Weights saved to:", result.path);
            } else {
                console.error("[PERSISTENCE] Failed to save weights:", result.error);
            }
        } catch (e) {
            console.error("[PERSISTENCE] Error saving weights:", e);
        }
    };

    const loadWeights = async () => {
        // In a real app, we'd add a loadWeights method to the bridge
        // For now, we simulate loading from local storage or mock
        const stored = localStorage.getItem('qiai_weights');
        return stored ? JSON.parse(stored) : null;
    };

    return (
        <PersistenceContext.Provider value={{ saveWeights, loadWeights, lastSaved }}>
            {children}
        </PersistenceContext.Provider>
    );
};

export const usePersistence = () => {
    const context = useContext(PersistenceContext);
    if (!context) {
        throw new Error('usePersistence must be used within a PersistenceProvider');
    }
    return context;
};
