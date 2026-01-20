import React, { useState, useEffect } from 'react';
import { UIStructure } from '../types';
import HolographicPreviewRenderer from './HolographicPreviewRenderer';

interface DeployedAppWrapperProps {
    structure: UIStructure;
    code: string;
}

const DeployedAppWrapper: React.FC<DeployedAppWrapperProps> = ({ structure, code }) => {
    const [appState, setAppState] = useState<{ [key: string]: any }>({});

    useEffect(() => {
        // Extract initial state from code
        const initialStateRegex = /const \[(\w+), set\w+\] = useState\((.*?)\);/g;
        let match;
        const newInitialState: { [key: string]: any } = {};
        while ((match = initialStateRegex.exec(code)) !== null) {
            const stateVar = match[1]; 
            const rawInitialValue = match[2];
            try { 
                newInitialState[stateVar] = JSON.parse(rawInitialValue.replace(/'/g, '"')); 
            } catch (e) { 
                newInitialState[stateVar] = rawInitialValue.replace(/^['"`]|['"`]$/g, ''); 
            }
        }
        setAppState(newInitialState);
    }, [code]);

    const handleAction = (handlerName: string) => {
        // Simple heuristic state updater logic similar to the previewer
        const functionRegex = new RegExp(`const ${handlerName} = \\(\\) =>\\s*\\{([\\s\\S]*?)\\}`, 'm');
        const match = code.match(functionRegex);
        const functionBody = match ? match[1].trim() : '';

        const incrementRegex = /set(\w+)\(\s*(\w+)\s*\+\s*1\s*\)/;
        const toggleRegex = /set(\w+)\(\s*!(\w+)\s*\)/;

        let incMatch = functionBody.match(incrementRegex);
        let togMatch = functionBody.match(toggleRegex);

        if (incMatch) {
            const varName = incMatch[1].toLowerCase().replace('prev', '');
            const stateKey = Object.keys(appState).find(k => k.toLowerCase().includes(varName) || varName.includes(k.toLowerCase()));
            if (stateKey) {
                setAppState(p => ({ ...p, [stateKey]: (typeof p[stateKey] === 'number' ? p[stateKey] + 1 : 1) }));
            }
        } else if (togMatch) {
            const varName = togMatch[1].toLowerCase();
            const stateKey = Object.keys(appState).find(k => k.toLowerCase().includes(varName) || varName.includes(k.toLowerCase()));
            if (stateKey) {
                setAppState(p => ({ ...p, [stateKey]: !p[stateKey] }));
            }
        }
    };

    return (
        <div className="h-full w-full bg-slate-950 text-white overflow-hidden rounded-lg">
            <HolographicPreviewRenderer structure={structure} state={appState} onAction={handleAction} />
        </div>
    );
};

export default DeployedAppWrapper;