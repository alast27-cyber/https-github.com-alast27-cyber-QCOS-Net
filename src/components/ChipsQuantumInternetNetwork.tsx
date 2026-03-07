
import React from 'react';
import CHIPSBrowser from './CHIPSBrowser';
import { AppDefinition, UIStructure } from '../types';

interface ChipsQuantumInternetNetworkProps {
    apps: AppDefinition[];
    onInstallApp: (id: string) => void;
    onToggleAgentQ: () => void;
    onDeployApp?: (details: { name: string; description: string; code: string; uiStructure?: UIStructure }) => void;
}

const ChipsQuantumInternetNetwork: React.FC<ChipsQuantumInternetNetworkProps> = ({ apps, onInstallApp, onToggleAgentQ, onDeployApp }) => {
    // We pass a dummy "Store" app definition to trigger the browser to open the store by default
    const storeAppMock: any = {
        id: 'chips-store-init',
        name: 'App Store',
        q_uri: 'chips://store',
        status: 'installed'
    };

    return (
        <div className="h-full w-full overflow-hidden rounded-lg">
            <CHIPSBrowser 
                initialApp={storeAppMock} 
                onToggleAgentQ={onToggleAgentQ} 
                apps={apps} 
                onInstallApp={onInstallApp}
                onDeployApp={onDeployApp}
            />
        </div>
    );
};

export default ChipsQuantumInternetNetwork;
