
import React from 'react';
import SecureMailClient from './ChipsMail';
import GlassPanel from './GlassPanel';
import { ServerCogIcon } from './Icons';

const CHIPSNetworkPanel: React.FC = () => {
    // For now, this panel directly houses the mail client.
    // It could be expanded later to include other CHIPS services.
    const userAddress = "operator@chipsmail.qcos"; // Hardcoded for simulation

    return (
        <GlassPanel title={<div className="flex items-center"><ServerCogIcon className="w-5 h-5 mr-2" /> CHIPS Network Services</div>}>
            <div className="h-full">
                <SecureMailClient userAddress={userAddress} />
            </div>
        </GlassPanel>
    );
};

export default CHIPSNetworkPanel;
