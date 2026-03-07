
import React from 'react';
import GlassPanel from './GlassPanel';
import { GlobeIcon } from './Icons';

const GlobalAbundanceEngine: React.FC = () => {
    return (
        <GlassPanel title={<div className="flex items-center"><GlobeIcon className="w-6 h-6 mr-2" />Global Abundance Engine</div>}>
            <div className="p-4 h-full flex items-center justify-center text-cyan-400 text-center">
                <p>Global Abundance Engine functionality is temporarily offline for diagnostics.</p>
            </div>
        </GlassPanel>
    );
};

export default GlobalAbundanceEngine;
    