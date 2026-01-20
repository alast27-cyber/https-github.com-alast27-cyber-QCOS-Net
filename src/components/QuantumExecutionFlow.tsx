
import React from 'react';
import IAIKernelStatus from './IAIKernelStatus';

interface QuantumExecutionFlowProps {
    ipsThroughput: number;
}

const QuantumExecutionFlow: React.FC<QuantumExecutionFlowProps> = ({ ipsThroughput }) => {
    return (
        <div className="h-full flex items-center justify-center">
             <IAIKernelStatus isRecalibrating={false} />
        </div>
    );
};

export default QuantumExecutionFlow;
