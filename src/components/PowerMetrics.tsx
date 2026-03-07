
import React, { useState, useEffect } from 'react';
import { ZapIcon, ChartBarIcon, ThermometerIcon, RocketLaunchIcon } from './Icons';

interface PowerMetricsProps {
  powerEfficiency: number;
}

const PowerMetrics: React.FC<PowerMetricsProps> = ({ powerEfficiency }) => {
  const [totalPowerDraw, setTotalPowerDraw] = useState(1250); // Watts
  const [pue, setPue] = useState(1.15);
  const [cryoTemp, setCryoTemp] = useState(0.015); // Kelvin
  const [cryoStatus, setCryoStatus] = useState('Optimal');
  const [qnnPrediction, setQnnPrediction] = useState(1.12); // Predicted PUE

  useEffect(() => {
    // Simulate real-time data updates influenced by powerEfficiency prop
    const interval = setInterval(() => {
      // A higher efficiency factor should lower power draw and PUE.
      // We model this by treating efficiency as an inverse multiplier on base values.
      const inefficiencyFactor = 1 / (powerEfficiency || 1);

      const baseDraw = 1250 * inefficiencyFactor;
      setTotalPowerDraw(baseDraw + (Math.random() - 0.5) * 10);

      const basePue = 1.15 * inefficiencyFactor;
      setPue(basePue + (Math.random() - 0.5) * 0.01);

      const baseTemp = 0.015 * inefficiencyFactor;
      setCryoTemp(baseTemp + (Math.random() - 0.5) * 0.001);
      
      const baseQnnPrediction = 1.12 * inefficiencyFactor;
      setQnnPrediction(baseQnnPrediction + (Math.random() - 0.5) * 0.005);

    }, 3000);
    return () => clearInterval(interval);
  }, [powerEfficiency]);

  // This effect handles the status based on temperature
  useEffect(() => {
    if (cryoTemp > 0.016 || cryoTemp < 0.014) {
      setTimeout(() => setCryoStatus('Warning'), 0);
    } else {
      setTimeout(() => setCryoStatus('Optimal'), 0);
    }
  }, [cryoTemp]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Optimal': return 'text-green-400';
      case 'Warning': return 'text-yellow-400';
      case 'Critical': return 'text-red-400';
      default: return 'text-cyan-300';
    }
  };

  return (
      <div className="flex flex-col space-y-4 p-4 text-cyan-300 h-full">
        {/* Total Power Draw */}
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <ZapIcon className="h-6 w-6 mr-3 text-cyan-400" />
            <div>
              <p className="text-sm text-gray-400">Total Power Draw</p>
              <p className="text-2xl font-bold">{(totalPowerDraw || 0).toFixed(1)} W</p>
            </div>
          </div>
          <div className="text-sm text-gray-400">
            <p className="text-right">QPU: 800W</p>
            <p className="text-right">Cryo: 350W</p>
            <p className="text-right">Ctrl: 100W</p>
          </div>
        </div>

        {/* Power Usage Effectiveness (PUE) */}
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <ChartBarIcon className="h-6 w-6 mr-3 text-cyan-400" />
            <div>
              <p className="text-sm text-gray-400">PUE (Power Usage Effectiveness)</p>
              <p className="text-2xl font-bold">{(pue || 0).toFixed(3)}</p>
            </div>
          </div>
          <div className="text-sm text-gray-400">
            <p className="text-right">Target: 1.10</p>
          </div>
        </div>

        {/* Cryo-Cooling Status */}
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <ThermometerIcon className="h-6 w-6 mr-3 text-cyan-400" />
            <div>
              <p className="text-sm text-gray-400">Cryo-Cooling Status</p>
              <p className="text-2xl font-bold">{Math.max(0, cryoTemp || 0).toFixed(3)} K</p>
            </div>
          </div>
          <div className={`flex flex-col items-end text-sm ${getStatusColor(cryoStatus)}`}>
            <p>{cryoStatus}</p>
            <p className="text-gray-400">Pump: Active</p>
          </div>
        </div>

        {/* QNN Predicted Efficiency */}
        <div className="flex items-center justify-between border-t border-cyan-700/50 pt-4 mt-2">
          <div className="flex items-center">
            <RocketLaunchIcon className="h-6 w-6 mr-3 text-purple-400" />
            <div>
              <p className="text-sm text-gray-400">QNN Predicted PUE (24h)</p>
              <p className="text-2xl font-bold text-purple-300">{(qnnPrediction || 0).toFixed(3)}</p>
            </div>
          </div>
          <div className="text-sm text-gray-400 text-right">
            <p>Optimization Potential</p>
          </div>
        </div>
      </div>
  );
};

export default PowerMetrics;
