
import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { ChartDataPoint } from '../types';

const data: ChartDataPoint[] = [
  { name: 't-60s', coherence: 99.8 },
  { name: 't-45s', coherence: 99.82 },
  { name: 't-30s', coherence: 99.81 },
  { name: 't-15s', coherence: 99.85 },
  { name: 'now', coherence: 99.9 },
  { name: 't+15s', coherence: 99.88 },
  { name: 't+30s', coherence: 99.87 },
];

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-900/80 p-2 border border-cyan-400 text-white rounded-md text-sm">
          <p className="label">{`${label} : ${(payload[0]?.value || 0)}%`}</p>
        </div>
      );
    }
    return null;
  };

const SystemHealthChart: React.FC = () => {
  return (
    <div className="flex flex-col h-full">
      <h3 className="text-cyan-300 text-sm tracking-widest flex-shrink-0 text-center md:text-left">QUBIT COHERENCE PREDICTION</h3>
      <div className="flex-grow w-full h-full mt-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
            <CartesianGrid strokeDasharray="1 1" stroke="rgba(0, 255, 255, 0.2)" />
            <XAxis dataKey="name" stroke="rgba(0, 255, 255, 0.7)" tick={{ fontSize: 10 }} />
            <YAxis stroke="rgba(0, 255, 255, 0.7)" domain={[99.7, 100]} tick={{ fontSize: 10 }}/>
            <Tooltip content={<CustomTooltip />} cursor={{stroke: 'cyan', strokeWidth: 1}}/>
            <Line type="monotone" dataKey="coherence" stroke="#00FFFF" strokeWidth={2} dot={{r: 3}} activeDot={{r: 6}} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default SystemHealthChart;
