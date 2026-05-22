
import React, { useState, useEffect } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from 'recharts';

const MAX_DATA_POINTS = 20;

interface DecoherenceDriftMonitorProps {
    decoherenceFactor: number;
}

const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-900/80 p-2 border border-cyan-400 text-white rounded-md text-sm">
          <p>{`Drift Rate: ${(payload[0]?.value || 0).toFixed(4)}`}</p>
        </div>
      );
    }
    return null;
};

const DecoherenceDriftMonitor: React.FC<DecoherenceDriftMonitorProps> = ({ decoherenceFactor }) => {
    const [data, setData] = useState(() => 
        Array.from({ length: MAX_DATA_POINTS }, (_, i) => ({
            time: i,
            drift: (0.0005 * decoherenceFactor) + (Math.random() - 0.5) * 0.0002
        }))
    );
    const timeRef = React.useRef(MAX_DATA_POINTS);

    useEffect(() => {
        const interval = setInterval(() => {
            setData(prevData => {
                const newData = [...prevData.slice(1)];
                const lastDrift = newData[newData.length - 1].drift;
                // Fluctuation is now smaller, making the base rate more important
                const newDrift = lastDrift + (Math.random() - 0.5) * 0.0001;
                // The base drift is influenced by the efficiency factor
                const baseDrift = 0.0005 * decoherenceFactor;
                newData.push({ time: timeRef.current++, drift: Math.max(baseDrift * 0.8, newDrift) });
                return newData;
            });
        }, 1000);
        return () => clearInterval(interval);
    }, [decoherenceFactor]);

    return (
        <div className="flex flex-col h-full">
            <h3 className="text-cyan-300 text-sm tracking-widest flex-shrink-0 text-center">DECOHERENCE DRIFT</h3>
            <div className="flex-grow w-full h-full mt-2">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="1 1" stroke="rgba(255, 100, 100, 0.2)" />
                        <XAxis dataKey="time" stroke="rgba(255, 255, 255, 0.5)" tick={{ fontSize: 10 }} domain={['dataMin', 'dataMax']} type="number" />
                        <YAxis stroke="rgba(255, 255, 255, 0.5)" domain={[0, 0.002]} tick={{ fontSize: 10 }} tickFormatter={(v) => v.toFixed(4)} />
                        <Tooltip content={<CustomTooltip />} cursor={{stroke: 'red', strokeWidth: 1}}/>
                        <ReferenceLine y={0.0015} label={{ value: 'Threshold', position: 'insideTopLeft', fill: '#f87171', fontSize: 10 }} stroke="#f87171" strokeDasharray="3 3" />
                        <Line type="monotone" dataKey="drift" stroke="#fb923c" strokeWidth={2} dot={false} activeDot={{r: 4}} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default DecoherenceDriftMonitor;
