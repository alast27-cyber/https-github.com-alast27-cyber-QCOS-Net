
import React, { useState, useEffect, useRef } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const MAX_DATA_POINTS = 30;

interface QubitStabilityChartProps {
    qubitStability: number; // Interval in ms, lower is more chaotic
}

const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-900/80 p-2 border border-yellow-400 text-white rounded-md text-sm">
          <p>{`Update Interval: ${(payload[0]?.value || 0).toFixed(0)} ms`}</p>
        </div>
      );
    }
    return null;
};

const QubitStabilityChart: React.FC<QubitStabilityChartProps> = ({ qubitStability }) => {
    const [data, setData] = useState(() =>
        Array.from({ length: MAX_DATA_POINTS }, (_, i) => ({
            time: i,
            stability: 200 // Start at a stable value
        }))
    );
    const timeRef = useRef(MAX_DATA_POINTS);

    useEffect(() => {
        const interval = setInterval(() => {
            setData(prevData => {
                const newData = [...prevData.slice(1)];
                // qubitStability is the new target value. We can add a bit of noise for visual effect.
                const noisyStability = qubitStability + (Math.random() - 0.5) * 10;
                newData.push({ time: timeRef.current++, stability: noisyStability });
                return newData;
            });
        }, 1000); // Update the chart every second

        return () => clearInterval(interval);
    }, [qubitStability]); // The dependency ensures we use the latest stability value inside the interval

    return (
        <div className="flex flex-col h-full">
            <h3 className="text-cyan-300 text-sm tracking-widest flex-shrink-0 text-center">QUBIT STABILITY</h3>
            <p className="text-cyan-400 text-xs text-center -mt-1 mb-1">Update interval in ms (lower is chaotic)</p>
            <div className="flex-grow w-full h-full">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="1 1" stroke="rgba(255, 255, 100, 0.2)" />
                        <XAxis dataKey="time" stroke="rgba(255, 255, 255, 0.5)" tick={{ fontSize: 10 }} domain={['dataMin', 'dataMax']} type="number" hide={true} />
                        <YAxis stroke="rgba(255, 255, 255, 0.5)" domain={[50, 250]} tick={{ fontSize: 10 }} tickFormatter={(v) => v.toFixed(0)} />
                        <Tooltip content={<CustomTooltip />} cursor={{stroke: 'yellow', strokeWidth: 1}}/>
                        <Line type="monotone" dataKey="stability" stroke="#facc15" strokeWidth={2} dot={false} activeDot={{r: 4}} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default QubitStabilityChart;
