import React from 'react';

const KpiDisplay: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center h-full p-4">
      <div className="text-center w-full">
        <h2 className="text-sm sm:text-base text-cyan-300 tracking-widest">SYSTEM HEALTH SCORE</h2>
        <p className="text-6xl sm:text-7xl font-bold text-white my-2 sm:my-3">98.7</p>
        <div className="w-full h-1 bg-cyan-400/30 mt-2 mb-3 rounded-full">
            <div className="w-[98.7%] h-1 bg-cyan-300 shadow-[0_0_8px_theme(colors.cyan.300)] rounded-full"></div>
        </div>
        <div className="text-xs sm:text-sm space-y-1">
            <p className="text-cyan-400">Qubit Fidelity: <span className="text-white">99.94%</span></p>
            <p className="text-cyan-400">Coherence Time: <span className="text-white">124μs</span></p>
        </div>
      </div>
    </div>
  );
};

export default KpiDisplay;