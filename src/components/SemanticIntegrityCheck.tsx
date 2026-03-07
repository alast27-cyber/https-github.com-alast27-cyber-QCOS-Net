
import React, { useState, useEffect } from 'react';
import { SearchIcon, FileCodeIcon, BrainCircuitIcon, GitBranchIcon, CheckCircle2Icon, AlertTriangleIcon } from './Icons';

interface SemanticIntegrityCheckProps {
  command: string | null;
}

const stages = [
  { name: 'Lexical Analysis', icon: SearchIcon },
  { name: 'Syntax Validation', icon: FileCodeIcon },
  { name: 'Intent Recognition', icon: BrainCircuitIcon },
  { name: 'Command Mapping', icon: GitBranchIcon },
];

type StageStatus = 'pending' | 'active' | 'complete' | 'error';

const SemanticIntegrityCheck: React.FC<SemanticIntegrityCheckProps> = ({ command }) => {
  const [stageStatuses, setStageStatuses] = useState<StageStatus[]>(() => Array(stages.length).fill('pending'));
  const [correction, setCorrection] = useState<{ original: string; corrected: string } | null>(null);

  useEffect(() => {
    if (!command) {
        return;
    }
    
    setCorrection(null);
    setStageStatuses(Array(stages.length).fill('pending'));

    const typoRegex = /algorithem/i;
    const hasTypo = typoRegex.test(command);
    
    let currentStage = 0;
    const interval = setInterval(() => {
      if (hasTypo && currentStage === 0) {
        setStageStatuses(prev => {
          const newStatuses = [...prev];
          newStatuses[0] = 'error';
          return newStatuses;
        });
        setCorrection({
          original: command.match(typoRegex)![0],
          corrected: 'algorithm',
        });
        clearInterval(interval);
        return;
      }

      if (currentStage < stages.length) {
        setStageStatuses(prev => {
          const newStatuses = [...prev];
          if (currentStage > 0) newStatuses[currentStage - 1] = 'complete';
          newStatuses[currentStage] = 'active';
          return newStatuses;
        });
        currentStage++;
      } else {
         setStageStatuses(prev => prev.map(() => 'complete'));
        clearInterval(interval);
      }
    }, 300);

    return () => clearInterval(interval);
  }, [command]);
  
  if (!command) return null;

  const getStatusIndicator = (status: StageStatus) => {
    switch (status) {
      case 'active':
        return <div className="w-3 h-3 rounded-full bg-cyan-400 animate-pulse border border-cyan-200" />;
      case 'complete':
        return <CheckCircle2Icon className="w-4 h-4 text-green-400" />;
      case 'error':
        return <AlertTriangleIcon className="w-4 h-4 text-red-400" />;
      case 'pending':
      default:
        return <div className="w-3 h-3 rounded-full bg-slate-700 border border-slate-500" />;
    }
  };

  return (
    <div className="animate-fade-in">
        <label className="block text-cyan-400 mb-1 text-sm font-semibold">
            Semantic Integrity Protocol
        </label>
        <div className="bg-black/30 border border-blue-500/50 rounded-md p-3 text-sm">
            <div className="flex justify-around items-start">
                {stages.map((stage, index) => (
                    <React.Fragment key={stage.name}>
                        <div className={`flex flex-col items-center text-center transition-opacity duration-300 w-24 ${stageStatuses[index] === 'pending' ? 'opacity-50' : 'opacity-100'}`}>
                            <div className="flex items-center justify-center w-8 h-8 rounded-full bg-slate-900/50 border border-cyan-800 mb-1">
                                <stage.icon className={`w-5 h-5 ${stageStatuses[index] === 'active' ? 'text-cyan-300' : 'text-cyan-500'}`} />
                            </div>
                            <span className="text-xs text-cyan-300 h-8 flex items-center">{stage.name}</span>
                            <div className="mt-1 h-4 flex items-center">
                                {getStatusIndicator(stageStatuses[index])}
                            </div>
                        </div>
                        {index < stages.length - 1 && (
                            <div className="flex-grow h-0.5 rounded-full mt-4 mx-2 transition-colors duration-300 min-w-4
                                ${stageStatuses[index] === 'complete' ? 'bg-green-500' : stageStatuses[index] === 'error' ? 'bg-red-500' : 'bg-slate-700'}" 
                            />
                        )}
                    </React.Fragment>
                ))}
            </div>
            {correction && (
                <div className="mt-3 p-2 bg-yellow-900/50 border border-yellow-600 rounded-md text-xs text-center animate-fade-in">
                    <span className="font-bold text-yellow-300">Correction Suggested: </span>
                    <span className="text-red-400 line-through">{correction.original}</span> &rarr; <span className="text-green-400">{correction.corrected}</span>
                </div>
            )}
        </div>
    </div>
  );
};
export default SemanticIntegrityCheck;
