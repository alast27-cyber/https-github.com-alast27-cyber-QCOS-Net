
import React, { useState } from 'react';

interface AICommandConsoleProps {
  onExecute: (command: string) => void;
}

const AICommandConsole: React.FC<AICommandConsoleProps> = ({ onExecute }) => {
  const [inputValue, setInputValue] = useState('');

  const handleExecute = () => {
    if (inputValue.trim()) {
        onExecute(inputValue.trim());
        setInputValue('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleExecute();
    }
  };

  return (
    <div className="h-full flex flex-col">
      <label htmlFor="ai-command-input" className="block text-cyan-400 mb-1 text-sm font-semibold">
        Command Input
      </label>
      <textarea
        id="ai-command-input"
        rows={3}
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onKeyDown={handleKeyDown}
        className="w-full bg-black/30 border border-blue-500/50 rounded-md p-2 text-white placeholder:text-gray-500 focus:ring-1 focus:ring-cyan-400 focus:outline-none flex-grow"
        placeholder="> e.g., 'run quantum teleportation'"
      />
      <button 
        onClick={handleExecute}
        className="mt-2 w-full bg-cyan-500/30 hover:bg-cyan-500/50 border border-cyan-500/50 text-cyan-200 font-bold py-2 px-4 rounded transition-colors"
      >
        EXECUTE
      </button>
    </div>
  );
};

export default AICommandConsole;
