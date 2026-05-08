
import React, { useState } from 'react';
import { ClipboardIcon, CheckCircle2Icon } from './Icons';

const highlightTsx = (line: string) => {
    line = line.replace(/(\/\/.*)$/, '<span class="text-gray-500">$1</span>');
    line = line.replace(/\b(import|from|export|default|const|let|return|as|interface|type|React)\b/g, '<span class="text-red-400">$1</span>');
    line = line.replace(/\b(useState|useEffect|useRef|React\.FC)\b/g, '<span class="text-teal-300">$1</span>');
    line = line.replace(/(&lt;\/?)([A-Z]\w+)/g, '$1<span class="text-green-300">$2</span>');
    line = line.replace(/(\w+)=/g, '<span class="text-yellow-300">$1</span>=');
    line = line.replace(/(['"`])(.*?)\1/g, '<span class="text-amber-400">$1$2$1</span>');
    line = line.replace(/([{}()[\];,.=&gt;])/g, '<span class="text-cyan-400">$1</span>');
    return line;
};

const SyntaxHighlighter = ({ code, language = 'tsx' }: { code: string, language?: string }) => {
  const [isCopied, setIsCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  const highlight = (line: string) => {
    line = line.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return highlightTsx(line);
  };

  return (
    <div className="relative group h-full">
        <pre className="w-full h-full bg-black/30 border border-blue-500/50 rounded-md p-2 text-white font-mono text-xs overflow-auto">
          <code dangerouslySetInnerHTML={{ __html: code.split('\n').map(highlight).join('\n') }} />
        </pre>
        <button
            onClick={handleCopy}
            className="absolute top-2 right-2 p-1.5 rounded-md bg-slate-800/50 hover:bg-slate-700/70 text-cyan-300 opacity-0 group-hover:opacity-100 transition-opacity"
        >
            {isCopied ? <CheckCircle2Icon className="w-4 h-4 text-green-400" /> : <ClipboardIcon className="w-4 h-4" />}
        </button>
    </div>
  );
};

export default SyntaxHighlighter;
