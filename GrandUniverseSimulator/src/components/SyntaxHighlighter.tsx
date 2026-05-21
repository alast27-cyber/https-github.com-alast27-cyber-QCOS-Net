
import React, { useState } from 'react';
import { ClipboardIcon, CheckCircle2Icon } from './Icons';

const highlightQLang = (line: string) => {
    // Comments
    line = line.replace(/(\/\/.*)$/, '<span class="text-gray-500">$1</span>');
    if (line.trim().startsWith('#')) {
      return `<span class="text-gray-500">${line}</span>`;
    }
    // Directives and Keywords
    // Use a single regex with alternations to catch all keyword types, including those with special characters or spaces.
    // The order of alternations is important to prevent partial matches (e.g., 'LOOP' in 'LOOP END').
    // Use $& to insert the entire matched string.
    line = line.replace(
        /\bLOOP END\b|@LOOP_START|\bSECURITY::\w+::\w+\b|\b(QREG|CREG|ALLOC|EXECUTE|IF|THEN|LOOP|EOF)\b/g,
        '<span class="text-cyan-400">$&</span>'
    );

    // Operators
    line = line.replace(/\b(OP::H|OP::X|OP::Y|OP::Z|OP::S|OP::T|OP::CNOT|OP::C4Z|OP::MEASURE|OP::RETURN_RESULT|OP::I|OP::CZ|OP::RZ|OP::RY)\b/g, '<span class="text-yellow-400">$1</span>');
    // Registers & Parameters
    line = line.replace(/\b(q\[\d+\]|c\[\d+\]|PI\/\d+)\b/g, '<span class="text-purple-400">$1</span>');
    // Numbers
    line = line.replace(/\b(0x[0-9a-fA-F]+\.\.\.[0-9a-fA-F]+|\d+)\b/g, '<span class="text-green-400">$1</span>');
    return line;
};

const highlightTsx = (line: string) => {
    // Comments
    line = line.replace(/(\/\/.*)$/, '<span class="text-gray-500">$1</span>');
    // Keywords
    line = line.replace(/\b(import|from|export|default|const|let|return|as|interface|type|React)\b/g, '<span class="text-red-400">$1</span>');
    // React Hooks & Types
    line = line.replace(/\b(useState|useEffect|useRef|React\.FC)\b/g, '<span class="text-teal-300">$1</span>');
    // Component Tags (PascalCase)
    line = line.replace(/(&lt;\/?)([A-Z]\w+)/g, '$1<span class="text-green-300">$2</span>');
    // HTML-like tags (lowercase)
    line = line.replace(/(&lt;\/?)(\w+)/g, (match, p1, p2) => {
       if (p2.match(/^[A-Z]/)) return match; // Already handled by component tags
       return `${p1}<span class="text-gray-400">${p2}</span>`;
    });
    // Props
    line = line.replace(/(\w+)=/g, '<span class="text-yellow-300">$1</span>=');
    // Strings
    line = line.replace(/(['"`])(.*?)\1/g, '<span class="text-amber-400">$1$2$1</span>');
     // Punctuation
    line = line.replace(/([{}()[\];,.=&gt;])/g, '<span class="text-cyan-400">$1</span>');

    return line;
};


const SyntaxHighlighter = ({ code, language = 'q-lang' }: { code: string, language?: string }) => {
  const [isCopied, setIsCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  const highlight = (line: string) => {
    // Basic HTML entity encoding for safety
    line = line.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    
    switch (language) {
      case 'tsx':
      case 'typescript':
      case 'jsx':
        return highlightTsx(line);
      case 'q-lang':
      default:
        return highlightQLang(line);
    }
  };

  return (
    <div className="relative group h-full">
        <pre className="w-full h-full bg-black/30 border border-blue-500/50 rounded-md p-2 text-white font-mono text-xs overflow-auto">
          <code dangerouslySetInnerHTML={{ __html: code.split('\n').map(highlight).join('\n') }} />
        </pre>
        <button
            onClick={handleCopy}
            className="absolute top-2 right-2 p-1.5 rounded-md bg-slate-800/50 hover:bg-slate-700/70 text-cyan-300 opacity-0 group-hover:opacity-100 transition-opacity"
            aria-label="Copy code"
        >
            {isCopied ? <CheckCircle2Icon className="w-4 h-4 text-green-400" /> : <ClipboardIcon className="w-4 h-4" />}
        </button>
    </div>
  );
};

export default SyntaxHighlighter;
