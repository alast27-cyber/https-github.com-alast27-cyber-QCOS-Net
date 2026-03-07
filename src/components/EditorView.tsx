import React, { useState } from 'react';
import EditorWorkspace from './EditorWorkspace';
import { XIcon } from './Icons';

interface EditorViewProps {
    onToggleView: () => void;
    agentQProps?: any; 
}

const EditorView: React.FC<EditorViewProps> = ({ onToggleView }) => {
    // Mock file state for the full screen editor
    const [files, setFiles] = useState<{[key: string]: string}>({
        'App.tsx': '// Full screen editor mode\nimport React from "react";\n\nexport default function App() {\n  return <div>Hello World</div>;\n}'
    });
    const [selectedFile, setSelectedFile] = useState('App.tsx');

    return (
        <div className="fixed inset-0 z-[100] bg-slate-900 flex flex-col">
            <div className="flex justify-between items-center p-2 bg-black/40 border-b border-cyan-900/50">
                <span className="text-cyan-400 font-bold ml-2">Source Nexus: Full Screen Editor</span>
                <button onClick={onToggleView} className="p-2 text-red-400 hover:text-red-300">
                    <XIcon className="w-6 h-6" />
                </button>
            </div>
            <div className="flex-grow min-h-0 p-4">
                <EditorWorkspace 
                    files={files}
                    selectedFile={selectedFile}
                    onSelectFile={setSelectedFile}
                    unsavedChanges={null}
                    onContentChange={(val) => setFiles(prev => ({...prev, [selectedFile]: val}))}
                    onRenameFile={() => {}}
                    onDeleteFile={() => {}}
                />
            </div>
        </div>
    );
};

export default EditorView;