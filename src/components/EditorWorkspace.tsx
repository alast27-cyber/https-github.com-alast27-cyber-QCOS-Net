
import React, { useState, useEffect } from 'react';
import { FileCodeIcon, PencilSquareIcon, TrashIcon, CheckCircle2Icon, XIcon, SparklesIcon, LoaderIcon, ArrowRightIcon } from './Icons';
import { GoogleGenAI } from '@google/genai';
import { generateContentWithRetry } from '../utils/gemini';

import MonacoEditorWrapper from './MonacoEditorWrapper';

const FileExplorer: React.FC<{ 
    files: string[], 
    selectedFile: string, 
    onSelect: (file: string) => void,
    onRename: (oldName: string, newName: string) => void,
    onDelete: (file: string) => void
}> = ({ files, selectedFile, onSelect, onRename, onDelete }) => {
    const [editingFile, setEditingFile] = useState<string | null>(null);
    const [editValue, setEditValue] = useState('');

    const startEditing = (file: string, e: React.MouseEvent) => {
        e.stopPropagation();
        setEditingFile(file);
        setEditValue(file);
    };

    const cancelEditing = (e?: React.MouseEvent) => {
        e?.stopPropagation();
        setEditingFile(null);
        setEditValue('');
    };

    const submitRename = (e?: React.MouseEvent) => {
        e?.stopPropagation();
        if (editingFile && editValue && editValue !== editingFile) {
            onRename(editingFile, editValue);
        }
        setEditingFile(null);
        setEditValue('');
    };

    const handleDelete = (file: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (window.confirm(`Delete ${file}?`)) {
            onDelete(file);
        }
    };

    return (
        <div className="h-full bg-black/30 p-2 overflow-y-auto border-r border-cyan-800/50">
            <h3 className="text-sm font-semibold text-cyan-300 mb-2 px-1">File Structure</h3>
            <div className="space-y-1">
                {files.map(file => (
                    <div
                        key={file}
                        onClick={() => onSelect(file)}
                        className={`group w-full text-left text-xs p-1.5 rounded flex items-center gap-2 font-mono transition-colors cursor-pointer ${selectedFile === file ? 'bg-cyan-900/50 border border-cyan-700/50 text-cyan-200' : 'text-cyan-400 hover:bg-cyan-500/10'}`}
                    >
                        <FileCodeIcon className="w-4 h-4 flex-shrink-0" />
                        
                        {editingFile === file ? (
                            <div className="flex items-center flex-grow gap-1" onClick={e => e.stopPropagation()}>
                                <input 
                                    className="bg-black/50 text-white border border-cyan-500 rounded px-1 py-0.5 w-full min-w-0 outline-none"
                                    value={editValue}
                                    onChange={e => setEditValue(e.target.value)}
                                    onKeyDown={e => {
                                        if (e.key === 'Enter') submitRename();
                                        if (e.key === 'Escape') cancelEditing();
                                    }}
                                    autoFocus
                                    onClick={e => e.stopPropagation()}
                                />
                                <button onClick={submitRename} className="text-green-400 hover:text-green-300"><CheckCircle2Icon className="w-3 h-3"/></button>
                                <button onClick={cancelEditing} className="text-red-400 hover:text-red-300"><XIcon className="w-3 h-3"/></button>
                            </div>
                        ) : (
                            <>
                                <span className="truncate flex-grow">{file}</span>
                                <div className="hidden group-hover:flex items-center gap-1 opacity-80">
                                    <button onClick={(e) => startEditing(file, e)} className="text-cyan-300 hover:text-white p-0.5 rounded hover:bg-cyan-900/50" title="Rename"><PencilSquareIcon className="w-3 h-3"/></button>
                                    <button onClick={(e) => handleDelete(file, e)} className="text-red-400 hover:text-red-200 p-0.5 rounded hover:bg-red-900/50" title="Delete"><TrashIcon className="w-3 h-3"/></button>
                                </div>
                            </>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

const Editor: React.FC<{ content: string, fileName: string, onContentChange: (newContent: string) => void }> = ({ content, fileName, onContentChange }) => {
    const [aiPrompt, setAiPrompt] = useState('');
    const [isGenerating, setIsGenerating] = useState(false);

    // Determine language based on file extension
    const getLanguage = (fname: string) => {
        if (fname.endsWith('.q') || fname.endsWith('.ql')) return 'q-lang';
        if (fname.endsWith('.ts') || fname.endsWith('.tsx')) return 'typescript';
        if (fname.endsWith('.js') || fname.endsWith('.jsx')) return 'javascript';
        if (fname.endsWith('.json')) return 'json';
        if (fname.endsWith('.css')) return 'css';
        if (fname.endsWith('.html')) return 'html';
        return 'plaintext';
    };

    const handleAiEdit = async () => {
        if (!aiPrompt.trim()) return;

        setIsGenerating(true);
        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            const systemInstruction = "You are an expert React/TypeScript developer. Your task is to modify or generate code based on the user's request. RETURN ONLY THE RAW CODE. Do not include markdown backticks, explanations, or any other text. If the user asks to modify existing code, return the FULL updated file content.";
            
            const prompt = `Request: ${aiPrompt}\n\nCurrent Code:\n${content}`;

            const response = await generateContentWithRetry(ai, {
                model: 'gemini-3-pro-preview',
                contents: prompt,
                config: { systemInstruction }
            });

            let newCode = response.text || '';
            newCode = newCode.replace(/^```(tsx|ts|javascript|js|jsx)?\n/, '').replace(/\n```$/, '');
            
            onContentChange(newCode);
            setAiPrompt('');
        } catch (e) {
            console.error("Agent Q Generation Error:", e);
        } finally {
            setIsGenerating(false);
        }
    };

    return (
        <div className="h-full flex flex-col relative group/editor">
            {/* Agent Q Integration Bar */}
            <div className="p-2 bg-black/40 border-b border-cyan-800/30 flex gap-2 items-center backdrop-blur-sm">
                <div className="flex items-center justify-center w-6 h-6 rounded-full bg-purple-900/50 border border-purple-500/50">
                    <SparklesIcon className="w-3 h-3 text-purple-300 animate-pulse" />
                </div>
                <input 
                    type="text" 
                    value={aiPrompt}
                    onChange={(e) => setAiPrompt(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAiEdit()}
                    placeholder="Ask Agent Q to generate or modify this code..."
                    disabled={isGenerating}
                    className="flex-grow bg-transparent border-none text-xs text-white placeholder-cyan-700/70 focus:outline-none font-mono"
                />
                <button 
                    onClick={handleAiEdit} 
                    disabled={isGenerating || !aiPrompt.trim()}
                    className="text-cyan-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed p-1 rounded hover:bg-white/10 transition-colors"
                >
                    {isGenerating ? <LoaderIcon className="w-4 h-4 animate-spin"/> : <ArrowRightIcon className="w-4 h-4"/>}
                </button>
            </div>

            <div className="relative flex-grow min-h-0 bg-black/20">
                <MonacoEditorWrapper
                    code={content}
                    onChange={(val) => onContentChange(val || '')}
                    language={getLanguage(fileName)}
                    theme="qcos-dark"
                />
            </div>
        </div>
    );
};

const Preview: React.FC = () => {
    return (
        <div className="h-full flex items-center justify-center text-cyan-600 italic p-4 text-center bg-black/20">
            <div className="max-w-xs">
                <SparklesIcon className="w-12 h-12 mx-auto mb-3 opacity-50 text-purple-400" />
                <p className="mb-2">Interactive Live Preview</p>
                <p className="text-xs opacity-70">Use Agent Q in the Code tab to generate a UI structure, then switch here to visualize it.</p>
            </div>
        </div>
    );
};

interface EditorWorkspaceProps {
    files: { [key: string]: string };
    selectedFile: string;
    onSelectFile: (file: string) => void;
    unsavedChanges: string | null;
    onContentChange: (content: string) => void;
    onRenameFile: (oldName: string, newName: string) => void;
    onDeleteFile: (fileName: string) => void;
}

const EditorWorkspace: React.FC<EditorWorkspaceProps> = ({ files, selectedFile, onSelectFile, unsavedChanges, onContentChange, onRenameFile, onDeleteFile }) => {
    const [mode, setMode] = useState<'preview' | 'code'>('code');
    const fileContent = unsavedChanges ?? files[selectedFile] ?? '';

    const getTabClass = (tabMode: 'preview' | 'code') => {
        return `px-4 py-1.5 text-xs font-bold rounded-md transition-colors ${mode === tabMode ? 'bg-cyan-700 text-white' : 'text-cyan-300 hover:bg-white/10'}`;
    };

    return (
        <div className="h-full flex flex-col bg-black/30 border border-cyan-800 rounded-lg overflow-hidden">
            <header className="flex-shrink-0 p-2 border-b border-cyan-800/50 flex items-center justify-between bg-black/40">
                <div className="flex items-center space-x-1 bg-black/30 p-1 rounded-lg">
                    <button onClick={() => setMode('code')} className={getTabClass('code')}>Code</button>
                    <button onClick={() => setMode('preview')} className={getTabClass('preview')}>Preview</button>
                </div>
                <div className="text-xs text-cyan-600 font-mono px-2">
                    {selectedFile} {unsavedChanges !== null && <span className="text-yellow-500">•</span>}
                </div>
            </header>
            <div className="flex-grow grid grid-cols-[200px_1fr] min-h-0">
                <FileExplorer 
                    files={Object.keys(files)} 
                    selectedFile={selectedFile} 
                    onSelect={onSelectFile} 
                    onRename={onRenameFile}
                    onDelete={onDeleteFile}
                />
                <div className="border-l border-cyan-800/50 relative bg-black/20">
                    {mode === 'code' ? (
                        <Editor content={fileContent} fileName={selectedFile} onContentChange={onContentChange} />
                    ) : (
                        <Preview />
                    )}
                </div>
            </div>
        </div>
    );
};

export default EditorWorkspace;
