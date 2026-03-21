
import React, { useRef, useEffect } from 'react';
import Editor, { loader, OnMount } from '@monaco-editor/react';

// Define the CDN path
const MONACO_VS_PATH = 'https://cdn.jsdelivr.net/npm/monaco-editor@0.44.0/min/vs';

const QLANG_KEYWORDS = [
    'QREG', 'CREG', 'ALLOC', 'EXECUTE', 'IF', 'THEN', 'ELSE', 'LOOP', 'EOF', 'MEASURE', 'BARRIER', 'RESET', 'FOR', 'FROM', 'TO', 'IMPORT', 'MODULE', 'FUNCTION', 'RETURN', 'START', 'END', 'JUMP-IF', 'END_JOB'
];

const QLANG_OPERATORS = [
    'OP::H', 'OP::X', 'OP::Y', 'OP::Z', 'OP::S', 'OP::T', 'OP::CNOT', 'OP::C4Z', 'OP::MEASURE', 'OP::RETURN_RESULT', 'OP::I', 'OP::CZ', 'OP::RZ', 'OP::RY', 'OP::SWAP', 'OP::TOFFOLI', 'OP::QAE', 'OP::QUANTUM_WALK', 'OP::INIT_STATE', 'OP::DEPLOY', 'OP::ROUTE'
];

// --- Fix: Worker Configuration for CDN (Global Scope) ---
if (typeof window !== 'undefined' && !(window as any).MonacoEnvironment) {
    (window as any).MonacoEnvironment = {
        getWorkerUrl: function (moduleId: string, label: string) {
            if (label === 'json') {
                return `data:text/javascript;charset=utf-8,${encodeURIComponent(`
                    self.MonacoEnvironment = { baseUrl: '${MONACO_VS_PATH}' };
                    importScripts('${MONACO_VS_PATH}/language/json/json.worker.js');
                `)}`;
            }
            if (label === 'css' || label === 'scss' || label === 'less') {
                return `data:text/javascript;charset=utf-8,${encodeURIComponent(`
                    self.MonacoEnvironment = { baseUrl: '${MONACO_VS_PATH}' };
                    importScripts('${MONACO_VS_PATH}/language/css/css.worker.js');
                `)}`;
            }
            if (label === 'html' || label === 'handlebars' || label === 'razor') {
                return `data:text/javascript;charset=utf-8,${encodeURIComponent(`
                    self.MonacoEnvironment = { baseUrl: '${MONACO_VS_PATH}' };
                    importScripts('${MONACO_VS_PATH}/language/html/html.worker.js');
                `)}`;
            }
            if (label === 'typescript' || label === 'javascript') {
                return `data:text/javascript;charset=utf-8,${encodeURIComponent(`
                    self.MonacoEnvironment = { baseUrl: '${MONACO_VS_PATH}' };
                    importScripts('${MONACO_VS_PATH}/language/typescript/ts.worker.js');
                `)}`;
            }
            // Fallback for editor worker
            return `data:text/javascript;charset=utf-8,${encodeURIComponent(`
                self.MonacoEnvironment = { baseUrl: '${MONACO_VS_PATH}' };
                importScripts('${MONACO_VS_PATH}/editor/editor.worker.js');
            `)}`;
        }
    };
}

interface MonacoEditorWrapperProps {
    code: string;
    onChange: (value: string | undefined) => void;
    language?: string;
    readOnly?: boolean;
    theme?: string;
    className?: string;
    onEditorMount?: (editor: any, monaco: any) => void;
}

const MonacoEditorWrapper: React.FC<MonacoEditorWrapperProps> = ({ 
    code, 
    onChange, 
    language = 'javascript', 
    readOnly = false, 
    theme = 'qcos-dark', 
    className = "",
    onEditorMount
}) => {
    const editorRef = useRef<any>(null);
    const monacoRef = useRef<any>(null);
    const subscriptionRef = useRef<any>(null);
    
    // Track mounting status to prevent "Canceled" errors if unmounted during load
    const isMountedRef = useRef(false);

    useEffect(() => {
        isMountedRef.current = true;
        return () => {
            isMountedRef.current = false;
            // Clean up subscription
            if (subscriptionRef.current) {
                try {
                    subscriptionRef.current.dispose();
                } catch (e) { /* ignore */ }
                subscriptionRef.current = null;
            }
            // Dispose editor to prevent leaks and "Canceled" promises
            if (editorRef.current) {
                try {
                    // Check if model is already disposed or editor is disposed
                    const model = editorRef.current.getModel();
                    if (model && !model.isDisposed()) {
                         // Sometimes disposing the model explicitly helps
                         model.dispose(); 
                    }
                    editorRef.current.dispose();
                } catch (e) {
                    // Ignore disposal errors which often cause the "Canceled" rejection
                }
                editorRef.current = null;
            }
        };
    }, []);

    // --- Q-Lang Linter Logic ---
    const validateQLang = (model: any, monaco: any) => {
        // Critical Check: If model is disposed, abort immediately to prevent "Canceled" error
        if (!isMountedRef.current || !model || model.isDisposed()) return;

        const value = model.getValue();
        const markers = [];
        const lines = value.split('\n');
        const braceStack: { line: number, col: number }[] = [];

        for (let i = 0; i < lines.length; i++) {
            const originalLine = lines[i];
            const line = originalLine.trim();
            if (!line || line.startsWith('//') || line.startsWith('#')) continue;

            // Rule 1: Semicolon check
            if (!line.endsWith(';') && !line.endsWith('{') && !line.endsWith('}')) {
                const keywordsWithoutSemi = ['IF', 'ELSE', 'LOOP', 'FOR', 'WHILE', 'MODULE', 'FUNCTION'];
                const firstWord = line.split(/[ \t(]/)[0];
                
                if (!keywordsWithoutSemi.includes(firstWord)) {
                     markers.push({
                        severity: monaco.MarkerSeverity.Warning,
                        startLineNumber: i + 1,
                        startColumn: originalLine.length + 1,
                        endLineNumber: i + 1,
                        endColumn: originalLine.length + 1,
                        message: "Missing semicolon",
                    });
                }
            }

            // Rule 2: Operator validation & Malformed OP syntax
            // We look for anything that starts with OP:: and check if it's in our valid list
            const opMatches = originalLine.matchAll(/OP::[^ \t,;{}()]+/g);
            for (const match of opMatches) {
                const op = match[0];
                if (!QLANG_OPERATORS.includes(op)) {
                    markers.push({
                        severity: monaco.MarkerSeverity.Error,
                        startLineNumber: i + 1,
                        startColumn: match.index! + 1,
                        endLineNumber: i + 1,
                        endColumn: match.index! + op.length + 1,
                        message: `Invalid or malformed operator: ${op}`,
                    });
                }
            }

            // Rule 3: Unknown keyword validation (first word)
            const firstWordMatch = line.match(/^[A-Z_]+/);
            if (firstWordMatch) {
                const firstWord = firstWordMatch[0];
                if (!QLANG_KEYWORDS.includes(firstWord) && !QLANG_OPERATORS.includes(firstWord) && !firstWord.startsWith('OP::')) {
                    markers.push({
                        severity: monaco.MarkerSeverity.Error,
                        startLineNumber: i + 1,
                        startColumn: originalLine.indexOf(firstWord) + 1,
                        endLineNumber: i + 1,
                        endColumn: originalLine.indexOf(firstWord) + firstWord.length + 1,
                        message: `Unknown keyword or instruction: ${firstWord}`,
                    });
                }
            }

            // Rule 4: Brace tracking for syntax errors
            for (let charIndex = 0; charIndex < originalLine.length; charIndex++) {
                if (originalLine[charIndex] === '{') {
                    braceStack.push({ line: i + 1, col: charIndex + 1 });
                } else if (originalLine[charIndex] === '}') {
                    if (braceStack.length === 0) {
                        markers.push({
                            severity: monaco.MarkerSeverity.Error,
                            startLineNumber: i + 1,
                            startColumn: charIndex + 1,
                            endLineNumber: i + 1,
                            endColumn: charIndex + 2,
                            message: "Unexpected closing brace '}'",
                        });
                    } else {
                        braceStack.pop();
                    }
                }
            }
        }

        // Rule 5: Unclosed braces
        while (braceStack.length > 0) {
            const unclosed = braceStack.pop()!;
            markers.push({
                severity: monaco.MarkerSeverity.Error,
                startLineNumber: unclosed.line,
                startColumn: unclosed.col,
                endLineNumber: unclosed.line,
                endColumn: unclosed.col + 1,
                message: "Unclosed brace '{'",
            });
        }
        
        // Double check before setting markers
        if (isMountedRef.current && !model.isDisposed()) {
            monaco.editor.setModelMarkers(model, 'owner', markers);
        }
    };

    const handleEditorDidMount: OnMount = (editor, monaco) => {
        // Safety check if component unmounted during async loading of Monaco
        if (!isMountedRef.current) {
            editor.dispose();
            return;
        }

        editorRef.current = editor;
        monacoRef.current = monaco;

        // --- Define Custom Holographic Theme ---
        monaco.editor.defineTheme('qcos-dark', {
            base: 'vs-dark',
            inherit: true,
            rules: [
                { token: 'keyword', foreground: 'A78BFA', fontStyle: 'bold' }, // Purple-400
                { token: 'type.identifier', foreground: '22D3EE', fontStyle: 'bold' }, // Cyan-400 (Operators)
                { token: 'identifier', foreground: 'E2E8F0' }, // Slate-200
                { token: 'number', foreground: '4ADE80' }, // Green-400
                { token: 'comment', foreground: '64748B', fontStyle: 'italic' }, // Slate-500
                { token: 'string', foreground: 'FACC15' }, // Yellow-400
                { token: 'operator', foreground: 'F472B6' }, // Pink-400
                { token: 'delimiter', foreground: '94A3B8' }, // Slate-400
                { token: 'annotation', foreground: 'F87171' }, // Red-400 (Directives like #PRAGMA)
            ],
            colors: {
                'editor.background': '#00000000', // Transparent to let glass UI show
                'editor.lineHighlightBackground': '#22d3ee10', // Subtle Cyan highlight
                'editorCursor.foreground': '#22d3ee',
                'editor.selectionBackground': '#22d3ee30',
                'editorLineNumber.foreground': '#475569',
                'editorLineNumber.activeForeground': '#22d3ee',
            }
        });

        // Register Q-Lang if not exists
        if (!monaco.languages.getLanguages().some((l: any) => l.id === 'q-lang')) {
            monaco.languages.register({ id: 'q-lang' });
            
            // Syntax Highlighting Configuration
            monaco.languages.setMonarchTokensProvider('q-lang', {
                keywords: QLANG_KEYWORDS,
                operators: QLANG_OPERATORS,
                symbols:  /[=><!~?:&|+\-*/^%]+/,
                escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,
                
                tokenizer: {
                    root: [
                        [/^\s*#\w+/, 'annotation'],
                        [/OP::[a-zA-Z0-9_]+/, 'type.identifier'],
                        [/[a-zA-Z_]\w*/, {
                            cases: {
                                '@keywords': 'keyword',
                                '@default': 'identifier'
                            }
                        }],
                        { include: '@whitespace' },
                        [/[{}()[\]]/, '@brackets'],
                        [/[<>](?!@symbols)/, '@brackets'],
                        [/@symbols/, { cases: { '@operators': 'operator', '@default': '' } }],
                        [/\d*\.\d+([eE][-+]?\d+)?/, 'number.float'],
                        [/0[xX][0-9a-fA-F]+/, 'number.hex'],
                        [/\d+/, 'number'],
                        [/"([^"\\]|\\.)*$/, 'string.invalid'],
                        [/"/,  { token: 'string.quote', bracket: '@open', next: '@string' }],
                    ],
                    string: [
                        [/[^\\"]+/,  'string'],
                        [/@escapes/, 'string.escape'],
                        [/\\./,      'string.escape.invalid'],
                        [/"/,        { token: 'string.quote', bracket: '@close', next: '@pop' }]
                    ],
                    whitespace: [
                        [/[ \t\r\n]+/, 'white'],
                        [/\/\/.*$/,    'comment'],
                        [/\/\*/,       'comment', '@comment']
                    ],
                    comment: [
                        [/[^/*]+/, 'comment'],
                        [/\*\//,    'comment', '@pop'],
                        [/[/*]/,   'comment']
                    ],
                }
            });

            // Auto-Completion
            monaco.languages.registerCompletionItemProvider('q-lang', {
                provideCompletionItems: (model: any, position: any) => {
                     const suggestions = [
                        { label: 'QREG', kind: monaco.languages.CompletionItemKind.Keyword, insertText: 'QREG ${1:name}[${2:size}];', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, documentation: 'Allocate Quantum Register' },
                        { label: 'CREG', kind: monaco.languages.CompletionItemKind.Keyword, insertText: 'CREG ${1:name}[${2:size}];', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, documentation: 'Allocate Classical Register' },
                        { label: 'OP::H', kind: monaco.languages.CompletionItemKind.Function, insertText: 'OP::H ${1:qubit};', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, documentation: 'Hadamard Gate' },
                        { label: 'OP::CNOT', kind: monaco.languages.CompletionItemKind.Function, insertText: 'OP::CNOT ${1:control}, ${2:target};', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, documentation: 'Controlled NOT Gate' },
                        { label: 'MEASURE', kind: monaco.languages.CompletionItemKind.Keyword, insertText: 'MEASURE ${1:qreg} -> ${2:creg};', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, documentation: 'Measurement Operation' },
                        { label: 'LOOP', kind: monaco.languages.CompletionItemKind.Snippet, insertText: 'LOOP ${1:count} {\n\t$0\n}', insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet, documentation: 'Loop Block' },
                    ];
                    return { suggestions };
                }
            });

             monaco.languages.registerDocumentFormattingEditProvider('q-lang', {
                provideDocumentFormattingEdits: (model: any, options: any, token: any) => {
                    const text = model.getValue();
                    let indentLevel = 0;
                    const lines = text.split('\n');
                    const formattedLines = lines.map((line: string) => {
                        const trimmed = line.trim();
                        if (trimmed.startsWith('}') || trimmed.startsWith('LOOP END')) indentLevel = Math.max(0, indentLevel - 1);
                        const indent = '  '.repeat(indentLevel);
                        const formatted = indent + trimmed;
                        if (trimmed.endsWith('{') || trimmed.includes('LOOP START')) indentLevel++;
                        return formatted;
                    });
                    return [{ range: model.getFullModelRange(), text: formattedLines.join('\n') }];
                }
            });
        }

        // Set the custom theme
        monaco.editor.setTheme('qcos-dark');

        // Setup real-time linting for Q-Lang with cleanup
        if (language === 'q-lang') {
            const model = editor.getModel();
            if (model) {
                if (subscriptionRef.current) {
                    try { subscriptionRef.current.dispose(); } catch(e){
                        console.error("Failed to dispose subscription", e);
                    }
                }
                subscriptionRef.current = model.onDidChangeContent(() => {
                    // Guard against disposed model within the listener callback
                    if (isMountedRef.current && !model.isDisposed()) {
                        validateQLang(model, monaco);
                    }
                });
                validateQLang(model, monaco);
            }
        }

        if (onEditorMount) {
            onEditorMount(editor, monaco);
        }
    };

    return (
        <div className={`h-full w-full ${className}`}>
            <Editor
                height="100%"
                defaultLanguage={language}
                language={language}
                value={code}
                onChange={onChange}
                theme={theme}
                onMount={handleEditorDidMount}
                options={{
                    readOnly,
                    minimap: { enabled: false },
                    fontSize: 12,
                    fontFamily: "'Share Tech Mono', monospace",
                    lineNumbers: 'on',
                    renderLineHighlight: 'none',
                    padding: { top: 16, bottom: 16 },
                    scrollBeyondLastLine: false,
                    automaticLayout: true,
                    tabSize: 2,
                    formatOnType: true,
                    formatOnPaste: true,
                    suggest: { preview: true },
                    scrollbar: {
                        useShadows: false,
                        vertical: 'visible',
                        horizontal: 'visible',
                        verticalScrollbarSize: 8,
                        horizontalScrollbarSize: 8
                    }
                }}
            />
        </div>
    );
};

export default MonacoEditorWrapper;
