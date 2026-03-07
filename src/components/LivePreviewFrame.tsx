
import React, { useRef, useState, useEffect } from 'react';
import { LoaderIcon, AlertTriangleIcon } from './Icons';

interface LivePreviewFrameProps {
    code: string;
    files: { [key: string]: string };
}

const LivePreviewFrame: React.FC<LivePreviewFrameProps> = ({ code, files }) => {
    const iframeRef = useRef<HTMLIFrameElement>(null);
    const [error, setError] = useState<string | null>(null);
    const [status, setStatus] = useState<'idle' | 'bundling' | 'transpiling' | 'ready'>('idle');

    useEffect(() => {
        const updateIframe = () => {
            if (!iframeRef.current) return;
            
            setStatus('bundling');
            setError(null);

            // 1. Detect Component Name
            let componentName = 'App';
            const exportDefaultMatch = code.match(/export\s+default\s+(?:function|class)?\s*(\w+)/);
            const functionMatch = code.match(/function\s+(\w+)/);
            const classMatch = code.match(/class\s+(\w+)/);
            const constMatch = code.match(/const\s+(\w+)\s*=/);

            if (exportDefaultMatch) {
                componentName = exportDefaultMatch[1];
            } else if (functionMatch) {
                componentName = functionMatch[1];
            } else if (classMatch) {
                componentName = classMatch[1];
            } else if (constMatch) {
                componentName = constMatch[1];
            }

            setStatus('transpiling');

            // 2. Prepare all files for injection
            const filesJson = JSON.stringify(files);

            // 3. Construct HTML Payload
            const htmlContent = `
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>ChipsDev Live Preview</title>
                    <script>
                        // Suppress Tailwind CDN production warning in preview
                        const originalWarn = console.warn;
                        console.warn = (...args) => {
                            if (args[0] && typeof args[0] === 'string' && args[0].includes('cdn.tailwindcss.com')) return;
                            originalWarn(...args);
                        };
                    </script>
                    <script src="https://cdn.tailwindcss.com"></script>
                    <script src="https://unpkg.com/react@18/umd/react.development.js" crossorigin></script>
                    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js" crossorigin></script>
                    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
                    <style>
                        body { background-color: transparent; color: white; margin: 0; font-family: 'Inter', sans-serif; overflow: auto; }
                        #root { height: 100vh; width: 100vw; overflow: auto; }
                        .error-container { color: #ef4444; background: rgba(127, 29, 29, 0.2); padding: 1rem; border-radius: 0.5rem; border: 1px solid #991b1b; font-family: monospace; white-space: pre-wrap; font-size: 12px; margin: 20px; }
                        ::-webkit-scrollbar { width: 6px; height: 6px; }
                        ::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.1); }
                        ::-webkit-scrollbar-thumb { background: rgba(6, 182, 212, 0.3); border-radius: 3px; }
                        ::-webkit-scrollbar-thumb:hover { background: rgba(6, 182, 212, 0.5); }
                    </style>
                </head>
                <body>
                    <div id="root"></div>
                    <script>
                        window.onerror = function(message, source, lineno, colno, error) {
                            if (source && (source.includes('installHook') || source.includes('extension'))) return;
                            if (typeof message === 'string' && (message.includes('installHook') || message.includes('Qj') || message.includes('overrideMethod'))) return;

                            const rootEl = document.getElementById('root');
                            if (rootEl && !rootEl.hasChildNodes()) {
                                rootEl.innerHTML = '<div class="error-container"><h3>Runtime Error</h3><div>' + message + '</div></div>';
                            }
                            window.parent.postMessage({ type: 'PREVIEW_ERROR', message: message }, '*');
                        };

                        const projectFiles = ${filesJson};
                        const mainEntryCode = ${JSON.stringify(code)};
                        const mainComponentName = "${componentName}";

                        function processCode(codeStr) {
                            return codeStr
                                .replace(/import\\s+[\\s\\S]*?from\\s+['"]lucide-react['"];?/g, '') // Remove lucide imports specifically
                                .replace(/import\\s+[\\s\\S]*?from\\s+['"].*?['"];?/g, '')
                                .replace(/import\\s+['"].*?['"];?/g, '')
                                // Attach exports to window to ensure global availability across evals
                                .replace(/export\\s+default\\s+(function|class)\\s+(\\w+)/g, 'window.$2 = $1 $2')
                                .replace(/export\\s+default\\s+(\\w+);?/g, 'window.$1 = $1;')
                                .replace(/export\\s+class\\s+(\\w+)/g, 'window.$1 = class $1') // Handle named class exports
                                .replace(/export\\s+function\\s+(\\w+)/g, 'window.$1 = function $1')
                                .replace(/export\\s+(const|let|var)\\s+(\\w+)\\s*=/g, 'window.$2 =');
                        }

                        function run() {
                            if (!window.React || !window.ReactDOM || !window.Babel) {
                                setTimeout(run, 50);
                                return;
                            }

                            try {
                                const { useState, useEffect, useRef, useMemo, useCallback, useReducer, useContext, createContext } = React;
                                
                                // Mock Icons - Expanded Set for AI Generation Compatibility
                                const MockIcon = (props) => React.createElement('svg', { ...props, width: 24, height: 24, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" }, React.createElement('rect', { x: 2, y: 2, width: 20, height: 20 }));
                                
                                const ArrowIcon = (props) => React.createElement('svg', { ...props, width: 24, height: 24, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2 }, React.createElement('path', { d: "M5 12h14M12 5l7 7-7 7" }));
                                
                                const iconMap = {
                                    // Internal App Names
                                    'CpuChipIcon': MockIcon, 'ActivityIcon': MockIcon, 'GlobeIcon': MockIcon, 'BoxIcon': MockIcon, 
                                    'SearchIcon': MockIcon, 'StarIcon': MockIcon, 'ShieldCheckIcon': MockIcon, 'UsersIcon': MockIcon, 
                                    'BriefcaseIcon': MockIcon, 'LayoutGridIcon': MockIcon, 'AtomIcon': MockIcon, 'ServerCogIcon': MockIcon, 
                                    'BrainCircuitIcon': MockIcon, 'RocketLaunchIcon': MockIcon, 'CodeBracketIcon': MockIcon, 
                                    'FileCodeIcon': MockIcon, 'TerminalIcon': MockIcon, 'LinkIcon': MockIcon, 'LockIcon': MockIcon,
                                    'CheckCircle2Icon': MockIcon, 'AlertTriangleIcon': MockIcon, 'LoaderIcon': MockIcon, 'StopIcon': MockIcon, 
                                    'PlayIcon': MockIcon, 'FastForwardIcon': MockIcon, 'ArrowRightIcon': ArrowIcon, 'TrashIcon': MockIcon, 
                                    'PlusIcon': MockIcon, 'XIcon': MockIcon, 'SparklesIcon': MockIcon, 'EyeIcon': MockIcon, 'HeartIcon': MockIcon, 
                                    'PhoneIcon': MockIcon, 'UploadCloudIcon': MockIcon, 'DownloadCloudIcon': MockIcon, 'ClockIcon': MockIcon, 
                                    'MessageSquareIcon': MockIcon, 'ChevronRightIcon': ArrowIcon, 'ChevronLeftIcon': ArrowIcon, 
                                    'ToggleLeftIcon': MockIcon, 'ToggleRightIcon': MockIcon, 'RefreshCwIcon': MockIcon, 
                                    'CurrencyDollarIcon': MockIcon, 'ClipboardIcon': MockIcon, 'MapPinIcon': MockIcon, 'Share2Icon': MockIcon,
                                    'WifiIcon': MockIcon, 'WifiOffIcon': MockIcon, 'ZapIcon': MockIcon,
                                    // Standard Lucide Names
                                    'ArrowUp': ArrowIcon, 'ArrowDown': ArrowIcon, 'ArrowLeft': ArrowIcon, 'ArrowRight': ArrowIcon,
                                    'ChevronUp': ArrowIcon, 'ChevronDown': ArrowIcon, 'ChevronLeft': ArrowIcon, 'ChevronRight': ArrowIcon,
                                    'Play': MockIcon, 'Pause': MockIcon, 'Settings': MockIcon, 'User': MockIcon, 'Home': MockIcon,
                                    'Menu': MockIcon, 'X': MockIcon, 'Check': MockIcon, 'Plus': MockIcon, 'Minus': MockIcon,
                                    'Info': MockIcon, 'AlertCircle': MockIcon, 'AlertTriangle': MockIcon, 'HelpCircle': MockIcon,
                                    'Cpu': MockIcon, 'Database': MockIcon, 'Server': MockIcon, 'Globe': MockIcon, 'Shield': MockIcon, 'Lock': MockIcon, 'Unlock': MockIcon
                                };

                                // Inject icons into window scope
                                Object.keys(iconMap).forEach(key => { window[key] = iconMap[key]; });

                                // 1. Evaluate Dependencies (All other TSX/JS files)
                                Object.entries(projectFiles).forEach(([filename, content]) => {
                                    if (content === mainEntryCode) return; // Skip main entry for now
                                    if (filename.endsWith('.tsx') || filename.endsWith('.jsx') || filename.endsWith('.js') || filename.endsWith('.ts')) {
                                        try {
                                            const processed = processCode(content);
                                            const compiled = Babel.transform(processed, {
                                                presets: [['env', { modules: false }], 'react', 'typescript'],
                                                filename: filename
                                            }).code;
                                            eval(compiled);
                                        } catch (e) {
                                            console.warn("Dependency eval error (" + filename + "):", e);
                                        }
                                    }
                                });

                                // 2. Evaluate Main Entry
                                const mainProcessed = processCode(mainEntryCode);
                                const mainCompiled = Babel.transform(mainProcessed, {
                                    presets: [['env', { modules: false }], 'react', 'typescript'],
                                    filename: 'App.tsx'
                                }).code;
                                eval(mainCompiled);

                                // 3. Mount
                                const TargetComponent = window[mainComponentName] || eval(mainComponentName);
                                if (!TargetComponent) throw new Error("Component " + mainComponentName + " not found. Ensure your code exports a component.");

                                const root = ReactDOM.createRoot(document.getElementById('root'));
                                
                                class ErrorBoundary extends React.Component {
                                    constructor(props) { super(props); this.state = { hasError: false, error: null }; }
                                    static getDerivedStateFromError(error) { return { hasError: true, error }; }
                                    componentDidCatch(error, errorInfo) { 
                                        if (error && error.stack && error.stack.includes('installHook')) return;
                                        window.parent.postMessage({ type: 'PREVIEW_ERROR', message: error.message }, '*');
                                    }
                                    render() {
                                        if (this.state.hasError) {
                                            return React.createElement('div', { className: 'error-container' }, 
                                                React.createElement('h3', null, 'Runtime Error'),
                                                React.createElement('div', null, this.state.error && this.state.error.toString())
                                            );
                                        }
                                        return this.props.children;
                                    }
                                }

                                root.render(React.createElement(ErrorBoundary, null, React.createElement(TargetComponent)));
                                window.parent.postMessage({ type: 'PREVIEW_READY' }, '*');

                            } catch (err) {
                                const rootEl = document.getElementById('root');
                                if (rootEl) {
                                     rootEl.innerHTML = '<div class="error-container"><h3>Build Error</h3><div>' + err.message + '</div></div>';
                                }
                                window.parent.postMessage({ type: 'PREVIEW_ERROR', message: err.message }, '*');
                            }
                        }

                        run();
                    </script>
                </body>
                </html>
            `;

            const blob = new Blob([htmlContent], { type: 'text/html' });
            const url = URL.createObjectURL(blob);
            
            iframeRef.current.src = url;
        };

        const timer = setTimeout(updateIframe, 600);
        return () => clearTimeout(timer);
    }, [code, files]);

    useEffect(() => {
        const handler = (e: MessageEvent) => {
            if (e.data?.type === 'PREVIEW_ERROR') {
                setStatus('idle');
                setError(e.data.message);
            } else if (e.data?.type === 'PREVIEW_READY') {
                setStatus('ready');
                setError(null);
            }
        };
        window.addEventListener('message', handler);
        return () => window.removeEventListener('message', handler);
    }, []);

    return (
        <div className="w-full h-full relative bg-slate-950 flex flex-col overflow-hidden rounded-lg shadow-inner">
            {(status === 'bundling' || status === 'transpiling') && (
                <div className="absolute top-2 right-2 z-20 flex items-center gap-2 bg-black/80 px-3 py-1.5 rounded-full text-xs text-cyan-400 border border-cyan-900 backdrop-blur-md shadow-lg animate-fade-in">
                    <LoaderIcon className="w-3 h-3 animate-spin" />
                    <span>{status === 'bundling' ? 'Bundling Assets...' : 'Transpiling...'}</span>
                </div>
            )}
            
            <iframe 
                ref={iframeRef}
                title="Live Preview"
                className="flex-grow w-full h-full border-none bg-transparent"
                sandbox="allow-scripts allow-same-origin allow-forms allow-modals" 
            />
            
            {error && (
                <div className="absolute bottom-0 left-0 right-0 bg-red-950/90 text-red-200 text-xs p-2 font-mono border-t border-red-500 backdrop-blur-sm z-30 max-h-32 overflow-y-auto">
                    <div className="flex items-center gap-2 font-bold mb-1">
                        <AlertTriangleIcon className="w-4 h-4" /> Runtime Exception
                    </div>
                    <div className="whitespace-pre-wrap break-words">{error}</div>
                </div>
            )}
        </div>
    );
};

export default LivePreviewFrame;
