
import React, { useState, useEffect, useRef, useCallback, useMemo, Suspense } from 'react';
import { 
    CodeBracketIcon, UsersIcon, LightBulbIcon, RocketLaunchIcon, 
    Share2Icon, GitBranchIcon, CheckCircle2Icon, PlayIcon, 
    BriefcaseIcon, ChatBubbleLeftRightIcon, PuzzlePieceIcon,
    CommandLineIcon, FileCodeIcon, PlusIcon, XIcon, LoaderIcon,
    ServerCogIcon, BoxIcon, ArrowRightIcon, TrashIcon, DocumentArrowUpIcon,
    SparklesIcon, EyeIcon, LayoutGridIcon, HeartIcon, PhoneIcon,
    SourceControlIcon, UploadCloudIcon, DownloadCloudIcon, ClockIcon,
    MessageSquareIcon, ChevronRightIcon, AtomIcon, ToggleLeftIcon, ToggleRightIcon,
    ShieldCheckIcon, RefreshCwIcon, CurrencyDollarIcon, GlobeIcon, LinkIcon, ClipboardIcon,
    CpuChipIcon, ActivityIcon, LockIcon, AlertTriangleIcon, StopIcon, FastForwardIcon, BrainCircuitIcon,
    TerminalIcon, TimelineIcon, GalaxyIcon, SaveIcon, GripIcon, ArrowTopRightOnSquareIcon,
    SettingsIcon, StarIcon, PencilSquareIcon, BugAntIcon, ChevronDownIcon, ChevronUpIcon,
    ArrowPathIcon, LayersIcon, NetworkIcon, CloudServerIcon
} from './Icons';
import { STANDARD_QLANG_TEMPLATES, QLangTemplate } from '../utils/agentUtils';
import { SystemHealth, UIStructure } from '../types';
import { useToast } from '../context/ToastContext';
import { useSimulation } from '../context/SimulationContext';
import { useAgentQ, FileSystemOps, ProjectOps } from '../hooks/useAgentQ';
import { GoogleGenAI, Type } from "@google/genai";
import { generateContentWithRetry } from '../utils/gemini';

// Local Imports
import SyntaxHighlighter from './SyntaxHighlighter';
import GlassPanel from './GlassPanel';
import UniverseSimulator from './UniverseSimulator';
import AgentQ from './AgentQ';
import HolographicPreviewRenderer from './HolographicPreviewRenderer';
import MonacoEditorWrapper from './MonacoEditorWrapper';
import LivePreviewFrame from './LivePreviewFrame';
import ChimeraCoreStatus from './ChimeraCoreStatus';

type DevTab = 'inspiration' | 'planning' | 'coding' | 'ops';
type BottomTab = 'console' | 'terminal';

// --- Types ---
type DeployTarget = 'store' | 'netlify' | 'sdk';

interface Task {
    id: string;
    title: string;
    status: 'To Do' | 'In Progress' | 'Done';
    assignee: string;
}

interface GitCommit {
    id: string;
    message: string;
    timestamp: number;
    author: string;
}

interface GitHistory {
    commits: GitCommit[];
    branch: string;
    syncStatus: 'synced' | 'ahead' | 'behind';
    pendingChanges: number;
}

interface Project {
    id: string;
    title: string;
    description: string;
    tasks: Task[];
    files: { [fileName: string]: string }; 
    env: 'Development' | 'Staging' | 'Production';
    version: string;
    uptime?: string;
    activeNodes?: number;
    lastEdited?: number;
    gitHistory?: GitHistory;
}

const classicalTemplates = [
    {
        id: 'blueprint_py_ml', title: 'Neural Net Trainer', icon: BrainCircuitIcon, desc: 'PyTorch-compatible training loop scaffold.',
        files: {
            'main.py': `import torch\nimport torch.nn as nn\n\nclass QuantumNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(10, 50)\n        self.fc2 = nn.Linear(50, 1)\n\n    def forward(self, x):\n        x = torch.relu(self.fc1(x))\n        return torch.sigmoid(self.fc2(x))\n\nmodel = QuantumNet()\nprint("Model initialized on NPU-1")`,
            'requirements.txt': 'torch\nnumpy\npandas'
        }
    },
    {
        id: 'blueprint_rust_sys', title: 'Rust System Tool', icon: ServerCogIcon, desc: 'High-performance CLI tool using Cargo.',
        files: {
            'main.rs': `fn main() {\n    println!("Initializing QCOS System Interface...");\n    let memory = 1024 * 64;\n    println!("Allocated {} quantum bytes", memory);\n}`,
            'Cargo.toml': `[package]\nname = "qcos_sys"\nversion = "0.1.0"\n\n[dependencies]\nserde = "1.0"`
        }
    },
    {
        id: 'blueprint_react_dash', title: 'React Dashboard', icon: LayoutGridIcon, desc: 'Modern dashboard with Tailwind support.',
        files: {
            'App.tsx': `import React from 'react';\n\nexport default function Dashboard() {\n  return (\n    <div className="p-4 bg-slate-900 text-white h-full">\n      <h1 className="text-2xl font-bold">QCOS Analytics</h1>\n      <div className="grid grid-cols-2 gap-4 mt-4">\n         <div className="p-4 bg-slate-800 rounded">Metric A</div>\n         <div className="p-4 bg-slate-800 rounded">Metric B</div>\n      </div>\n    </div>\n  );\n}`,
            'package.json': `{\n  "name": "qcos-dash",\n  "dependencies": {\n    "react": "^18.2.0"\n  }\n}`
        }
    }
];

const appTemplates = [
    {
        id: 'blueprint_prod', title: 'Quantum Productivity', icon: BriefcaseIcon, desc: 'Task management with quantum-optimized scheduling.',
        files: { 
            'App.tsx': `import React, { useState } from 'react';\n\nexport default function ProductivityApp() {\n  const [tasks, setTasks] = useState(3);\n  const [optimized, setOptimized] = useState(false);\n\n  const optimizeSchedule = () => {\n    setOptimized(!optimized);\n  };\n\n  return (\n    <div className="p-6 bg-slate-900 text-white h-full flex flex-col gap-4">\n      <h1 className="text-3xl font-bold mb-2 text-cyan-400">Quantum Tasks</h1>\n      <div className="p-4 bg-gray-800 rounded border border-gray-700">\n        <h2 className="text-xl mb-2">Pending Tasks: {tasks}</h2>\n        <p className="text-gray-400 text-sm mb-4">Optimization Status: {optimized ? "Active (QPU)" : "Standard"}</p>\n        <button onClick={optimizeSchedule} className="px-4 py-2 bg-purple-600 rounded text-white font-bold">Toggle Q-Optimization</button>\n      </div>\n    </div>\n  );\n}`,
            'scheduler.q': `// Quantum Scheduler\nQREG tasks[4];\nOP::H tasks;\nMEASURE tasks;`,
            'styles.css': `.task-card { background: #1a202c; }`
        }
    },
    {
        id: 'blueprint_bayq', title: 'Bayq-Lang Protocol', icon: GlobeIcon, desc: 'L1 Functional Extension using Reformed Baybayin Orthography.',
        files: {
            'main.bq': `// Bayq-Lang Source (L1-Q-Lang)\n// Architecture: CHIPS Network / QCOS\n// Uses Reformed Orthography for precision\n\nᜀᜎᜓᜃ᜔ Q_BUS[4];  // ALLOC QREG\nᜀᜎᜓᜃ᜔ C_OUT[4];  // ALLOC CREG\n\n// Functional Block: Entanglement\nFUNCTION ᜁᜈ᜔ᜆᜅ᜔ᜄᜎ᜔(q_target) {\n   ᜄᜏ OP::H q_target[0];       // EXECUTE Hadamard\n   ᜄᜏ OP::CNOT q_target[0], q_target[1]; // EXECUTE CNOT\n}\n\n// Main Execution\nᜁᜈ᜔ᜆᜅ᜔ᜄᜎ᜔(Q_BUS);\n\n// Measurement with Virama suppression\nᜐᜓᜃᜆ᜔ Q_BUS -> C_OUT; // MEASURE\n`,
            'README.md': `# Bayq-Lang Specification\nThis project uses the L1 functional extension layer for Q-Lang.\nIt leverages Baybayin characters for mnemonics to ground the paradigm in non-Western orthography.\n\nMappings:\n- ALLOC -> ᜀᜎᜓᜃ᜔ (Alok)\n- EXECUTE -> ᜄᜏ (Gawa)\n- MEASURE -> ᜐᜓᜃᜆ᜔ (Sukat)`
        }
    },
    {
        id: 'blueprint_l1_hybrid', title: 'L1-Q Hybrid Variational', icon: AtomIcon, desc: 'High-level L1-Q-Lang with classical control flow and QED directives.',
        files: {
            'vqe.l1': `// L1-Q-Lang: Hybrid VQE\n// Target: QCOS QHAL Agnostic Layer\n\n// High-Level Resource Definition (Implicit)\nUSE q = Qubit(4);\nUSE c = Classical(4);\n\n// Error Mitigation Directive\n#PRAGMA EM::ZERO_NOISE_EXTRAPOLATION\n#PRAGMA QED::DETECT_LEAKAGE\n\n// Hybrid Control Flow\nFUNCTION OPTIMIZE_LOOP(theta) {\n    // Dynamic Parameter Injection\n    OP::RY(theta) q[0];\n    OP::CNOT q[0], q[1];\n    \n    // Real-time Classical Branching\n    MEASURE q[0] -> c[0];\n    IF (c[0] == 1) {\n        OP::X q[1]; // Fast-feedback correction\n    }\n}\n\n// Compile to CHIPS Packet\nTARGET_SCOPE::GEO::REG;\n`,
            'App.tsx': `import React from 'react';\nexport default function HybridDash() {\n return <div className="p-4"><h1>Hybrid VQE Running</h1><p>QHAL Topology: Mapped</p></div>;\n}`
        }
    }
];

const allTemplates = [...classicalTemplates, ...appTemplates];

const initialProjects: Project[] = [
    {
        id: 'proj_1',
        title: 'Quantum Entanglement Visualizer',
        description: 'A tool to visualize Bell states in real-time.',
        env: 'Production',
        version: 'v1.0.4',
        uptime: '99.9%',
        activeNodes: 8,
        tasks: [
            { id: 't1', title: 'Optimize rendering loop', status: 'Done', assignee: 'Dev_Alpha' },
            { id: 't2', title: 'Add QKD support', status: 'In Progress', assignee: 'Crypto_Expert' },
            { id: 't3', title: 'Refactor Q-Lang Bridge', status: 'To Do', assignee: 'Agent Q' },
            { id: 't4', title: 'Update Manifest', status: 'To Do', assignee: 'System' }
        ],
        files: {
            'App.tsx': `import React, { useState } from 'react';\n\nexport default function App() {\n  const [entangled, setEntangled] = useState(false);\n  \n  const toggleEntanglement = () => {\n    setEntangled(!entangled);\n  };\n\n  return (\n    <div className="h-full flex flex-col items-center justify-center bg-gray-900 text-white p-6">\n      <div className="text-center mb-8">\n        <h1 className="text-3xl font-bold text-cyan-400 mb-4">Entanglement Visualizer</h1>\n        <div className="flex gap-8 justify-center items-center mb-8">\n            <div className={\`w-16 h-16 rounded-full border-4 flex items-center justify-center text-xl font-bold transition-all duration-500 \${entangled ? 'border-purple-500 bg-purple-900/50 shadow-[0_0_30px_purple] scale-110' : 'border-gray-600 bg-gray-800'}\`}>\n                {entangled ? 'Ψ' : '|0⟩'}\n            </div>\n            <div className={\`h-1 w-24 rounded transition-all duration-500 \${entangled ? 'bg-gradient-to-r from-purple-500 to-cyan-500 animate-pulse' : 'bg-gray-700'}\`}></div>\n            <div className={\`w-16 h-16 rounded-full border-4 flex items-center justify-center text-xl font-bold transition-all duration-500 \${entangled ? 'border-cyan-500 bg-cyan-900/50 shadow-[0_0_30px_cyan] scale-110' : 'border-gray-600 bg-gray-800'}\`}>\n                {entangled ? 'Ψ' : '|0⟩'}\n            </div>\n        </div>\n        <p className="text-sm text-gray-400 font-mono mb-6">\n          State: {entangled ? "|Φ⁺⟩ = (|00⟩ + |11⟩)/√2" : "|00⟩ (Separable)"}\n        </p>\n        <button \n            onClick={toggleEntanglement} \n            className={\`px-8 py-3 rounded-lg font-bold transition-all \${entangled ? 'bg-red-600 hover:bg-red-500' : 'bg-green-600 hover:bg-green-500'}\`}\n        >\n            {entangled ? "Collapse Wavefunction" : "Entangle Qubits"}\n        </button>\n      </div>\n    </div>\n  );\n}`,
            'main.q': `// Q-Lang: Entanglement Viz\nQREG q[2];\nCREG c[2];\n\n// Create Bell Pair\nOP::H q[0];\nOP::CNOT q[0], q[1];\n\n// Measurement\nMEASURE q[0] -> c[0];\nMEASURE q[1] -> c[1];`,
            'manifest.json': `{ "name": "Entanglement Viz", "description": "Visualizes quantum entanglement states.", "version": "1.0.4" }`
        },
        lastEdited: Date.now(),
        gitHistory: {
            branch: 'main',
            syncStatus: 'synced',
            pendingChanges: 0,
            commits: [
                { id: 'a1b2c3d', message: 'Initial commit', timestamp: Date.now() - 100000, author: 'Dev_Alpha' }
            ]
        }
    }
];

const extractStateFromCode = (code: string) => {
    const initialStateRegex = /const \[(\w+), set\w+\] = useState\((.*?)\);/g;
    let match;
    const newInitialState: { [key: string]: any } = {};
    while ((match = initialStateRegex.exec(code)) !== null) {
        const stateVar = match[1]; 
        const rawInitialValue = match[2];
        try { 
            newInitialState[stateVar] = JSON.parse(rawInitialValue.replace(/'/g, '"')); 
        } catch (e) { 
            newInitialState[stateVar] = rawInitialValue.replace(/^['"`]|['"`]$/g, ''); 
        }
    }
    return newInitialState;
};

const transpileCodeToStructure = (code: string): UIStructure | null => {
    try {
        const returnStart = code.indexOf('return (');
        if (returnStart === -1) {
             const simpleReturn = code.match(/return\s+<([\s\S]*)>;/);
             if (simpleReturn) return parseNode(simpleReturn[1]).structure as UIStructure;
             return null;
        }
        
        let balance = 1;
        let returnEnd = -1;
        for (let i = returnStart + 8; i < code.length; i++) {
            if (code[i] === '(') balance++;
            else if (code[i] === ')') balance--;
            
            if (balance === 0) {
                returnEnd = i;
                break;
            }
        }
        
        if (returnEnd === -1) return null;
        const jsxContent = code.substring(returnStart + 8, returnEnd);

        function parseProps(propsString: string) {
            const props: any = {};
            const propRegex = /(\w+)=(?:["']([^"']*)["']|\{([^}]*)\})/g;
            let pMatch;
            while ((pMatch = propRegex.exec(propsString)) !== null) {
                const key = pMatch[1];
                const strValue = pMatch[2]; 
                const exprValue = pMatch[3]; 
                
                if (key === 'className') {
                    props.className = strValue; 
                } else if (key.startsWith('on')) {
                    props[key] = exprValue; 
                } else if (exprValue) {
                    if (!exprValue.includes('?')) {
                        props[key] = `STATE:${exprValue}`;
                    }
                } else {
                    props[key] = strValue;
                }
            }
            return props;
        }

        function parseNode(xml: string): { structure: UIStructure | string, remainder: string } {
            xml = xml.trim();
            if (!xml) return { structure: '', remainder: '' };

            if (!xml.startsWith('<')) {
                const nextTag = xml.indexOf('<');
                if (nextTag === -1) return { structure: xml, remainder: '' };
                return { structure: xml.substring(0, nextTag), remainder: xml.substring(nextTag) };
            }

            const tagRegex = /^<([a-zA-Z0-9]+)((?:\s+\w+=(?:["'][^"']*["']|\{[^}]*\}))*)?\s*(\/?)>/;
            const tagMatch = xml.match(tagRegex);

            if (!tagMatch) return { structure: '', remainder: xml.substring(1) }; 

            const [fullMatch, tagName, propsStr, selfClosing] = tagMatch;
            const props = parseProps(propsStr || '');
            let remainder = xml.substring(fullMatch.length);
            const children: (UIStructure | string)[] = [];

            if (!selfClosing) {
                while (remainder && !remainder.startsWith(`</${tagName}>`)) {
                    if (remainder.startsWith('{')) {
                        const closeBrace = remainder.indexOf('}');
                        if (closeBrace > -1) {
                            const varContent = remainder.substring(1, closeBrace);
                            if (!varContent.includes('?') && !varContent.includes('&&')) {
                                children.push(`STATE:${varContent}`);
                            } else if (varContent.startsWith('`')) {
                                children.push(varContent.replace(/`/g, '').replace(/\${/g, '').replace(/}/g, ''));
                            } else {
                                children.push(`{${varContent}}`); 
                            }
                            remainder = remainder.substring(closeBrace + 1);
                            continue;
                        }
                    }

                    const childResult = parseNode(remainder);
                    if (childResult.structure) {
                        if (typeof childResult.structure === 'string' && !childResult.structure.trim()) {
                            // skip empty string
                        } else {
                            children.push(childResult.structure);
                        }
                    }
                    remainder = childResult.remainder;
                }
                const closingTagLen = `</${tagName}>`.length;
                if (remainder.startsWith(`</${tagName}>`)) {
                    remainder = remainder.substring(closingTagLen);
                }
            }

            return {
                structure: { type: 'div', component: tagName, props, children },
                remainder
            };
        }

        const result = parseNode(jsxContent);
        return typeof result.structure === 'string' ? null : result.structure;

    } catch (e) {
        console.error("Simple Transpiler Error", e);
        return null;
    }
};

const NavButton: React.FC<{ active: boolean; onClick: () => void; icon: any; label: string }> = ({ active, onClick, icon: Icon, label }) => (
    <button onClick={onClick} className={`flex-1 min-w-[90px] px-3 py-2 text-xs font-bold rounded transition-all flex items-center justify-center gap-2 ${active ? 'bg-cyan-700 text-white shadow-lg shadow-cyan-900/50' : 'text-cyan-500 hover:bg-white/5'}`}>
        <Icon className="w-4 h-4" /> {label}
    </button>
);

const Terminal: React.FC<{ onLog: (msg: string) => void }> = ({ onLog }) => {
    const [input, setInput] = useState('');
    const [history, setHistory] = useState<string[]>(['QCOS Neural-Shell v4.5.0', 'Type "help" for a list of available commands.']);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }, [history]);

    const handleCommand = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && input.trim()) {
            const cmd = input.trim().toLowerCase();
            const newHistory = [...history, `> ${input}`];
            
            if (cmd === 'help') {
                newHistory.push('  ls - List files', '  clear - Clear terminal');
            } else if (cmd === 'clear') {
                setHistory([]);
                setInput('');
                return;
            } else {
                newHistory.push('Command executed.');
            }

            setHistory(newHistory);
            setInput('');
        }
    };
    return (
        <div className="h-full flex flex-col font-mono text-[10px] overflow-hidden bg-black/40">
            <div ref={scrollRef} className="flex-grow overflow-y-auto p-2 space-y-1 custom-scrollbar">
                {history.map((line, i) => <div key={i} className="text-green-500 opacity-90">{line}</div>)}
            </div>
            <div className="flex items-center gap-2 p-2 bg-black/60 border-t border-cyan-900/30">
                <span className="text-cyan-500 font-bold">$</span>
                <input className="flex-grow bg-transparent border-none text-white outline-none" value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleCommand} placeholder="run command..." />
            </div>
        </div>
    );
};

const InspirationView: React.FC<{ templates: any[]; onFork: (template: any) => void }> = ({ templates, onFork }) => {
    return <div className="p-4 grid grid-cols-2 md:grid-cols-3 gap-4 overflow-y-auto h-full">
        {templates.map(tpl => (
            <div key={tpl.id} className="bg-black/30 border border-cyan-900/50 rounded-lg p-4 hover:bg-cyan-900/10 hover:border-cyan-500 transition-all group flex flex-col">
                <div className="flex items-start justify-between mb-2">
                    <div className="p-2 bg-cyan-950/40 rounded-full group-hover:scale-110 transition-transform">
                        <tpl.icon className="w-6 h-6 text-cyan-400" />
                    </div>
                </div>
                <h3 className="text-sm font-bold text-white mb-1">{tpl.title}</h3>
                <p className="text-xs text-gray-400 mb-3 flex-grow">{tpl.desc}</p>
                <button onClick={() => onFork(tpl)} className="mt-auto w-full py-2 bg-cyan-800/30 border border-cyan-700/50 rounded text-xs font-bold text-cyan-200 hover:bg-cyan-700/50 transition-colors flex items-center justify-center gap-2">
                    <GitBranchIcon className="w-3 h-3" /> Fork Template
                </button>
            </div>
        ))}
    </div>;
};

const PlanningView: React.FC<{ 
    project: Project; 
    onUpdate: (id: string, data: Partial<Project>) => void;
    projects: Project[];
    onSelectProject: (id: string) => void;
    onDeleteProject: (id: string) => void;
    onUseBlueprint: (bp: any) => void;
    onEditCode: () => void;
}> = ({ project, onUpdate, projects, onSelectProject, onDeleteProject, onUseBlueprint, onEditCode }) => {
    const [now] = useState(() => Date.now());
    return <div className="p-4 flex flex-col gap-4 h-full overflow-y-auto">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
             <div className="md:col-span-1 bg-black/30 border border-cyan-900/50 rounded-lg p-3">
                 <h4 className="text-xs font-bold text-cyan-400 uppercase tracking-widest mb-2 border-b border-cyan-900/50 pb-1">Projects</h4>
                 <div className="space-y-1">
                     {projects.map(p => (
                         <div key={p.id} className={`flex flex-col p-2 rounded cursor-pointer border ${p.id === project.id ? 'bg-cyan-900/40 border-cyan-500 text-white' : 'border-transparent hover:bg-white/5 text-gray-400'}`} onClick={() => onSelectProject(p.id)}>
                             <div className="flex justify-between items-center">
                                <span className="text-xs font-bold truncate">{p.title}</span>
                                <button onClick={(e) => { e.stopPropagation(); onDeleteProject(p.id); }} className="text-gray-500 hover:text-red-400"><TrashIcon className="w-3 h-3" /></button>
                             </div>
                             <span className="text-[9px] text-gray-500 mt-1">Last edited: {new Date(p.lastEdited || now).toLocaleDateString()}</span>
                         </div>
                     ))}
                 </div>
             </div>
             
             <div className="md:col-span-2 space-y-4">
                 <div className="bg-black/30 border border-cyan-900/50 rounded-lg p-4">
                     <h4 className="text-sm font-bold text-white mb-2">Project Details</h4>
                     <input className="w-full bg-black/50 border border-cyan-800 rounded p-2 mb-2 text-sm text-white" value={project.title} onChange={e => onUpdate(project.id, { title: e.target.value })} />
                     <textarea className="w-full bg-black/50 border border-cyan-800 rounded p-2 text-sm text-white h-20 resize-none" value={project.description} onChange={e => onUpdate(project.id, { description: e.target.value })} />
                     
                     <div className="flex justify-end mt-4">
                        <button
                            onClick={onEditCode}
                            className="w-full py-2 bg-purple-600/30 border border-purple-500 text-purple-200 font-bold rounded flex items-center justify-center gap-2 hover:bg-purple-600/50 transition-all"
                        >
                            <CodeBracketIcon className="w-4 h-4" /> Open in Code Studio
                        </button>
                    </div>
                 </div>
             </div>
        </div>
    </div>;
};

const DEPLOY_PIPELINE = [
    { id: 'gateway', label: 'Quantum Gateway', icon: NetworkIcon, desc: 'Handshake & Routing' },
    { id: 'hosting', label: 'Hybrid Hosting', icon: CloudServerIcon, desc: 'Pod Provisioning' },
    { id: 'domain', label: 'Domain Registry', icon: GlobeIcon, desc: 'QNS Binding' },
    { id: 'deploy', label: 'Deployment', icon: RocketLaunchIcon, desc: 'Artifact Push' },
    { id: 'protocol', label: 'CHIPS Protocol', icon: ShieldCheckIcon, desc: 'EKS Verification' },
    { id: 'network', label: 'Network Sim', icon: ActivityIcon, desc: 'Propagation' },
    { id: 'store', label: 'App Store', icon: BoxIcon, desc: 'Public Listing' }
];

const OpsView: React.FC<{ 
    project: Project; 
    onUpdate: (id: string, data: Partial<Project>) => void; 
    onGlobalDeploy?: (details: { name: string; description: string; code: string; uiStructure?: UIStructure }) => void 
}> = ({ project, onUpdate, onGlobalDeploy }) => {
    const [deployStatus, setDeployStatus] = useState<'IDLE' | 'RUNNING' | 'SUCCESS'>('IDLE');
    const [pipelineStep, setPipelineStep] = useState<number>(0);
    const [logs, setLogs] = useState<string[]>([]);
    const logsEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (logsEndRef.current) logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    const addLog = (msg: string) => setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);

    const handleDeploy = () => {
        if (deployStatus === 'RUNNING') return;
        setDeployStatus('RUNNING');
        setPipelineStep(0);
        setLogs(['Initializing Automated CHIPS Deployment Sequence...', `Target Project: ${project.title}`]);
        
        let currentStep = 0;

        const processNextStep = () => {
            if (currentStep >= DEPLOY_PIPELINE.length) {
                setDeployStatus('SUCCESS');
                addLog("Deployment Pipeline Complete. App is Live.");
                if (onGlobalDeploy) {
                    // Bundle all files into a JSON structure for deployment to ensure dependencies work
                    const entryFileName = 'App.tsx';
                    const fileCount = Object.keys(project.files).length;
                    
                    let codePayload = project.files[entryFileName] || Object.values(project.files)[0] || '';
                    let uiStruct: UIStructure | undefined;

                    // If multiple files, bundle them
                    if (fileCount > 1 || !project.files[entryFileName]) {
                        codePayload = JSON.stringify({
                            entry: project.files[entryFileName] ? entryFileName : Object.keys(project.files)[0],
                            files: project.files
                        });
                    }

                    try {
                        // Attempt to extract UI Structure from entry point for preview
                        const entryContent = project.files[entryFileName] || Object.values(project.files)[0] || '';
                        uiStruct = transpileCodeToStructure(entryContent) || undefined;
                    } catch (e) { console.warn("Failed to extract UI Structure", e); }
                    
                    onGlobalDeploy({
                        name: project.title || "Untitled Project",
                        description: project.description || "No description provided.",
                        code: codePayload,
                        uiStructure: uiStruct
                    });
                }
                setTimeout(() => { setDeployStatus('IDLE'); setPipelineStep(0); }, 5000);
                return;
            }

            setPipelineStep(currentStep);
            const step = DEPLOY_PIPELINE[currentStep];
            addLog(`Step ${currentStep + 1}: ${step.label} - ${step.desc}...`);

            // Simulate specific step actions log
            const subLogs = [
                `Connecting to ${step.label} node...`,
                `Verifying integrity...`,
                `${step.label} task completed successfully.`
            ];

            let subLogIndex = 0;
            const subInterval = setInterval(() => {
                if (subLogIndex < subLogs.length) {
                    addLog(subLogs[subLogIndex]);
                    subLogIndex++;
                } else {
                    clearInterval(subInterval);
                    currentStep++;
                    setTimeout(processNextStep, 800); // Delay between major steps
                }
            }, 600); // Delay between sub-logs
        };

        processNextStep();
    };

     return <div className="p-4 flex flex-col h-full gap-4 overflow-y-auto">
         <div className="bg-black/30 border border-purple-900/50 rounded-lg p-4">
             <div className="flex justify-between items-center mb-4">
                 <h4 className="text-sm font-bold text-purple-300 uppercase tracking-widest flex items-center gap-2">
                     <RocketLaunchIcon className="w-4 h-4" /> CHIPS Deployment Sequence
                 </h4>
                 <div className="flex gap-2">
                     <span className="text-[10px] bg-green-900/30 text-green-400 px-2 py-0.5 rounded border border-green-800">ENV: {project.env}</span>
                     <span className="text-[10px] bg-blue-900/30 text-blue-400 px-2 py-0.5 rounded border border-blue-800">VER: {project.version}</span>
                 </div>
             </div>
             
             {/* Pipeline Visual */}
             <div className="flex items-center justify-between bg-black/50 p-4 rounded-lg border border-gray-800 mb-4 relative overflow-hidden">
                 <div className="absolute inset-0 bg-gradient-to-r from-transparent via-purple-500/5 to-transparent animate-shimmer pointer-events-none"></div>
                 
                 {DEPLOY_PIPELINE.map((step, idx) => {
                     const isActive = deployStatus === 'RUNNING' && pipelineStep === idx;
                     const isDone = (deployStatus === 'RUNNING' && pipelineStep > idx) || deployStatus === 'SUCCESS';
                     
                     return (
                         <React.Fragment key={step.id}>
                             <div className="text-center z-10 relative group">
                                 <div className={`w-8 h-8 rounded-full mx-auto mb-1 flex items-center justify-center transition-all duration-500 ${
                                     isActive ? 'bg-purple-600 shadow-[0_0_15px_rgba(168,85,247,0.5)] scale-110' : 
                                     isDone ? 'bg-green-500 text-black' : 'bg-gray-800 text-gray-500'
                                 }`}>
                                     {isDone ? <CheckCircle2Icon className="w-5 h-5"/> : <step.icon className={`w-4 h-4 ${isActive ? 'text-white animate-pulse' : ''}`} />}
                                 </div>
                                 <span className={`text-[8px] font-bold uppercase block transition-colors ${isActive ? 'text-purple-300' : isDone ? 'text-green-400' : 'text-gray-600'}`}>{step.label}</span>
                                 
                                 {/* Tooltip */}
                                 <div className="absolute top-full left-1/2 -translate-x-1/2 mt-1 w-max px-2 py-1 bg-black/90 border border-gray-700 text-[9px] text-gray-300 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-20">
                                     {step.desc}
                                 </div>
                             </div>
                             {idx < DEPLOY_PIPELINE.length - 1 && (
                                 <div className="flex-grow h-0.5 mx-1 relative bg-gray-800">
                                     <div className={`absolute top-0 left-0 h-full bg-green-500 transition-all duration-1000 ${isDone ? 'w-full' : isActive ? 'w-1/2 animate-pulse' : 'w-0'}`}></div>
                                 </div>
                             )}
                         </React.Fragment>
                     );
                 })}
             </div>

             {/* Deployment Logs */}
             <div className="h-48 bg-black/60 border border-cyan-900/30 rounded-lg p-2 font-mono text-[10px] overflow-y-auto custom-scrollbar mb-4 shadow-inner">
                 {logs.map((log, i) => (
                     <div key={i} className="mb-0.5 text-cyan-300/80 border-b border-white/5 pb-0.5 last:border-0">{log}</div>
                 ))}
                 <div ref={logsEndRef} />
             </div>

             <button 
                onClick={handleDeploy} 
                disabled={deployStatus === 'RUNNING'}
                className={`w-full py-3 border font-bold rounded transition-all flex items-center justify-center gap-2 ${
                    deployStatus === 'IDLE' ? 'bg-purple-600/30 border-purple-500 text-purple-200 hover:bg-purple-600/50' :
                    deployStatus === 'SUCCESS' ? 'bg-green-600/30 border-green-500 text-green-200' :
                    'bg-gray-800/50 border-gray-600 text-gray-400 cursor-not-allowed'
                }`}
            >
                 {deployStatus === 'IDLE' && <><RocketLaunchIcon className="w-4 h-4" /> Initiate Sequence</>}
                 {deployStatus === 'RUNNING' && <><LoaderIcon className="w-4 h-4 animate-spin" /> Deploying Sequence...</>}
                 {deployStatus === 'SUCCESS' && <><CheckCircle2Icon className="w-4 h-4" /> Deployment Successful</>}
             </button>
         </div>
     </div>;
};

// ... (Other helper components: FileExplorer, Editor, Preview, EditorWorkspace - these remain mostly internal to ChipsDevPlatform)

// Include the internal components here for completeness (simplified for this update to focus on imports)
const FileExplorer: React.FC<{ 
    files: string[], 
    selectedFile: string, 
    onSelect: (file: string) => void,
    onRename: (oldName: string, newName: string) => void,
    onDelete: (file: string) => void
}> = ({ files, selectedFile, onSelect, onRename, onDelete }) => {
    // ... (FileExplorer Implementation) ...
    return <div className="h-full bg-black/30 p-2 overflow-y-auto border-r border-cyan-800/50">
        {/* Simplified File List for brevity in this update block */}
        {files.map(file => (
            <div key={file} onClick={() => onSelect(file)} className={`cursor-pointer p-1 ${selectedFile === file ? 'bg-cyan-900' : ''}`}>{file}</div>
        ))}
    </div>;
};

const Editor: React.FC<{ content: string, fileName: string, onContentChange: (newContent: string) => void }> = ({ content, fileName, onContentChange }) => {
    return <MonacoEditorWrapper code={content} onChange={(v) => onContentChange(v || '')} language={fileName.endsWith('json') ? 'json' : 'typescript'} />;
};

const EditorWorkspace: React.FC<{
    files: { [key: string]: string };
    selectedFile: string;
    onSelectFile: (file: string) => void;
    unsavedChanges: string | null;
    onContentChange: (content: string) => void;
    onRenameFile: (oldName: string, newName: string) => void;
    onDeleteFile: (fileName: string) => void;
}> = ({ files, selectedFile, onSelectFile, onContentChange }) => {
    return (
        <div className="h-full flex flex-col">
            <div className="flex-grow grid grid-cols-[200px_1fr] min-h-0">
                <FileExplorer files={Object.keys(files)} selectedFile={selectedFile} onSelect={onSelectFile} onRename={()=>{}} onDelete={()=>{}} />
                <Editor content={files[selectedFile]} fileName={selectedFile} onContentChange={onContentChange} />
            </div>
        </div>
    );
};

interface StudioViewProps { 
    project: Project; 
    onUpdate: (id: string, data: Partial<Project>) => void;
    onAiAssist?: (currentCode: string, instruction: string) => Promise<string>;
    onGlobalDeploy?: (details: { name: string; description: string; code: string; uiStructure?: UIStructure }) => void;
    onCreateProject: (name: string, description: string, files: { [key: string]: string }) => void;
    onSave?: () => void;
}

const StudioView: React.FC<StudioViewProps> = ({ project, onUpdate, onAiAssist, onGlobalDeploy, onCreateProject, onSave }) => {
    const [activeFile, setActiveFile] = useState<string>('App.tsx');
    const [showPreview, setShowPreview] = useState(true);

    const [previewState, setPreviewState] = useState<{ [key: string]: any }>({});

    const uiStructure = useMemo(() => {
        const code = project.files['App.tsx'] || '';
        return transpileCodeToStructure(code);
    }, [project.files['App.tsx']]);

    useEffect(() => {
        const code = project.files['App.tsx'] || '';
        setPreviewState(extractStateFromCode(code));
    }, [project.files['App.tsx']]);

    const handlePreviewAction = useCallback((actionName: string) => {
        setPreviewState(prev => {
            if (actionName.startsWith('toggle')) {
                const possibleState = actionName.replace('toggle', '');
                const key = Object.keys(prev).find(k => 
                    k.toLowerCase() === possibleState.toLowerCase() ||
                    possibleState.toLowerCase().startsWith(k.toLowerCase())
                );
                if (key && typeof prev[key] === 'boolean') {
                    return { ...prev, [key]: !prev[key] };
                }
            }
            return prev;
        });
    }, []);
    
    // ... (State logic for Studio View) ...
    const fileSystemOps = useMemo(() => ({
        listFiles: () => Object.keys(project.files),
        readFile: (path: string) => project.files[path] || null,
        writeFile: (path: string, content: string) => {
             onUpdate(project.id, { files: { ...project.files, [path]: content }, lastEdited: Date.now() });
        },
        deleteFile: (path: string) => {
            const newFiles = { ...project.files };
            delete newFiles[path];
            onUpdate(project.id, { files: newFiles, lastEdited: Date.now() });
        }
    }), [project, onUpdate]);
    
    const projectOps = useMemo(() => ({
        createProject: (name: string, desc: string, files: any) => onCreateProject(name, desc, files),
        listProjects: () => [], 
        switchProject: (id: string) => {} 
    }), [onCreateProject]);

    const { agentQProps } = useAgentQ({
        focusedPanelId: 'chips-dev-platform',
        panelInfoMap: {},
        qcosVersion: 4.5,
        systemHealth: { neuralLoad: 45 } as any,
        onDashboardControl: () => {},
        fileSystemOps,
        projectOps
    });

    return (
        <div className="h-full flex gap-2 animate-fade-in relative overflow-hidden">
             <div className="w-80 flex-shrink-0 flex flex-col bg-black/60 border-r border-cyan-900/50 backdrop-blur-md relative z-20 h-full">
                <AgentQ 
                    {...agentQProps} 
                    isOpen={true} 
                    onToggleOpen={() => {}} 
                    embedded={true} 
                    currentContextName="AGI-Native Dev"
                />
            </div>
            <div className="flex-grow flex flex-col relative min-w-0 bg-black/40 overflow-hidden">
                <div className={`flex-grow grid min-h-0 ${showPreview ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'}`}>
                    <div className="flex flex-col border-r border-cyan-900/30 overflow-hidden">
                        <EditorWorkspace 
                            files={project.files} 
                            selectedFile={activeFile} 
                            onSelectFile={setActiveFile} 
                            unsavedChanges={null}
                            onContentChange={(val) => onUpdate(project.id, { files: {...project.files, [activeFile]: val} })}
                            onRenameFile={()=>{}}
                            onDeleteFile={()=>{}}
                        />
                    </div>
                    {showPreview && (
                        <div className="flex flex-col bg-slate-950/50 overflow-hidden border-l border-cyan-900/30">
                            <div className="p-2 border-b border-cyan-900/30 bg-black/40 flex justify-between items-center flex-shrink-0">
                                <div className="flex items-center gap-2">
                                    <EyeIcon className="w-3.5 h-3.5 text-cyan-400" />
                                    <span className="text-[10px] font-black text-white uppercase tracking-widest">Live UI Preview</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-green-900/20 border border-green-800/50 text-[8px] text-green-400 font-mono">
                                        <div className="w-1 h-1 bg-green-400 rounded-full animate-pulse"></div>
                                        SYNCED
                                    </div>
                                    <button onClick={() => setShowPreview(false)} className="p-1 hover:bg-white/10 rounded text-gray-500 hover:text-white transition-colors">
                                        <XIcon className="w-3.5 h-3.5" />
                                    </button>
                                </div>
                            </div>
                            <div className="flex-grow p-6 overflow-auto custom-scrollbar bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-cyan-950/10 via-black to-black relative">
                                <div className="absolute inset-0 opacity-10 pointer-events-none bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')]"></div>
                                
                                <div className="h-full w-full max-w-2xl mx-auto">
                                    {uiStructure ? (
                                        <div className="h-full w-full rounded-2xl border border-cyan-500/20 bg-black/60 shadow-[0_0_50px_rgba(0,0,0,0.5)] overflow-hidden relative flex flex-col">
                                            <div className="flex-grow overflow-auto custom-scrollbar">
                                                <HolographicPreviewRenderer 
                                                    structure={uiStructure} 
                                                    state={previewState} 
                                                    onAction={handlePreviewAction} 
                                                />
                                            </div>
                                            <div className="p-2 bg-black/80 border-t border-cyan-900/30 flex justify-between items-center text-[8px] text-cyan-800 font-mono uppercase">
                                                <span>Render Engine: Holographic v2</span>
                                                <span>App: {project.title}</span>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="h-full flex flex-col items-center justify-center text-gray-600 text-center p-8 bg-black/40 rounded-2xl border border-dashed border-cyan-900/30">
                                            <AlertTriangleIcon className="w-12 h-12 mb-4 opacity-20 text-cyan-500" />
                                            <p className="text-sm font-black text-cyan-900 uppercase tracking-widest">Parsing Error</p>
                                            <p className="text-[10px] mt-2 text-gray-700 max-w-xs">Unable to extract UI structure from App.tsx. Ensure the component uses a standard return statement with JSX.</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
                {!showPreview && (
                    <button 
                        onClick={() => setShowPreview(true)}
                        className="absolute bottom-6 right-6 p-4 bg-cyan-600 rounded-full text-white shadow-[0_0_20px_rgba(6,182,212,0.5)] hover:bg-cyan-500 hover:scale-110 transition-all z-30 group"
                        title="Show Preview"
                    >
                        <EyeIcon className="w-6 h-6 group-hover:animate-pulse" />
                    </button>
                )}
            </div>
        </div>
    );
};

interface ChipsDevPlatformProps {
    onAiAssist?: (currentCode: string, instruction: string) => Promise<string>;
    onDeploy?: (details: { name: string; description: string; code: string; uiStructure?: UIStructure }) => void;
}

const ChipsDevPlatform: React.FC<ChipsDevPlatformProps> = ({ onAiAssist, onDeploy }) => {
    const [activeTab, setActiveTab] = useState<DevTab>('coding');
    const [projects, setProjects] = useState<Project[]>(initialProjects);
    const [activeProjectId, setActiveProjectId] = useState<string>(initialProjects[0].id);

    const activeProject = projects.find(p => p.id === activeProjectId) || projects[0];

    const handleUpdateProject = (id: string, data: Partial<Project>) => {
        setProjects(prev => prev.map(p => p.id === id ? { ...p, ...data } : p));
    };

    const handleCreateProject = (name: string, description: string, files: { [key: string]: string }) => {
        const newProject: Project = {
            id: `proj_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            title: name,
            description,
            env: 'Development',
            version: '0.1.0',
            tasks: [],
            files,
            lastEdited: Date.now()
        };
        setProjects(prev => [...prev, newProject]);
        setActiveProjectId(newProject.id);
        setActiveTab('coding');
    };

    const handleDeleteProject = (id: string) => {
        const newProjects = projects.filter(p => p.id !== id);
        setProjects(newProjects);
        if (activeProjectId === id && newProjects.length > 0) {
            setActiveProjectId(newProjects[0].id);
        }
    };
    
    // Fork template
    const handleForkTemplate = (tpl: any) => {
        handleCreateProject(tpl.title, tpl.desc, tpl.files);
    };

    return (
        <GlassPanel title="ChipsDev Platform">
            <div className="flex flex-col h-full bg-slate-950/50 overflow-hidden">
                {/* Navigation Bar */}
                <div className="flex items-center justify-between p-2 border-b border-cyan-900/50 bg-black/40">
                    <div className="flex gap-2 overflow-x-auto no-scrollbar">
                        <NavButton active={activeTab === 'inspiration'} onClick={() => setActiveTab('inspiration')} icon={LightBulbIcon} label="Inspiration" />
                        <NavButton active={activeTab === 'planning'} onClick={() => setActiveTab('planning')} icon={BriefcaseIcon} label="Planning" />
                        <NavButton active={activeTab === 'coding'} onClick={() => setActiveTab('coding')} icon={CodeBracketIcon} label="Studio" />
                        <NavButton active={activeTab === 'ops'} onClick={() => setActiveTab('ops')} icon={RocketLaunchIcon} label="Ops" />
                    </div>
                    
                    <div className="flex items-center gap-2">
                        <select 
                            className="bg-black/50 border border-cyan-800 rounded px-2 py-1 text-xs text-white outline-none focus:border-cyan-500 max-w-[150px]"
                            value={activeProjectId}
                            onChange={(e) => setActiveProjectId(e.target.value)}
                        >
                            {projects.map(p => <option key={p.id} value={p.id}>{p.title}</option>)}
                        </select>
                        <div className="h-6 w-px bg-cyan-900/50"></div>
                        <div className="flex items-center gap-1 text-[10px] text-cyan-600 bg-black/30 px-2 py-0.5 rounded border border-cyan-900/30">
                            <CpuChipIcon className="w-3 h-3" />
                            <span>Q-Core: Online</span>
                        </div>
                    </div>
                </div>

                {/* Main Content */}
                <div className="flex-grow min-h-0 overflow-hidden relative">
                    {activeTab === 'inspiration' && (
                        <InspirationView templates={allTemplates} onFork={handleForkTemplate} />
                    )}
                    
                    {activeTab === 'planning' && activeProject && (
                        <PlanningView 
                            project={activeProject} 
                            onUpdate={handleUpdateProject} 
                            projects={projects}
                            onSelectProject={setActiveProjectId}
                            onDeleteProject={handleDeleteProject}
                            onUseBlueprint={handleForkTemplate}
                            onEditCode={() => setActiveTab('coding')}
                        />
                    )}

                    {activeTab === 'coding' && activeProject && (
                        <StudioView 
                            project={activeProject} 
                            onUpdate={handleUpdateProject} 
                            onAiAssist={onAiAssist}
                            onGlobalDeploy={onDeploy}
                            onCreateProject={handleCreateProject}
                            onSave={() => handleUpdateProject(activeProject.id, { lastEdited: Date.now() })}
                        />
                    )}

                    {activeTab === 'ops' && activeProject && (
                         <OpsView 
                            project={activeProject} 
                            onUpdate={handleUpdateProject}
                            onGlobalDeploy={onDeploy}
                         />
                    )}

                    {!activeProject && (
                        <div className="h-full flex items-center justify-center text-gray-500">
                            No project selected.
                        </div>
                    )}
                </div>
            </div>
        </GlassPanel>
    );
};

export default ChipsDevPlatform;
