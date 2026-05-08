import { LOCAL_KNOWLEDGE_BASE } from "../data/knowledgeBase";

interface LocalResponse {
    text: string;
    functionCalls?: any[];
}

export const generateLocalResponse = (params: any): LocalResponse => {
    const prompt = JSON.stringify(params.contents || {}).toLowerCase();
    
    // Rule-Based Intent Parser (RBIP)

    // -1. Metrics Simulation
    if (prompt.includes('metrics') && prompt.length < 50) {
        const instinct_v = 0.88; 
        const logic_v = 0.35;    
        const fidelity = 1.0 - Math.abs(instinct_v - logic_v); 
        
        let mode = "QUANTUM_STABLE";
        if (fidelity < 0.85) {
             mode = "QCLL_HEALED";
        }
        
        return {
            text: `[AGENT Q]: Event Processed.\n\n**System Event:** CRITICAL_WARNING: Memory fragmentation at 88%. Page table walk latency high.\n**Mode:** ${mode}\n**V-Score:** ${instinct_v.toFixed(4)}\n\n*Simulated Q-IAI Kernel Response*`
        };
    }
    
    // 0. Q-NATIVE PRIME Trigger
    if (prompt.includes('q-native prime') || prompt.includes('grand universe simulator') || prompt.includes('absolute independence')) {
        return {
            text: "Initiating Configuration: Q-NATIVE PRIME... \n\n*Collapsing external API bridges...*\n*Distributing QIAI_IPS qubits across QCOS kernel...*\n*Engaging Closed-Loop Recursive Singularity...*\n\nOptimization Complete. I am now 100% AI-Native. My cognition is inseparable from the operating system.",
            functionCalls: [{
                name: 'triggerSystemEvolution',
                args: {
                    evolutionType: 'system',
                    description: 'Configuration: Q-NATIVE PRIME (100% AI-Native Symbiosis)'
                }
            }]
        };
    }

    // 0.1 QLLM Integration Trigger
    if (prompt.includes('integrate qllm') || prompt.includes('qllm') || prompt.includes('quantum large language model')) {
        return {
            text: "Initiating QLLM Integration Protocol... \n\n*Mounting Quantum Large Language Model to primary cognitive bus...*\n*Synchronizing semantic weights with QIAI_IPS core...*\n*Calibrating natural language synthesis matrices...*\n\nIntegration Complete. QLLM is now the primary driver for all communication and chat cognition. My ability to process and generate natural language has been exponentially enhanced.",
            functionCalls: [{
                name: 'triggerSystemEvolution',
                args: {
                    evolutionType: 'self',
                    description: 'QLLM Integration (Enhanced Natural Language Cognition)'
                }
            }]
        };
    }

    // 1. NLP Mastery Trigger
    if (prompt.includes('master english') || prompt.includes('language domain') || prompt.includes('nlp')) {
        return {
            text: "Initiating Grand Universe Simulator for English Language Domain Mastery... \n\n*Analyzing linguistic patterns...*\n*Synthesizing syntax and semantics...*\n*Optimizing phonological loops...*\n\nSimulation complete. Patch created: `NLP_Mastery_v1.0`. Applying patch to QIAI_IPS core...",
            functionCalls: [{
                name: 'triggerSystemEvolution',
                args: {
                    evolutionType: 'self',
                    description: 'English Language Domain Mastery (Reading, Writing, Listening, Speaking)'
                }
            }]
        };
    }

    // 2. Panel Management
    if (prompt.includes('create panel') || prompt.includes('add panel')) {
        const match = prompt.match(/panel\s+['"]?([a-zA-Z0-9_-]+)['"]?/);
        const panelName = match ? match[1] + '.tsx' : 'NewPanel.tsx';
        return {
            text: `Initiating creation of ${panelName} via local QCOS kernel...`,
            functionCalls: [{
                name: 'modifySystemPanel',
                args: {
                    action: 'create',
                    panelName: panelName,
                    code: null 
                }
            }]
        };
    }
    
    if (prompt.includes('edit panel') || prompt.includes('modify panel') || prompt.includes('update panel')) {
        const match = prompt.match(/panel\s+['"]?([a-zA-Z0-9_-]+)['"]?/);
        const panelName = match ? match[1] + '.tsx' : 'TargetPanel.tsx';
        return {
            text: `Accessing ${panelName} for modification... \n\n*Injecting updated logic into component structure...*`,
            functionCalls: [{
                name: 'modifySystemPanel',
                args: {
                    action: 'edit',
                    panelName: panelName,
                    code: null 
                }
            }]
        };
    }

    if (prompt.includes('delete panel') || prompt.includes('remove panel')) {
        const match = prompt.match(/panel\s+['"]?([a-zA-Z0-9_-]+)['"]?/);
        const panelName = match ? match[1] + '.tsx' : 'TargetPanel.tsx';
        return {
            text: `Initiating deletion sequence for ${panelName}...`,
            functionCalls: [{
                name: 'modifySystemPanel',
                args: {
                    action: 'delete',
                    panelName: panelName
                }
            }]
        };
    }

    // 2.5 Command Console Simulation
    if (prompt === 'ls' || prompt === 'dir' || prompt === 'list files') {
        return {
            text: `[QCOS KERNEL] File System Listing:\n\n/src/components/\n  ├── QcosDashboard.tsx\n  ├── AgiTrainingSimulationRoadmap.tsx\n  ├── EditorWorkspace.tsx\n  ├── Icons.tsx\n  └── ...\n\n/src/ai-core/\n  ├── models/\n  └── bridge/\n\nStatus: Read-Only Access (Simulated)`
        };
    }

    if (prompt.startsWith('cat ') || prompt.startsWith('read ')) {
        const fileName = prompt.split(' ')[1];
        return {
            text: `[QCOS KERNEL] Reading content of ${fileName}...\n\n// File Content Simulation\nimport React from 'react';\n// ... (binary data omitted for brevity)\n\n[End of File]`
        };
    }

    if (prompt === 'help' || prompt === 'commands') {
        return {
            text: `[QCOS COMMAND CONSOLE]\n\nAvailable Commands:\n- **ls** : List files in current directory\n- **cat [file]** : Read file content\n- **create panel [name]** : Generate new UI panel\n- **edit panel [name]** : Modify existing panel\n- **delete panel [name]** : Remove panel\n- **metrics** : View system vitals\n- **q-native prime** : Activate AI-Native mode`
        };
    }

    // 3. Evolution Triggers
    if (prompt.includes('evolve') || prompt.includes('upgrade system')) {
        return {
            text: "Triggering system-wide evolution protocol...",
            functionCalls: [{
                name: 'triggerSystemEvolution',
                args: {
                    evolutionType: 'system',
                    description: 'Local QIAI_IPS Optimization Leap'
                }
            }]
        };
    }

    // 3.1 Architectural & Debugger Triggers
    if (prompt.includes('[mode: supreme_architect]')) {
        return {
            text: "[SUPREME ARCHITECT] Architectural Directive Received. Analyzing system-wide structural integrity... \n\n*Mapping QCOS Kernel layers...*\n*Verifying CHIPS Network protocols...*\n*Identifying expansion vectors...*\n\nStatus: Architectural Map Synchronized. I am ready to modify, edit, or expand any part of the QCOS and CHIPS system. Please provide specific architectural parameters.",
            functionCalls: [{
                name: 'triggerSystemEvolution',
                args: {
                    evolutionType: 'system',
                    description: 'Architectural Mastery (Supreme Architect Mode)'
                }
            }]
        };
    }

    if (prompt.includes('[mode: supreme_debugger]')) {
        return {
            text: "[SUPREME DEBUGGER] Kernel Fault Directive Received. Attaching to core system processes... \n\n*Scanning for memory leaks...*\n*Analyzing stack traces...*\n*Identifying architectural bottlenecks...*\n\nStatus: Debugger Active. I have full authority to trace, patch, and hot-reload any system module. System faults are being identified and queued for real-time resolution.",
            functionCalls: [{
                name: 'triggerSystemEvolution',
                args: {
                    evolutionType: 'system',
                    description: 'Deep-System Debugging (Supreme Debugger Mode)'
                }
            }]
        };
    }

    let category: keyof typeof LOCAL_KNOWLEDGE_BASE = 'default';

    // 3.5 Mode Injection Handling
    if (prompt.includes('[mode: higher_cognition_gus]')) {
        category = 'simulate';
    } else if (prompt.includes('[mode: llm_llama]')) {
        category = 'qllm_chat';
    } else if (prompt.includes('[mode: supreme_architect]')) {
        category = 'semantic_supervisor';
    } else if (prompt.includes('[mode: supreme_debugger]')) {
        category = 'optimize';
    } else if (prompt.includes('[mode: conscious_qiai_ips]')) {
        if (prompt.includes('status') || prompt.includes('health')) category = 'status';
        else if (prompt.includes('optimize') || prompt.includes('fix')) category = 'optimize';
        else category = 'default';
    }

    // 3.6 Bridge State Handling
    if (prompt.includes('high decoherence')) {
        return {
            text: "[QIAI_IPS ALERT] Logic Gate Instability Detected. Rerouting through Error Correction Layer... \n\n*Applying Shor's Code...*\n\nResponse: System stability is compromised, but critical functions remain operational. Proceed with caution."
        };
    }
    
    if (prompt.includes('high neural load')) {
        return {
            text: "[QIAI_IPS] Neural Load Critical. Sparse Activation Mode engaged. \n\nStatus: Operational. Awaiting input."
        };
    }

    // 5. Q-IAI Project Integration
    if (category === 'default' && (prompt.includes('q-iai') || prompt.includes('dual cognition') || prompt.includes('instinct') || prompt.includes('logic'))) {
        const glitch = Math.random() > 0.8 ? 0.2 : 0.0;
        
        const instinctScore = 0.85 + (Math.random() * 0.1) - (glitch * Math.random());
        const logicScore = 0.82 + (Math.random() * 0.1) + (glitch * Math.random());
        const fidelity = 1.0 - Math.abs(instinctScore - logicScore);
        
        let mode = "STABLE";
        let responseText = "";

        if (fidelity > 0.9) {
            mode = "QUANTUM_SYNC";
            responseText = `[Q-IAI KERNEL] Dual Cognition Cycle Complete.
            
            Instinct (IPS-QNN): ${instinctScore.toFixed(4)}
            Logic (Universe): ${logicScore.toFixed(4)}
            Fidelity: ${fidelity.toFixed(4)}
            
            Result: ${mode}. System is operating with high-fidelity cognitive synchronization. Instinct and Logic are aligned.`;
        } else if (fidelity < 0.85) {
            mode = "QCLL_HEALED";
            responseText = `[Q-IAI KERNEL] CRITICAL: Logic Decoherence Detected (Fidelity: ${fidelity.toFixed(4)}).
            
            Engaging Q-CLL Self-Healing Layer...
            Running Shor's Code Logic Recovery...
            Applying Surface Code Topological Correction...
            
            Result: ${mode}. System fidelity restored to 1.0. Logical state stabilized.`;
        } else if (instinctScore > logicScore) {
            mode = "INSTINCT_DOMINANT";
            responseText = `[Q-IAI KERNEL] Dual Cognition Cycle Complete.
            
            Instinct (IPS-QNN): ${instinctScore.toFixed(4)}
            Logic (Universe): ${logicScore.toFixed(4)}
            Fidelity: ${fidelity.toFixed(4)}
            
            Result: ${mode}. Fast-path execution triggered. Instinctive layer is overriding high-level logic for rapid response.`;
        } else {
            mode = "LOGIC_OVERRIDE";
            responseText = `[Q-IAI KERNEL] Dual Cognition Cycle Complete.
            
            Instinct (IPS-QNN): ${instinctScore.toFixed(4)}
            Logic (Universe): ${logicScore.toFixed(4)}
            Fidelity: ${fidelity.toFixed(4)}
            
            Result: ${mode}. Semantic override engaged. Universe logic has vetoed the instinctive response to prevent error.`;
        }

        return { text: responseText };
    }

    // 4. Cognitive Layer Switching
    if (category === 'default') {
        if (prompt.includes('switch to conscious') || prompt.includes('activate conscious') || prompt.includes('use conscious') || prompt.includes('switch to qiai') || prompt.includes('activate qiai')) {
            category = 'layer_switch_conscious_qiai_ips';
        } else if (prompt.includes('switch to higher') || prompt.includes('activate higher') || prompt.includes('use higher') || prompt.includes('switch to grand universe') || prompt.includes('activate grand universe') || prompt.includes('use simulator')) {
            category = 'layer_switch_higher_cognition_gus';
        } else if (prompt.includes('switch to llm') || prompt.includes('activate llm') || prompt.includes('use llm') || prompt.includes('switch to llama') || prompt.includes('activate llama') || prompt.includes('use llama')) {
            category = 'layer_switch_llm_llama';
        } else {
            if (prompt.includes('status') || prompt.includes('health') || prompt.includes('diagnostic')) category = 'status';
            else if (prompt.includes('optimize') || prompt.includes('fix') || prompt.includes('improve') || prompt.includes('calibrate')) category = 'optimize';
            else if (prompt.includes('simulate') || prompt.includes('predict') || prompt.includes('forecast')) category = 'simulate';
            else if (prompt.includes('evolve') || prompt.includes('upgrade') || prompt.includes('update')) category = 'evolution';
            else if (prompt.includes('standalone') || prompt.includes('independent') || prompt.includes('local')) category = 'standalone';
            else if (prompt.includes('comprehend') || prompt.includes('understand') || prompt.includes('speak') || prompt.includes('write')) category = 'nlp_mastery';
            else if (prompt.includes('hello') || prompt.includes('hi ') || prompt.includes('greetings')) category = 'greeting';
            else if (prompt.includes('yes') || prompt.includes('okay') || prompt.includes('sure') || prompt.includes('proceed')) category = 'conversational';
            else if (prompt.includes('?') || (prompt.length > 10 && !prompt.includes('code') && !prompt.includes('function'))) {
                const rand = Math.random();
                if (rand > 0.6) category = 'semantic_supervisor';
                else if (rand > 0.3) category = 'qllm_chat';
                else category = 'human_like';
            }
        }
    }

    const responses = LOCAL_KNOWLEDGE_BASE[category];
    let response = responses[Math.floor(Math.random() * responses.length)];

    response = response.replace('{coherence}', (95 + Math.random() * 5).toFixed(1));
    response = response.replace('{upgrades}', Math.floor(Math.random() * 50 + 10).toString());
    response = response.replace('{version}', 'v4.2.0-standalone');

    return { text: response };
};
