import { ElectronBridge } from '../types/electron';

// Mock Implementation for Browser Preview
const mockBridge: ElectronBridge = {
    installChips: async () => {
        console.log("[MOCK] Installing Chips Browser...");
        await new Promise(resolve => setTimeout(resolve, 2000));
        return { success: true, message: "Installation Complete (Mock)" };
    },
    monitorPillars: async () => {
        console.log("[MOCK] Monitoring Pillars...");
        return [
            { name: 'pillar_alpha.dat', status: 'SECURE', integrity: 100 },
            { name: 'pillar_beta.dat', status: 'SECURE', integrity: 100 },
            { name: 'pillar_gamma.dat', status: 'SECURE', integrity: 100 },
            { name: 'pillar_delta.dat', status: 'SECURE', integrity: 100 }
        ];
    },
    runCommand: async (command: string) => {
        console.log(`[MOCK] Running Command: ${command}`);
        await new Promise(resolve => setTimeout(resolve, 500));
        if (command === 'ls') return { output: 'pillar_alpha.dat\npillar_beta.dat\n', error: false };
        if (command === 'status') return { output: 'QCOS KERNEL: ACTIVE', error: false };
        return { output: `Command '${command}' executed successfully.`, error: false };
    },
    saveWeights: async (weights: any) => {
        console.log("[MOCK] Saving Weights:", weights);
        localStorage.setItem('qiai_weights', JSON.stringify(weights));
        return { success: true, path: 'local_storage' };
    },
    on: (channel: string, callback: any) => {},
    off: (channel: string, callback: any) => {}
};

// Export the real bridge if available, otherwise the mock
export const electronBridge: ElectronBridge = (window as any).electron || mockBridge;
