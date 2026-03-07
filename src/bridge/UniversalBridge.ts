// Define the shape of our bridge to match the Main process logic
export interface IUniversalBridge {
    isElectron: boolean;
    installChips: () => Promise<{ success: boolean; message: string }>;
    // Updated to match the { status, details } shape from main.ts
    monitorPillars: () => Promise<{ 
        status: string; 
        details: { file: string; exists: boolean; isGrounded: boolean }[] 
    }>;
    runPowerShell: (command: string) => Promise<string>; // Main.ts returns a raw string
    saveWeights: (weights: any) => Promise<{ success: boolean; path: string; error?: string }>;
}

// Mock implementation for Web/Vercel environment
const WebMockBridge: IUniversalBridge = {
    isElectron: false,
    installChips: async () => {
        await new Promise(resolve => setTimeout(resolve, 1500));
        return { success: true, message: "Cloud Preview Installation Complete" };
    },
    monitorPillars: async () => {
        return {
            status: "CLOUD_SIM",
            details: [
                { file: 'ChipsBrowser.exe', exists: true, isGrounded: true },
                { file: 'AgentCommandConsole.exe', exists: true, isGrounded: true },
                { file: 'EKSBridgeService.exe', exists: true, isGrounded: true },
                { file: 'qlang.exe', exists: true, isGrounded: true }
            ]
        };
    },
    runPowerShell: async (command: string) => {
        return `[CLOUD_PREVIEW] Simulated response for: ${command}\n> Physical hardware not detected.\n> Operating in Neural Simulation mode.`;
    },
    saveWeights: async (weights: any) => {
        localStorage.setItem('qiai_weights', JSON.stringify(weights));
        return { success: true, path: 'browser_local_storage' };
    }
};

// The Universal Bridge - Environment Aware
export const UniversalBridge: IUniversalBridge = (() => {
    // Check for the API exposed by our updated preload.ts
    const api = (window as any).electronAPI;

    if (api) {
        return {
            isElectron: true,
            installChips: api.installChips,
            monitorPillars: api.monitorPillars,
            runPowerShell: api.runPowerShell,
            saveWeights: api.saveWeights
        };
    }

    return WebMockBridge;
})();