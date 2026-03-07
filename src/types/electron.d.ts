export interface ElectronBridge {
    installChips: () => Promise<{ success: boolean; message: string }>;
    monitorPillars: () => Promise<{ name: string; status: string; integrity: number }[]>;
    runCommand: (command: string) => Promise<{ output: string; error: boolean }>;
    saveWeights: (weights: any) => Promise<{ success: boolean; path: string; error?: string }>;
    on: (channel: string, callback: (event: any, ...args: any[]) => void) => void;
    off: (channel: string, callback: (event: any, ...args: any[]) => void) => void;
}

declare global {
    interface Window {
        electron?: ElectronBridge;
    }
}
