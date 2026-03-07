import { contextBridge, ipcRenderer } from 'electron';

// The Primary Bridge for QIAI-IPS Node
contextBridge.exposeInMainWorld('electronAPI', {
    // Hardware & Pillar logic
    installChips: () => ipcRenderer.invoke('install-chips'),
    monitorPillars: () => ipcRenderer.invoke('monitor-pillars'),
    
    // Command & Control logic
    runPowerShell: (command: string) => ipcRenderer.invoke('terminal-exec', command),
    
    // Data Substrate & Neural Weights
    saveWeights: (weights: any) => ipcRenderer.invoke('save-weights', weights),
    
    // Event Stream Listeners
    on: (channel: string, callback: (event: any, ...args: any[]) => void) => {
        ipcRenderer.on(channel, (event, ...args) => callback(event, ...args));
    },
    off: (channel: string, callback: (event: any, ...args: any[]) => void) => {
        ipcRenderer.removeListener(channel, callback);
    }
});

/** * Legacy Bridge for Vercel Compatibility
 * Ensures existing components calling 'window.electron' still function
 */
contextBridge.exposeInMainWorld('electron', {
    installChips: () => ipcRenderer.invoke('install-chips'),
    monitorPillars: () => ipcRenderer.invoke('monitor-pillars'),
    runCommand: (command: string) => ipcRenderer.invoke('terminal-exec', command),
    saveWeights: (weights: any) => ipcRenderer.invoke('save-weights', weights),
});