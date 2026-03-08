import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
    installChips: () => ipcRenderer.invoke('install-chips'),
    monitorPillars: () => ipcRenderer.invoke('monitor-pillars'),
    runPowerShell: (command: string) => ipcRenderer.invoke('terminal-exec', command),
    saveWeights: (weights: any) => ipcRenderer.invoke('save-weights', weights),
    on: (channel: string, callback: (event: any, ...args: any[]) => void) => {
        ipcRenderer.on(channel, callback);
    },
    off: (channel: string, callback: (event: any, ...args: any[]) => void) => {
        ipcRenderer.removeListener(channel, callback);
    }
});

// Keep existing 'electron' bridge for backward compatibility if needed
contextBridge.exposeInMainWorld('electron', {
    installChips: () => ipcRenderer.invoke('install-chips'),
    monitorPillars: () => ipcRenderer.invoke('monitor-pillars'),
    runCommand: (command: string) => ipcRenderer.invoke('terminal-exec', command),
    saveWeights: (weights: any) => ipcRenderer.invoke('save-weights', weights),
    on: (channel: string, callback: (event: any, ...args: any[]) => void) => {
        ipcRenderer.on(channel, callback);
    },
    off: (channel: string, callback: (event: any, ...args: any[]) => void) => {
        ipcRenderer.removeListener(channel, callback);
    }
});
