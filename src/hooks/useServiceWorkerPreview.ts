
import { useEffect, useState, useRef, useCallback } from 'react';
import { WebContainer, FileSystemTree } from '@webcontainer/api';

interface UseWebContainerPreviewProps {
    files: { [path: string]: string };
    onTerminalOutput?: (data: string) => void;
}

// Singleton instance to prevent multiple boots during hot-reloads
let webContainerInstance: WebContainer | null = null;

export const useServiceWorkerPreview = ({ files, onTerminalOutput }: UseWebContainerPreviewProps) => {
    const [iframeUrl, setIframeUrl] = useState<string | null>(null);
    const [isReady, setIsReady] = useState(false);
    const [isBooting, setIsBooting] = useState(true);
    const serverProcessRef = useRef<any>(null);

    // --- 1. VFS Integration: Convert flat file map to FileSystemTree ---
    const convertToFileSystemTree = (fileMap: { [path: string]: string }): FileSystemTree => {
        const tree: FileSystemTree = {};

        for (const [path, content] of Object.entries(fileMap)) {
            const parts = path.split('/');
            let currentLevel = tree;

            for (let i = 0; i < parts.length; i++) {
                const part = parts[i];
                const isFile = i === parts.length - 1;

                if (isFile) {
                    currentLevel[part] = {
                        file: { contents: content }
                    };
                } else {
                    if (!currentLevel[part]) {
                        currentLevel[part] = {
                            directory: {}
                        };
                    }
                    // Type assertion safe here due to structure
                    currentLevel = (currentLevel[part] as any).directory;
                }
            }
        }
        return tree;
    };

    // --- 2. Boot Sequence & Smart Start ---
    useEffect(() => {
        const boot = async () => {
            try {
                // Initialize Singleton
                if (!webContainerInstance) {
                    webContainerInstance = await WebContainer.boot();
                }

                const tree = convertToFileSystemTree(files);
                await webContainerInstance.mount(tree);

                // Setup listener for server-ready *before* starting processes
                webContainerInstance.on('server-ready', (port, url) => {
                    onTerminalOutput?.(`[System] Server ready on port ${port}: ${url}`);
                    setIframeUrl(url);
                    setIsReady(true);
                    setIsBooting(false);
                });

                // Detect Project Structure
                const hasPackageJson = 'package.json' in files;
                let installProcess;

                if (hasPackageJson) {
                    onTerminalOutput?.('[System] package.json detected. Installing dependencies...');
                    installProcess = await webContainerInstance.spawn('npm', ['install']);
                    
                    installProcess.output.pipeTo(new WritableStream({
                        write(data) { onTerminalOutput?.(data); }
                    }));

                    const installExitCode = await installProcess.exit;
                    if (installExitCode !== 0) {
                        onTerminalOutput?.(`[Error] npm install failed with code ${installExitCode}`);
                        setIsBooting(false);
                        return; // Stop if install fails
                    }
                    onTerminalOutput?.('[System] Dependencies installed.');
                }

                // Start Command Logic
                let startCmd = 'npm';
                let startArgs = ['run', 'start']; // Default

                if (hasPackageJson) {
                    try {
                        const pkg = JSON.parse(files['package.json']);
                        if (pkg.scripts?.dev) {
                            startArgs = ['run', 'dev'];
                        } else if (!pkg.scripts?.start) {
                            // No start scripts found, fallback to serve
                            onTerminalOutput?.('[System] No start script found. Falling back to npx serve.');
                            startCmd = 'npx';
                            startArgs = ['serve', '.'];
                        }
                    } catch (e) {
                        onTerminalOutput?.('[Error] Failed to parse package.json');
                    }
                } else {
                    // Fallback for static projects
                    onTerminalOutput?.('[System] Static project detected. Starting static server...');
                    startCmd = 'npx';
                    startArgs = ['serve', '.'];
                }

                // Kill previous process if exists (though usually handled by unmount/remount logic in a real IDE)
                if (serverProcessRef.current) {
                    serverProcessRef.current.kill();
                }

                onTerminalOutput?.(`[System] Starting server: ${startCmd} ${startArgs.join(' ')}`);
                const startProcess = await webContainerInstance.spawn(startCmd, startArgs);
                serverProcessRef.current = startProcess;

                startProcess.output.pipeTo(new WritableStream({
                    write(data) { onTerminalOutput?.(data); }
                }));

            } catch (error: any) {
                console.error("WebContainer Error:", error);
                onTerminalOutput?.(`[Critical Error] ${error.message}`);
                setIsBooting(false);
            }
        };

        // Only boot once files are available. 
        // In a real app, we might want to debounce this or only run on explicit "Run" button.
        if (Object.keys(files).length > 0 && !iframeUrl) {
             boot();
        }
        
    }, []); // Run once on mount for this specific implementation to establish the container

    // --- 3. Hot Updates (Write File) ---
    // This allows the editor to push changes without restarting the container
    const writeFile = useCallback(async (path: string, content: string) => {
        if (webContainerInstance) {
            try {
                await webContainerInstance.fs.writeFile(path, content);
                // No log here to keep terminal clean, or optional verbose log
            } catch (e) {
                console.error("Failed to write file:", e);
            }
        }
    }, []);

    // Watch for file changes in the parent component and sync them
    useEffect(() => {
        if (!webContainerInstance || !isReady) return;

        // Simple diffing or just writing all (optimally we only write changed files)
        // For this implementation, we expose the writeFile function, but we can also
        // auto-sync specific known files if the 'files' prop updates.
        Object.entries(files).forEach(([path, content]) => {
            writeFile(path, content);
        });
    }, [files, isReady, writeFile]);

    const refresh = () => {
        // Trigger a re-render of iframe if needed, or restart process
        setIframeUrl(prev => prev ? prev + '' : null);
    };

    return { iframeUrl, isReady, refresh, isBooting };
};
