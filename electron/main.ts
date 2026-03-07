import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';
import fs from 'fs';

// --- Replicate __dirname for ES Modules ---
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let mainWindow: BrowserWindow | null = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    title: "AGENT Q :: NEURAL TERMINAL",
    backgroundColor: '#000000',
    webPreferences: {
      // Points to your preload file
      preload: path.join(__dirname, 'preload.js'), 
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // During development, it loads from the React dev server
  // In production, it would load the build/index.html
  const startUrl = process.env.NODE_ENV === 'development' 
    ? 'http://localhost:3000' 
    : `file://${path.join(__dirname, '../build/index.html')}`;

  mainWindow.loadURL(startUrl);

  // Open DevTools automatically in dev mode to see IPC logs
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }
}

// --- IPC HANDLERS (The Bridge) ---

// 1. Physical Pillar Density Check
ipcMain.handle('monitor-pillars', async () => {
  const pillarPath = "C:\\Program Files\\QCOS\\ChipsBrowser\\bin\\";
  const files = ['ChipsBrowser.exe', 'AgentCommandConsole.exe', 'EKSBridgeService.exe', 'qlang.exe'];
  
  try {
    let results = files.map(file => {
      const fullPath = path.join(pillarPath, file);
      const exists = fs.existsSync(fullPath);
      const size = exists ? fs.statSync(fullPath).size : 0;
      // Check if size is roughly 150MB (157286400 bytes)
      const isGrounded = size >= 150000000; 
      return { file, exists, isGrounded };
    });

    const allGrounded = results.every(r => r.isGrounded);
    return { 
      status: allGrounded ? "SYNCED" : "UNSTABLE", 
      details: results 
    };
  } catch (error) {
    return { status: "ERROR", message: String(error) };
  }
});

// 2. Real PowerShell Command Execution
ipcMain.handle('terminal-exec', async (event, command: string) => {
  return new Promise((resolve) => {
    // Basic safety: only allow specific QCOS commands or simple echoes
    const allowedCommands = ['scan', 'whoami', 'status', 'dir'];
    const baseCmd = command.split(' ')[0].toLowerCase();

    if (!allowedCommands.includes(baseCmd)) {
      resolve(`ACCESS DENIED: Command '${baseCmd}' is not authorized.`);
      return;
    }

    exec(`powershell -Command "${command}"`, (error, stdout, stderr) => {
      if (error) resolve(`ERROR: ${stderr || error.message}`);
      resolve(stdout || "COMMAND EXECUTED SUCCESSFULLY.");
    });
  });
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});