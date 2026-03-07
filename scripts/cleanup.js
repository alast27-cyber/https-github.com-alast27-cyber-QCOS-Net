
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

// Files to be deleted from the root directory (now moved to src/)
const filesToDelete = [
  'App.tsx',
  'index.tsx',
  'types.ts',
  'constants.ts',
  'index.css',
  // Old generated files or backups
  'index-1.html', 'index-1.tsx', 'index (1).tsx', 'App-1.tsx', 'App (1).tsx',
  'metadata-1.json', 'package (1).json', 'gitignore (1).txt',
  '[full_path_of_file_1].txt', '[full_path_of_file_2].txt'
];

// Directories to be removed from root if they exist (now in src/)
const dirsToDelete = [
  'components',
  'context',
  'hooks',
  'services',
  'utils',
  'qllm'
];

console.log('Starting QCOS Project Cleanup...');

// Delete Files
filesToDelete.forEach(file => {
  const filePath = path.join(rootDir, file);
  if (fs.existsSync(filePath)) {
    try {
      fs.unlinkSync(filePath);
      console.log(`Deleted file: ${file}`);
    } catch (err) {
      console.error(`Error deleting ${file}: ${err.message}`);
    }
  }
});

// Delete Directories (only if empty or we want to force cleanup)
dirsToDelete.forEach(dir => {
    const dirPath = path.join(rootDir, dir);
    if (fs.existsSync(dirPath)) {
        try {
            fs.rmSync(dirPath, { recursive: true, force: true });
            console.log(`Deleted directory: ${dir}`);
        } catch (err) {
            console.error(`Error deleting directory ${dir}: ${err.message}`);
        }
    }
});

console.log('Cleanup complete. Project structure optimized for src/ architecture.');
