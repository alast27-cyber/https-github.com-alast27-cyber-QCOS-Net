const fs = require('fs');
const path = require('path');

const srcDir = '/src/components';
const destDir = '/ChipsBackOffice';

const filesToCopy = [
    'CHIPSBackOffice.tsx',
    'CHIPSStoreAdmin.tsx',
    'MergedEvolutionPanel.tsx',
    'ChipsEconomy.tsx',
    'SecurityMonitorAndSimulator.tsx',
    'QuantumCognitiveArchitecture.tsx',
    'CHIPSGatewayAdmin.tsx',
    'ChipsDevPlatform.tsx',
    'MemoryMatrix.tsx',
    'SyntaxHighlighter.tsx',
    'GrandUniverseSimulator.tsx',
    'HolographicPreviewRenderer.tsx',
    'MonacoEditorWrapper.tsx',
    'LivePreviewFrame.tsx',
    'ChimeraCoreStatus.tsx',
    'GlassPanel.tsx',
    'LoadingSkeleton.tsx',
    'Icons.tsx'
];

if (!fs.existsSync(destDir)) {
    fs.mkdirSync(destDir, { recursive: true });
}

filesToCopy.forEach(file => {
    const srcPath = path.join(srcDir, file);
    const destPath = path.join(destDir, file);
    if (fs.existsSync(srcPath)) {
        fs.copyFileSync(srcPath, destPath);
        console.log(`Copied: ${file}`);
    } else {
        console.warn(`File not found: ${srcPath}`);
    }
});

fs.writeFileSync(path.join(destDir, 'README.md'), `# CHIPS Back Office Files\n\nThis directory contains all components and files that run the CHIPS Back Office from the Chips - Dash panel.`);
console.log('Copy operation completed.');
