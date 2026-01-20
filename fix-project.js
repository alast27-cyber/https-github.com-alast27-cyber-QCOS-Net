import fs from 'fs';
import path from 'path';

const root = process.cwd();
const componentsDir = path.join(root, 'src', 'components');

console.log("🛠️ Starting Global Project Auto-Fix...");

// 1. Fix AGISingularityInterface JSX Syntax (the '>>>' error)
const agiPath = path.join(componentsDir, 'AGISingularityInterface.tsx');
if (fs.existsSync(agiPath)) {
    let content = fs.readFileSync(agiPath, 'utf8');
    
    // Replace raw >>> with {'>>>'}
    const fixedJSX = content.replace(/>>>/g, "{'>>>'}");
    
    if (content !== fixedJSX) {
        fs.writeFileSync(agiPath, fixedJSX);
        console.log("✅ Fixed JSX syntax error (>>>) in AGISingularityInterface.tsx");
    }
}

// 2. Fix Broken Imports in all components
if (fs.existsSync(componentsDir)) {
    const files = fs.readdirSync(componentsDir);
    
    files.forEach(file => {
        if (file.endsWith('.tsx') || file.endsWith('.ts')) {
            const filePath = path.join(componentsDir, file);
            let content = fs.readFileSync(filePath, 'utf8');
            
            // Check for the specific broken relative path
            const brokenImport = 'from "./components/Icons"';
            const fixedImport = 'from "@/components/Icons"';
            
            if (content.includes(brokenImport)) {
                const updatedContent = content.split(brokenImport).join(fixedImport);
                fs.writeFileSync(filePath, updatedContent);
                console.log(`✅ Fixed Icons import in: ${file}`);
            }
        }
    });
}

console.log("\n✨ All known errors have been patched.");
console.log("🚀 Run 'npm run dev' to start the dashboard.");