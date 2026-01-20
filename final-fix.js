import fs from 'fs';
import path from 'path';

const agiPath = path.join(process.cwd(), 'src', 'components', 'AGISingularityInterface.tsx');

if (fs.existsSync(agiPath)) {
    let content = fs.readFileSync(agiPath, 'utf8');

    // 1. Clean up the double-wrapped braces
    content = content.replace(/\{'\{'>>>'\}'\}/g, "{'>>>'}");
    content = content.replace(/\{\{">>>"\}\}/g, "{'>>>'}");

    // 2. Ensure the Icons import is correct
    content = content.replace(/from\s+['"]\.\/components\/Icons['"]/g, 'from "@/components/Icons"');

    fs.writeFileSync(agiPath, content);
    console.log("✅ AGISingularityInterface.tsx has been cleaned and corrected.");
}