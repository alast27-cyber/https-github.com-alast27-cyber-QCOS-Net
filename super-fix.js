import fs from 'fs';
import path from 'path';

const fileToFix = path.join(process.cwd(), 'src', 'components', 'AGISingularityInterface.tsx');

if (fs.existsSync(fileToFix)) {
    let content = fs.readFileSync(fileToFix, 'utf8');

    console.log("🛠️  Applying high-precision patch to AGISingularityInterface...");

    // 1. Fix the Import: Matches any variation of from "./components/Icons" or './components/Icons'
    const updatedContent = content.replace(
        /from\s+['"]\.\/components\/Icons['"]/g, 
        'from "@/components/Icons"'
    );

    // 2. Double check JSX: Ensure >>> is escaped
    const finalContent = updatedContent.replace(/>>>/g, "{'>>>'}");

    if (content !== finalContent) {
        fs.writeFileSync(fileToFix, finalContent);
        console.log("✅ SUCCESSFULLY PATCHED: Import paths and JSX syntax.");
    } else {
        console.log("ℹ️  No changes needed or patterns didn't match.");
    }
} else {
    console.error("❌ ERROR: Could not find the file at src/components/AGISingularityInterface.tsx");
}