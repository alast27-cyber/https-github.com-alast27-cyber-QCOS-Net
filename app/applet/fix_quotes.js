const fs = require('fs');
let c = fs.readFileSync('/app/applet/GrandUniverseSimulator/StandaloneApp.tsx', 'utf8');
c = c.replace('bg-[url("https://grainy-gradients.vercel.app/noise.svg")]', "bg-[url('https://grainy-gradients.vercel.app/noise.svg')]");
fs.writeFileSync('/app/applet/GrandUniverseSimulator/StandaloneApp.tsx', c);
