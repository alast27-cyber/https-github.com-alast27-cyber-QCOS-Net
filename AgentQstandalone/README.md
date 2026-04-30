# AgentQ Standalone Interface

This is the standalone, high-dimensional chat interface for **AgentQ**, the supreme technical architect of QCOS.

## Features
- **Isolated Operation**: Runs independently of the main QCOS dashboard.
- **Quantum Entanglement (QAPI)**: Maintains a real-time state synchronization with the main app via `BroadcastChannel` and unified services.
- **Production Ready**: Optimized Vite build with code splitting and lazy loading.

## Deployment to Vercel

### Option 1: Automatic (via Subdirectory)
Point your Vercel project to this repository and set the **Root Directory** to `AgentQstandalone`.

### Option 2: Manual Build
Run the following from the project root:
```bash
npm run build:standalone
```
The output will be in `/dist-standalone`.

## Architecture: The QAPI
AgentQ can be reached from other apps using the `QAPI` bridge:

```typescript
import { QAPI } from './AgentQstandalone/QAPI';

// Entangle your app
QAPI.entangle('my-app-id');

// Dispatch commands
const result = await QAPI.dispatch('Identify system anomalies');
```

"Information is not just logic, it is physical."
