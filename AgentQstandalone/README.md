# AgentQ Standalone Interface

This is the standalone, high-dimensional chat interface for **AgentQ**, the supreme technical architect of QCOS.

## Features
- **Isolated Operation**: Runs independently of the main QCOS dashboard.
- **Quantum Entanglement (QAPI)**: Maintains a real-time state synchronization with the main app via `BroadcastChannel` and unified services.
- **Production Ready**: Optimized Vite build with code splitting and lazy loading.

## Deployment to Vercel

To deploy this standalone app successfully, follow these exact settings in the Vercel Dashboard:

### 1. Project Configuration
- **Framework Preset**: `Vite`
- **Root Directory**: `.` (Keep as the project root, **NOT** `AgentQstandalone`)

### 2. Build & Development Settings
- **Build Command**: `npm run build:standalone`
- **Output Directory**: `AgentQstandalone/dist`

### 3. Environment Variables
Ensure `GEMINI_API_KEY` is set in your Vercel project environment variables if you want AgentQ to maintain its cognitive functions in production.

## Why keep Root Directory as `.`?
The standalone app imports shared components and types from the `/src` directory. By keeping the root as the project root, Vercel can access those files during the build process.

## Architecture: The QAPI
AgentQ can be reached from other apps using the `QAPI` bridge:

```typescript
import { QAPI } from './AgentQstandalone/QAPI';

// Entangle your app
QAPI.entangle('my-app-id');

// Subscribe to events
QAPI.subscribe((event) => console.log(event));
```

"Information is not just logic, it is physical."
