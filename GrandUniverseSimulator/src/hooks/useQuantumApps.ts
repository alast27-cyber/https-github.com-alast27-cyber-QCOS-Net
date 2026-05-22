import { useState, useCallback, useEffect } from 'react';
import { AppDefinition, URIAssignment, LogEntry, UIStructure } from '../types';
import { 
    CpuChipIcon, GlobeIcon, BuildingFarmIcon, UsersIcon, BoxIcon, FlaskConicalIcon, 
    ShieldCheckIcon, ActivityIcon, BeakerIcon, Share2Icon, GitBranchIcon, AtomIcon, 
    BrainCircuitIcon, AcademicCapIcon, CurrencyDollarIcon, PuzzlePieceIcon, ArrowTrendingUpIcon,
    MicIcon, DownloadCloudIcon, PlayIcon, BriefcaseIcon, ZapIcon, TerminalIcon
} from '../components/Icons';
import { useToast } from '../context/ToastContext';

const availableIconsData = [AtomIcon, GitBranchIcon, ShieldCheckIcon, FlaskConicalIcon, BrainCircuitIcon, BoxIcon];

const initialAppsData: Omit<AppDefinition, 'component'>[] = [
    {
        id: 'qos-kernel-manager',
        name: 'QOS Kernel Manager',
        description: 'Direct interface for the Quantum Operating System Kernel. Manage priority scheduling, HAL integrity, and error mitigation protocols.',
        icon: TerminalIcon,
        status: 'installed',
        isCustom: false,
        q_uri: 'CHIPS://qos.kernel.sys/console',
        https_url: 'https://qcos.sys/kernel'
    },
    {
        id: 'der-grid-optimizer',
        name: 'DER Grid Optimizer',
        description: 'Quantum-optimized Distributed Energy Resource management for real-time grid balancing. Features VQE load optimization and IEC 61850 telemetry.',
        icon: ZapIcon,
        status: 'installed',
        isCustom: false,
        q_uri: 'CHIPS://der-opt.qcos.apps/main',
        https_url: 'https://qcos.apps.web/der-opt'
    },
    { id: 'q-vox', name: 'Quantum Voice Chat', description: 'Quantum-Secured Voice Channel using BB84 protocol for unhackable communication.', icon: MicIcon, status: 'installed', q_uri: 'CHIPS://q-vox.qcos.apps/main', https_url: 'https://qcos.apps.web/q-vox' },
    { id: 'chimera-browser', name: 'Chimera Browser', description: 'AI-Native browser for the CHIPS network and public web.', icon: GlobeIcon, status: 'installed' },
    { id: 'chips-browser-sdk', name: 'Chips Browser SDK', description: 'Complete source code and installer for the Chips Quantum Browser. Build your own quantum-native navigation tools. (SDK Installer)', icon: DownloadCloudIcon, status: 'available' },
    { id: 'quantum-swine-intelligence', name: 'Quantum Swine Intelligence', description: 'An ecosystem of quantum-powered apps for the global swine industry.', icon: CpuChipIcon, status: 'installed' },
    { id: 'global-swine-foresight', name: 'Global Swine Foresight', description: 'Strategic predictive analytics for global swine markets.', icon: ArrowTrendingUpIcon, status: 'installed' },
    { id: 'philippine-swine-resilience', name: 'Philippine Swine Resilience', description: 'Actionable quantum insights for the Philippine swine industry.', icon: BuildingFarmIcon, status: 'installed' },
    { id: 'pighaven-consumer-trust', name: 'PigHaven Consumer Trust', description: 'Quantum-secured traceability and market insights for consumers.', icon: UsersIcon, status: 'installed' },
    { id: 'mol-sim', name: 'Molecular Simulation Toolkit', description: 'Simulate complex molecular interactions.', icon: FlaskConicalIcon, status: 'installed' },
    { id: 'generic-solver', name: 'Quantum Optimization Solver', description: 'General-purpose solver for optimization problems.', icon: PuzzlePieceIcon, status: 'installed' },
    { id: 'qkd-sim', name: 'QKD Simulator', description: 'Simulate BB84 and other quantum key distribution protocols.', icon: ShieldCheckIcon, status: 'installed' },
    { id: 'qmc-finance', name: 'Quantum Monte Carlo: Finance', description: 'Perform complex financial risk analysis using quantum-accelerated Monte Carlo methods.', icon: CurrencyDollarIcon, status: 'installed' },
    { id: 'q-biomed', name: 'Q-BioMed: Drug Discovery', description: 'Accelerate drug discovery by simulating molecular structures on a quantum level.', icon: BeakerIcon, status: 'installed' },
    { id: 'qnet-viz', name: 'Quantum Network Visualizer', description: 'Monitor and visualize entanglement distribution across the quantum network.', icon: Share2Icon, status: 'installed' },
    { id: 'vqe-toolkit', name: 'VQE Toolkit', description: 'Use the Variational Quantum Eigensolver to find molecular ground states.', icon: GitBranchIcon, status: 'installed' },
    { id: 'q-denoise', name: 'Q-DeNoise Signal Processor', description: 'Utilizes quantum error correction codes to denoise classical signals, improving fidelity for scientific and communication data.', icon: ActivityIcon, status: 'installed' },
];

export const useQuantumApps = (
    addLog: (level: LogEntry['level'], msg: string) => void, 
    handlePanelSelect: (panelId: string) => void
) => {
    const [marketApps, setMarketApps] = useState<Omit<AppDefinition, 'component'>[]>(() => {
        const saved = localStorage.getItem('qcos_market_apps');
        if (saved) {
            try {
                const parsed = JSON.parse(saved);
                const hydrated = parsed.map((savedApp: any) => {
                    const staticApp = initialAppsData.find(a => a.id === savedApp.id);
                    if (staticApp) {
                        return { 
                            ...staticApp, 
                            status: savedApp.status,
                            icon: staticApp.icon
                        };
                    }
                    return {
                        ...savedApp,
                        icon: availableIconsData[Math.floor(Math.random() * availableIconsData.length)]
                    };
                });
                const existingIds = new Set(hydrated.map((h: any) => h.id));
                const newStaticApps = initialAppsData.filter(a => !existingIds.has(a.id));
                return [...hydrated, ...newStaticApps];
            } catch (e) {
                return initialAppsData;
            }
        }
        return initialAppsData;
    });

    const [uriAssignments, setUriAssignments] = useState<URIAssignment[]>(() => {
        const saved = localStorage.getItem('qcos_uri_assignments');
        return saved ? JSON.parse(saved) : [
            {
                appName: 'Quantum Voice Chat',
                q_uri: 'CHIPS://q-vox.qcos.apps/main',
                https_url: 'https://qcos.apps.web/q-vox',
                timestamp: new Date().toLocaleTimeString()
            }
        ];
    });
    
    const { addToast } = useToast();

    useEffect(() => {
        const serializableApps = marketApps.map(({ icon, ...rest }) => rest);
        localStorage.setItem('qcos_market_apps', JSON.stringify(serializableApps));
    }, [marketApps]);

    useEffect(() => {
        localStorage.setItem('qcos_uri_assignments', JSON.stringify(uriAssignments));
    }, [uriAssignments]);
    
    const handleInstallApp = useCallback((id: string) => {
        const appToInstall = marketApps.find(app => app.id === id);
        if (!appToInstall) return;

        addLog('INFO', `Download initiated for ${appToInstall.name} package...`);
        setMarketApps(prevApps => prevApps.map(app => 
            app.id === id ? { ...app, status: 'downloading' } : app
        ));

        setTimeout(() => {
            addLog('INFO', `Verifying package integrity for ${appToInstall.name}...`);
            setTimeout(() => {
                setMarketApps(prevApps => prevApps.map(app => 
                    app.id === id ? { ...app, status: 'installing' } : app
                ));
                setTimeout(() => {
                    setMarketApps(prevApps => prevApps.map(app => 
                        app.id === id ? { ...app, status: 'installed' } : app
                    ));
                    addLog('SUCCESS', `App successfully installed: ${appToInstall.name}`);
                    addToast(`${appToInstall.name} installed successfully.`, 'success');
                }, 2000);
            }, 1500);
        }, 2000);
    }, [addLog, marketApps, addToast]);

    const handleFullDeployment = useCallback((appDetails: { name: string; description: string; code: string; uiStructure?: UIStructure }) => {
        const appName = appDetails.name || "Untitled App";
        const appDescription = appDetails.description || "User-deployed application.";
        const appSlug = appName.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '') || `app-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const q_uri = `CHIPS://${appSlug}.qcos.apps/main`;
        const https_url = `https://qcos.apps.web/${appSlug}`;
        const timestamp = new Date().toLocaleTimeString('en-GB', { hour12: false });
        
        setUriAssignments(prev => [{ appName, q_uri, https_url, timestamp }, ...prev]);
        
        const newApp: Omit<AppDefinition, 'component'> = {
            id: appSlug, 
            name: appName,
            description: appDescription,
            icon: availableIconsData[Math.floor(Math.random() * availableIconsData.length)],
            status: 'installed', 
            isCustom: true,
            q_uri,
            https_url,
            code: appDetails.code,
            uiStructure: appDetails.uiStructure
        };

        setMarketApps(prev => {
            const exists = prev.some(a => a.id === appSlug);
            if (exists) return prev.map(a => a.id === appSlug ? newApp : a);
            return [newApp, ...prev];
        });

        addLog('SUCCESS', `Deployment successful. "${appName}" is live.`);
        addToast(`Deployed ${appName} to CHIPS Network.`, 'success');
    }, [addLog, addToast]);

    const handleTogglePublic = useCallback((appId: string) => {
        const app = marketApps.find(a => a.id === appId);
        if (!app) return;
        const isCurrentlyPublic = uriAssignments.some(a => a.appName === app.name);
        if (isCurrentlyPublic) {
            setUriAssignments(prev => prev.filter(a => a.appName !== app.name));
            addToast(`Public access revoked for ${app.name}.`, 'warning');
        } else {
            const appSlug = app.name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');
            const q_uri = app.q_uri || `CHIPS://${appSlug}.qcos.apps/main`;
            const https_url = app.https_url || `https://qcos.apps.web/${appSlug}`;
            const timestamp = new Date().toLocaleTimeString('en-GB', { hour12: false });
            setUriAssignments(prev => [{ appName: app.name, q_uri, https_url, timestamp }, ...prev]);
            addToast(`Public access granted: ${https_url}`, 'success');
        }
    }, [marketApps, uriAssignments, addToast]);

    return { marketApps, uriAssignments, handleInstallApp, handleFullDeployment, handleTogglePublic };
};