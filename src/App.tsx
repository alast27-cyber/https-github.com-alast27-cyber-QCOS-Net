import React, { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ToastProvider, useToast } from './context/ToastContext';
import { SimulationProvider, useSimulation } from './context/SimulationContext';
import { useQuantumApps } from './hooks/useQuantumApps';
import { useAgentQ } from './hooks/useAgentQ';
import { useAdminChat } from './hooks/useAdminChat';
import { usePanelContent } from './hooks/usePanelContent';
import { useVoiceCommands } from './hooks/useVoiceCommands';

// Core Components
import LoginScreen from './components/LoginScreen';
import ResourceSteward from './components/ResourceSteward';
import ToastContainer from './components/ToastContainer';
import AgentQ from './components/AgentQ';
import AdminChat from './components/AdminChat';
import AnimatedBackground from './components/AnimatedBackground';
import EditorView from './components/EditorView';
import GlassPanel from './components/GlassPanel';
import FullScreenSwitcher from './components/FullScreenSwitcher';
import OnboardingFlow from './components/OnboardingFlow'; 
import ChipsBrowserInstaller from './components/ChipsBrowserInstaller'; 

// Icons
import { 
  XIcon, BrainCircuitIcon, CpuChipIcon, ServerCogIcon, GlobeIcon, 
  ShieldCheckIcon, LayersIcon, BeakerIcon, AtomIcon, GridIcon,
  RefreshCwIcon, ActivityIcon, SparklesIcon, NetworkIcon, 
  RocketLaunchIcon, BoxIcon
} from './components/Icons';

// Utils & Types
import { initialLogs, initialSystemHealth, getPanelMetadata, FaceData } from './utils/dashboardConfig';
import { LogEntry } from './types';

// Panel Components
import IAIKernelStatus from './components/IAIKernelStatus';
import DistributedCognitiveArchitecture from './components/DistributedCognitiveArchitecture'; 
import UtilityHubPanel from './components/UtilityHubPanel';
import QLangCoreEvolutionPanel from './components/QLangCoreEvolutionPanel';
import SecurityMonitorAndSimulator from './components/SecurityMonitorAndSimulator';
import QPUHealth from './components/QPUHealth';
import QuantumDataIngestion from './components/QuantumDataIngestion';
import SystemDiagnostic from './components/SystemDiagnostic';
import CHIPSBackOffice from './components/CHIPSBackOffice'; 
import NeuralProgrammingPanel from './components/NeuralProgrammingPanel';
import QuantumCognitiveArchitecture from './components/QuantumCognitiveArchitecture';
import QuantumDeepLearning from './components/QuantumDeepLearning';
import QuantumMachineLearning from './components/QuantumMachineLearning';
import QBioMedDrugDiscovery from './components/QBioMedDrugDiscovery';
import MolecularSimulationToolkit from './components/MolecularSimulationToolkit';
import QOSKernelConsole from './components/QOSKernelConsole';
import CHIPSBrowserSDK from './components/CHIPSBrowserSDK'; 
import QuantumEngineeringDesign from './components/QuantumEngineeringDesign';

const QCOS_VERSION = 4.5;

const EntanglementBeams: React.FC<{ active: boolean; isLinked?: boolean }> = ({ active, isLinked = false }) => {
    if (!active) return null;
    return (
        <svg className="fixed inset-0 w-full h-full pointer-events-none z-10 opacity-40">
            <defs>
                <linearGradient id="entangleGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="rgba(0, 255, 255, 0)" />
                    <stop offset="50%" stopColor="rgba(0, 255, 255, 0.4)" />
                    <stop offset="100%" stopColor="rgba(0, 255, 255, 0)" />
                </linearGradient>
            </defs>
            <path d="M 0,200 Q 400,200 1200,500" stroke="url(#entangleGrad)" strokeWidth="1" fill="none" className="animate-pulse" />
            <path d="M 0,800 Q 400,800 1200,500" stroke="url(#entangleGrad)" strokeWidth="1" fill="none" className="animate-pulse" style={{ animationDelay: '0.5s' }} />
        </svg>
    );
};

const DashboardContent: React.FC = () => {
  const { isAuthenticated, adminLevel } = useAuth();
  const { addToast } = useToast();
  
  // FIX: Access qllm and entanglementMesh safely with defaults to prevent the 'efficiencyBoost' undefined crash
  const simulation = useSimulation();
  const systemStatus = simulation.systemStatus || initialSystemHealth;
  const qllm = simulation.qllm || { efficiencyBoost: 0 };
  const entanglementMesh = simulation.entanglementMesh || { isUniverseLinkedToQLang: false };
  const startAllSimulations = simulation.startAllSimulations || (() => {});
  
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [isInstalled, setIsInstalled] = useState(false); 

  const [tilt, setTilt] = useState({ x: 0, y: 0 });
  const dashboardRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!dashboardRef.current) return;
    const { innerWidth: w, innerHeight: h } = window;
    const x = (e.clientX - w / 2) / (w / 2);
    const y = (e.clientY - h / 2) / (h / 2);
    setTilt({ x: x * 2.5, y: -y * 2.5 }); 
  }, []);

  useEffect(() => {
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [handleMouseMove]);

  useEffect(() => {
    if (isAuthenticated) {
        const installed = localStorage.getItem('chips_browser_installed') === 'true';
        setIsInstalled(installed);

        if (installed) {
             const hasOnboarded = localStorage.getItem('qcos_onboarded') === 'true';
             if (!hasOnboarded) {
                 setShowOnboarding(true);
             } else {
                  const timer = setTimeout(() => {
                       startAllSimulations();
                   }, 1000);
                   return () => clearTimeout(timer);
             }
        }
    }
  }, [isAuthenticated, startAllSimulations]);
  
  const handleInstallationComplete = () => {
      localStorage.setItem('chips_browser_installed', 'true');
      setIsInstalled(true);
      setShowOnboarding(true); 
  };

  const handleOnboardingComplete = () => {
      setShowOnboarding(false);
      localStorage.setItem('qcos_onboarded', 'true');
      startAllSimulations();
  };

  const [dashboardDeck, setDashboardDeck] = useState<'primary' | 'secondary'>('primary');
  
  const [flipP1, setFlipP1] = useState(false);
  const [flipP2, setFlipP2] = useState(false);
  const [flipP3, setFlipP3] = useState(false);

  const [flipS1, setFlipS1] = useState(false);
  const [flipS2, setFlipS2] = useState(false);
  const [flipS3, setFlipS3] = useState(false);
  
  const [logs, setLogs] = useState<LogEntry[]>(initialLogs);
  const [maximizedPanelId, setMaximizePanelId] = useState<string | null>(null);
  const [agentWeaverInput, setAgentWeaverInput] = useState('');
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const [isSwitcherOpen, setIsSwitcherOpen] = useState(false);
  const [connectedAppId, setConnectedAppId] = useState<string | null>(null);

  const [activeDataStreams] = useState<string[]>([
      'GQML Stream (Live)', 'Quantum Entanglement Feed', 'LHC Collision Data', 'NASA Exoplanet Archive'
  ]);

  // FIX: Generate unique IDs for log entries to prevent the duplicate key error
  const addLog = useCallback((level: LogEntry['level'], msg: string) => {
    const id = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    setLogs(prev => [{ id, time: new Date().toLocaleTimeString(), level, msg }, ...prev].slice(0, 50));
  }, []);

  const handlePanelSelect = useCallback((panelId: string) => {
      setMaximizePanelId(panelId);
      setIsSwitcherOpen(false);
  }, []);

  const { marketApps, uriAssignments, handleInstallApp, handleFullDeployment, handleTogglePublic } = useQuantumApps(addLog, handlePanelSelect);
  const { isOpen: isAdminChatOpen, onToggle: toggleAdminChat, adminChatProps } = useAdminChat();

  const panelInfoMap = useMemo(() => {
      const metadata = getPanelMetadata(QCOS_VERSION);
      const map: any = {};
      Object.values(metadata).forEach((face: any) => {
          if (face.panels) {
              face.panels.forEach((p: any) => {
                  map[p.id] = { title: p.title, description: p.description };
              });
          }
      });
      return map;
  }, []);

  const { isAgentQOpen, toggleAgentQ, generateApp, updateAppForChips, debugAndFixApp, editCode, agentQProps } = useAgentQ({
      focusedPanelId: maximizedPanelId || 'agentq-core',
      panelInfoMap,
      qcosVersion: QCOS_VERSION,
      systemHealth: { ...initialSystemHealth, ...systemStatus },
      onDashboardControl: () => {}
  });

  const { listeningState, toggleListening, isSupported } = useVoiceCommands([
      { command: ['open agent q', 'agent q', 'help'], callback: () => !isAgentQOpen && toggleAgentQ() },
      { command: ['reset view', 'minimize'], callback: () => setMaximizePanelId(null) },
      { command: ['switch deck', 'next page', 'swap view'], callback: () => setDashboardDeck(prev => prev === 'primary' ? 'secondary' : 'primary') }
  ]);

  const faceData = usePanelContent({
      systemHealth: { ...initialSystemHealth, ...systemStatus },
      isRecalibrating: false,
      isUpgrading: false,
      activeDataStreams, 
      uriAssignments,
      agentWeaverInput,
      onPublishToExchange: handleFullDeployment,
      onConnectToGateway: (source) => addLog('INFO', `Connecting to gateway source: ${source}`),
      logs,
      addLog,
      codebase: {}, 
      onApplyPatch: (f, c) => addLog('SUCCESS', `Patch applied to ${f}`),
      marketApps,
      onInstallApp: handleInstallApp,
      onLaunchApp: (id) => setMaximizePanelId(id),
      onGenerateApp: generateApp,
      onUpdateApp: updateAppForChips,
      onDebugApp: debugAndFixApp,
      onTogglePublic: handleTogglePublic,
      onSendToWeaver: setAgentWeaverInput,
      onCloseSystemDiagnostic: () => setMaximizePanelId(null),
      adminLevel,
      qcosVersion: QCOS_VERSION,
      agentQProps,
      editCode,
      onOpenAppCreator: (p) => {},
      onSimulateApp: (id) => setConnectedAppId(id),
      connectedAppId,
      toggleAgentQ
  });

  if (!isAuthenticated) {
      return (
          <>
            <AnimatedBackground />
            <div className="absolute inset-0 flex items-center justify-center">
              <LoginScreen />
            </div>
            <ToastContainer />
          </>
      );
  }

  if (!isInstalled) {
      return <ChipsBrowserInstaller onInstallComplete={handleInstallationComplete} />;
  }

  if (showOnboarding) {
      return <OnboardingFlow onComplete={handleOnboardingComplete} />;
  }

  if (isEditorOpen) {
      return <EditorView agentQProps={agentQProps} onToggleView={() => setIsEditorOpen(false)} />;
  }

  let maximizedContent = null;
  if (maximizedPanelId) {
      let foundPanel: any = null;
      Object.values(faceData).some((face: FaceData) => {
          foundPanel = face.panels.find(p => p.id === maximizedPanelId);
          return !!foundPanel;
      });

      if (!foundPanel) {
          const directMap: any = {
             'qiai-kernel-status': { title: 'Kernel Node Status', content: <IAIKernelStatus isRecalibrating={false} /> },
             'qiai-cognitive-mesh': { title: 'Neural Topology', content: <DistributedCognitiveArchitecture activeDataStreams={activeDataStreams} /> },
             'utility-hub': { title: 'Service Hub', content: <UtilityHubPanel onMaximizeSubPanel={setMaximizePanelId} /> },
             'security-monitor': { title: 'Security Monitor', content: <SecurityMonitorAndSimulator /> },
             'qpu-health': { title: 'QPU Vitals', content: <QPUHealth systemHealth={{ ...initialSystemHealth, ...systemStatus }} /> },
             'data-ingestion': { title: 'Quantum Data Ingestion', content: <QuantumDataIngestion /> },
             'system-diagnostic': { title: 'System Diagnostics', content: <SystemDiagnostic onClose={() => setMaximizePanelId(null)} /> },
             'q-lang-evolution': { title: 'Evolution Matrix', content: <QLangCoreEvolutionPanel /> },
             'chips-back-office': { title: 'Chips Back Office', content: <CHIPSBackOffice uriAssignments={uriAssignments} marketApps={marketApps} /> },
             'neural-programming': { title: 'Neural Programming', content: <NeuralProgrammingPanel /> },
             'quantum-cognitive-architecture': { title: 'Cognitive Architecture', content: <QuantumCognitiveArchitecture /> },
             'quantum-deep-learning': { title: 'Quantum Deep Learning', content: <QuantumDeepLearning /> },
             'quantum-engineering-design': { title: 'Quantum Engineering Design', content: <QuantumEngineeringDesign /> },
             'quantum-machine-learning': { title: 'Quantum Machine Learning', content: <QuantumMachineLearning /> },
             'q-biomed': { title: 'BioMed Discovery', content: <QBioMedDrugDiscovery /> },
             'mol-sim': { title: 'Molecular Toolkit', content: <MolecularSimulationToolkit /> },
             'qos-kernel-manager': { title: 'QOS Kernel Console', content: <QOSKernelConsole /> },
             'chips-quantum-network': { 
                 title: 'Chips Quantum Browser', 
                 content: <CHIPSBrowserSDK initialApp={undefined} onToggleAgentQ={() => {}} apps={marketApps} onInstallApp={handleInstallApp} /> 
             }
          };
          foundPanel = directMap[maximizedPanelId];
      }
      
      if (foundPanel) {
          maximizedContent = (
              <GlassPanel title={foundPanel.title} onMaximize={() => setMaximizePanelId(null)} isMaximized={true}>
                  {foundPanel.content}
              </GlassPanel>
          );
      }
  }

  const hudVisibility = maximizedPanelId ? 'opacity-0 pointer-events-none' : 'opacity-100 pointer-events-auto';

  return (
    <div ref={dashboardRef} className="relative w-screen h-screen bg-black text-cyan-100 font-mono overflow-hidden flex flex-col perspective-viewport">
      <AnimatedBackground />
      {/* SAFE ACCESS: Null check qllm.efficiencyBoost to prevent production crashes */}
      <EntanglementBeams active={(qllm?.efficiencyBoost || 0) > 1} isLinked={entanglementMesh.isUniverseLinkedToQLang} />
      
      <div className="lattice-overlay"></div>

      <FullScreenSwitcher 
          isOpen={isSwitcherOpen}
          onToggle={() => setIsSwitcherOpen(!isSwitcherOpen)}
          onPanelSelect={handlePanelSelect}
          corePanels={[
            { id: 'agentq-core', title: 'QIAI-IPS Core', icon: BrainCircuitIcon },
            { id: 'chips-quantum-network', title: 'Chips Browser', icon: GlobeIcon },
            { id: 'qcos-core-gateway', title: 'Gateway', icon: ServerCogIcon },
            { id: 'utility-hub', title: 'Vitals', icon: ShieldCheckIcon },
            { id: 'neural-programming', title: 'Neural Prog', icon: LayersIcon },
            { id: 'quantum-engineering-design', title: 'Engineering', icon: AtomIcon },
            { id: 'q-biomed', title: 'Material Sci', icon: BeakerIcon },
          ]}
          appPanels={marketApps.filter(app => app.status === 'installed').map(app => ({ id: app.id, title: app.name, icon: app.icon }))}
          className={`fixed bottom-4 left-[32rem] z-50 pointer-events-auto transition-opacity duration-500 ${hudVisibility}`}
      />

      <div 
        className={`flex-grow h-0 grid grid-cols-12 gap-6 p-6 transition-all duration-700 overflow-hidden z-20 floating-hologram ${hudVisibility}`}
        style={{ transform: `rotateY(${tilt.x}deg) rotateX(${tilt.y}deg)` }}
      >
          <div className="col-span-12 lg:col-span-4 h-full relative" style={{ perspective: '3000px' }}>
              <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${dashboardDeck === 'primary' ? (flipP1 ? 'rotate-y-180' : '') : (flipS1 ? 'rotate-y-180' : '')}`}>
                  <div className={`absolute inset-0 backface-hidden ${dashboardDeck === 'primary' && !flipP1 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                      <GlassPanel onMaximize={() => setMaximizePanelId('agentq-core')} title="QIAI-IPS Cognitive Architecture">
                          <div className="absolute top-2 right-14 z-50">
                            <button onClick={(e) => { e.stopPropagation(); setFlipP1(true); }} className="p-1.5 hover:bg-cyan-500/20 rounded-md border border-cyan-800 text-cyan-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                          </div>
                          {faceData[0].panels.find(p => p.id === 'agentq-core')?.content}
                      </GlassPanel>
                  </div>
                  <div className={`absolute inset-0 backface-hidden rotate-y-180 ${dashboardDeck === 'primary' && flipP1 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                      <GlassPanel onMaximize={() => setMaximizePanelId('q-lang-evolution')} title="Evolution Matrix">
                          <div className="absolute top-2 right-14 z-50">
                            <button onClick={(e) => { e.stopPropagation(); setFlipP1(false); }} className="p-1.5 hover:bg-purple-500/20 rounded-md border border-purple-800 text-purple-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                          </div>
                          <QLangCoreEvolutionPanel />
                      </GlassPanel>
                  </div>
              </div>
          </div>

          <div className="col-span-12 lg:col-span-5 h-full relative" style={{ perspective: '3000px' }}>
               <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${dashboardDeck === 'primary' ? (flipP2 ? 'rotate-y-180' : '') : (flipS2 ? 'rotate-y-180' : '')}`}>
                  <div className={`absolute inset-0 backface-hidden ${dashboardDeck === 'primary' && !flipP2 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                      <GlassPanel onMaximize={() => setMaximizePanelId('chips-quantum-network')} title="Chips Quantum Network">
                          <div className="absolute top-2 right-14 z-50">
                            <button onClick={(e) => { e.stopPropagation(); setFlipP2(true); }} className="p-1.5 hover:bg-blue-500/20 rounded-md border border-blue-800 text-blue-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                          </div>
                          <div className="h-full overflow-hidden">{faceData[1].panels.find(p => p.id === 'chips-quantum-network')?.content}</div>
                      </GlassPanel>
                  </div>
                  <div className={`absolute inset-0 backface-hidden rotate-y-180 ${dashboardDeck === 'primary' && flipP2 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                      <GlassPanel onMaximize={() => setMaximizePanelId('chips-back-office')} title="Chips Back Office">
                          <div className="absolute top-2 right-14 z-50">
                            <button onClick={(e) => { e.stopPropagation(); setFlipP2(false); }} className="p-1.5 hover:bg-purple-500/20 rounded-md border border-purple-800 text-purple-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                          </div>
                          <CHIPSBackOffice uriAssignments={uriAssignments} marketApps={marketApps} />
                      </GlassPanel>
                  </div>
              </div>
          </div>

          <div className="col-span-12 lg:col-span-3 h-full relative" style={{ perspective: '3000px' }}>
              <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${dashboardDeck === 'primary' ? (flipP3 ? 'rotate-y-180' : '') : (flipS3 ? 'rotate-y-180' : '')}`}>
                  <div className={`absolute inset-0 backface-hidden ${dashboardDeck === 'primary' && !flipP3 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                      <GlassPanel onMaximize={() => setMaximizePanelId('qcos-core-gateway')} title="Engine Hub">
                          <div className="absolute top-2 right-14 z-50">
                            <button onClick={(e) => { e.stopPropagation(); setFlipP3(true); }} className="p-1.5 hover:bg-red-500/20 rounded-md border border-red-800 text-red-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                          </div>
                          <div className="h-full overflow-hidden">{faceData[0].panels.find(p => p.id === 'qcos-core-gateway')?.content}</div>
                      </GlassPanel>
                  </div>
                  <div className={`absolute inset-0 backface-hidden rotate-y-180 ${dashboardDeck === 'primary' && flipP3 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                      <GlassPanel onMaximize={() => setMaximizePanelId('utility-hub')} title="Utility Nexus">
                          <div className="absolute top-2 right-14 z-50">
                            <button onClick={(e) => { e.stopPropagation(); setFlipP3(false); }} className="p-1.5 hover:bg-green-500/20 rounded-md border border-green-800 text-green-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                          </div>
                          <div className="h-full overflow-hidden">
                            <UtilityHubPanel onMaximizeSubPanel={setMaximizePanelId} />
                          </div>
                      </GlassPanel>
                  </div>
              </div>
          </div>
      </div>

      {maximizedPanelId && (
          <div className="absolute inset-4 z-[90] bg-black/95 backdrop-blur-3xl border border-cyan-500/30 rounded-2xl shadow-2xl animate-fade-in-up flex flex-col overflow-hidden">
              <div className="absolute top-4 right-4 z-[95]">
                  <button onClick={() => setMaximizePanelId(null)} className="p-2 bg-black/50 hover:bg-red-900/50 rounded-full text-cyan-400 border border-cyan-800 transition-all">
                      <XIcon className="w-6 h-6" />
                  </button>
              </div>
              {maximizedContent}
          </div>
      )}

      <div className={`h-16 flex-shrink-0 flex items-center justify-between px-8 bg-black/60 border-t border-cyan-900/30 z-50 transition-opacity duration-500 backdrop-blur-md ${hudVisibility}`}>
          <div className="flex items-center gap-6">
              <ResourceSteward listeningState={listeningState} onToggleListen={toggleListening} isVoiceSupported={isSupported} />
              <div className="h-8 w-px bg-cyan-900/40"></div>
              <button onClick={() => setIsEditorOpen(true)} className="holographic-projection px-5 py-2 text-[10px] font-black uppercase tracking-widest bg-cyan-900/10 border-cyan-500/40 rounded-full hover:bg-cyan-500/20 transition-all">
                  Source Nexus
              </button>
              
              <button 
                onClick={() => setDashboardDeck(prev => prev === 'primary' ? 'secondary' : 'primary')}
                className={`holographic-projection px-5 py-2 text-[10px] font-black uppercase tracking-widest border rounded-full transition-all flex items-center gap-2 ${dashboardDeck === 'secondary' ? 'bg-purple-900/20 border-purple-500/40 text-purple-300' : 'bg-cyan-900/10 border-cyan-500/40 text-cyan-300'}`}
              >
                 <GridIcon className="w-3 h-3" />
                 {dashboardDeck === 'primary' ? 'Shift to Science Deck' : 'Shift to System Deck'}
              </button>
          </div>

          <div className="flex items-center gap-6">
               <div className="flex gap-6 text-[10px] font-mono text-cyan-600 bg-black/40 px-6 py-2 rounded-full border border-cyan-900/30">
                   <div className="flex items-center gap-2">
                       <CpuChipIcon className="w-3 h-3" />
                       {/* SAFE ACCESS: Null check neuralLoad for performance monitoring */}
                       LOAD: <span className={(systemStatus?.neuralLoad || 0) > 80 ? 'text-red-400' : 'text-green-400'}>{(systemStatus?.neuralLoad || 0).toFixed(1)}%</span>
                   </div>
                   <div className="flex items-center gap-2">
                       <ActivityIcon className="w-3 h-3" />
                       DIM_SHIFT: <span className="text-white">ACTIVE</span>
                   </div>
               </div>

              <AgentQ 
                isOpen={isAgentQOpen} 
                onToggleOpen={toggleAgentQ} 
                triggerClassName={`fixed bottom-4 left-[24rem] z-50 pointer-events-auto group transition-opacity duration-500 ${hudVisibility}`}
                {...agentQProps} 
              />
          </div>
      </div>

      <AdminChat 
        isOpen={isAdminChatOpen} 
        onToggle={toggleAdminChat} 
        triggerClassName={`fixed bottom-4 left-[28rem] z-50 pointer-events-auto group transition-opacity duration-500 ${hudVisibility}`}
        {...adminChatProps} 
      />

      <ToastContainer />
      
      <style>{`
        .transform-style-preserve-3d { transform-style: preserve-3d; }
        .backface-hidden { backface-visibility: hidden; }
        .rotate-y-180 { transform: rotateY(180deg); }
      `}</style>
    </div>
  );
};

const App: React.FC = () => (
  <AuthProvider>
    <ToastProvider>
      <SimulationProvider>
        <DashboardContent />
      </SimulationProvider>
    </ToastProvider>
  </AuthProvider>
);

export default App;