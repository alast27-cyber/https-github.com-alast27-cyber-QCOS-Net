import React, { useMemo, Suspense } from 'react';
import { AppDefinition, URIAssignment, LogEntry, UIStructure, SystemHealth } from '../types';
import { getPanelMetadata, FaceData } from '../utils/dashboardConfig';

// Standardize imports to relative paths for browser ESM compatibility
import AgentQCore from '../components/AgentQCore';
import AgentQ from '../components/AgentQ';
import SystemDiagnostic from '../components/SystemDiagnostic';
import QCOSGateway from '../components/QCOSGateway';
import CHIPSBrowserSDK from '../components/CHIPSBrowserSDK';
import CHIPSAppStore from '../components/CHIPSAppStore';
import QuantumAppExchange from '../components/QuantumAppExchange';
import UniverseSimulator from '../components/UniverseSimulator';
import CHIPSBackOffice from '../components/CHIPSBackOffice';
import ChipsEconomy from '../components/ChipsEconomy';
import QPUHealth from '../components/QPUHealth';
import ImageAnalysis from '../components/ImageAnalysis';
import NeuralProgrammingPanel from '../components/NeuralProgrammingPanel';
import GlobalAbundanceEngine from '../components/GlobalAbundanceEngine';
import MergedEvolutionPanel from '../components/MergedEvolutionPanel';
import QuantumVoiceChat from '../components/QuantumVoiceChat';
import QuantumMachineLearning from '../components/QuantumMachineLearning';
import SecurityMonitorAndSimulator from '../components/SecurityMonitorAndSimulator';
import QuantumLargeLanguageModel from '../components/QuantumLargeLanguageModel'; 
import QuantumCognitiveArchitecture from '../components/QuantumCognitiveArchitecture';
import QuantumMonteCarloFinance from '../components/QuantumMonteCarloFinance';
import QBioMedDrugDiscovery from '../components/QBioMedDrugDiscovery';
import MolecularSimulationToolkit from '../components/MolecularSimulationToolkit';
import GlobalSwineForesight from '../components/GlobalSwineForesight';
import PhilippineSwineResilience from '../components/PhilippineSwineResilience';
import PigHavenConsumerTrust from '../components/PigHavenConsumerTrust';
import GenericQuantumSolver from '../components/GenericQuantumSolver';
import QKDSimulator from '../components/QKDSimulator';
import QuantumNetworkVisualizer from '../components/QuantumNetworkVisualizer';
import VQEToolkit from '../components/VQEToolkit';
import QDeNoiseProcessor from '../components/QDeNoiseProcessor';
import QuantumReinforcementLearning from '../components/QuantumReinforcementLearning';
import QuantumDeepLearning from '../components/QuantumDeepLearning';
import DeployedAppWrapper from '../components/DeployedAppWrapper';
import ChipsQuantumInternetNetwork from '../components/ChipsQuantumInternetNetwork';
import QuantumDataIngestion from '../components/QuantumDataIngestion';
import QuantumSwineIntelligence from '../components/QuantumSwineIntelligence';
import AGISingularityInterface from '../components/AGISingularityInterface';
import DERGridOptimizer from '../components/DERGridOptimizer';
import QOSKernelConsole from '../components/QOSKernelConsole';
import QuantumEngineeringDesign from '../components/QuantumEngineeringDesign';
import AgentQSelfTrainingEvolution from '../components/AgentQSelfTrainingEvolution';

const ChipsDevPlatform = React.lazy(() => import('../components/ChipsDevPlatform'));

export interface UsePanelContentProps {
    systemHealth: SystemHealth;
    isRecalibrating: boolean;
    isUpgrading: boolean;
    activeDataStreams: string[];
    uriAssignments: URIAssignment[];
    agentWeaverInput: string;
    onPublishToExchange: (details: { name: string; description: string; code: string; uiStructure?: UIStructure }) => void;
    onConnectToGateway: (source: string) => void;
    logs: LogEntry[];
    addLog: (level: LogEntry['level'], msg: string) => void;
    codebase: { [key: string]: string };
    onApplyPatch: (filePath: string, newContent: string) => void;
    marketApps: AppDefinition[]; 
    onInstallApp: (id: string) => void;
    onLaunchApp: (id: string) => void;
    onGenerateApp: (description: string) => Promise<{ files: { [path: string]: string }, uiStructure: UIStructure | null }>;
    onUpdateApp: (files: { [path: string]: string }) => Promise<{ updatedFiles: { [path: string]: string }, summary: string }>;
    onDebugApp: (files: { [path: string]: string }) => Promise<{ fixedFiles: { [path: string]: string }, summary: string, uiStructure: UIStructure | null }>;
    onTogglePublic: (appId: string) => void;
    onSendToWeaver: (input: string) => void;
    onCloseSystemDiagnostic: () => void;
    adminLevel: number;
    qcosVersion: number;
    agentQProps: any; 
    editCode: (code: string, instruction: string) => Promise<string>;
    onOpenAppCreator: (prompt: string) => void;
    onSimulateApp: (appId: string) => void;
    connectedAppId: string | null;
    toggleAgentQ: () => void;
}

export const usePanelContent = (props: UsePanelContentProps) => {
  const panelMetadata = useMemo(() => getPanelMetadata(props.qcosVersion), [props.qcosVersion]);

  const hydratedApps = useMemo(() => {
    return props.marketApps.map(app => {
        let comp: React.ReactNode | undefined = undefined;
        switch (app.id) {
            case 'q-vox': comp = <QuantumVoiceChat />; break;
            case 'qmc-finance': comp = <QuantumMonteCarloFinance />; break;
            case 'q-biomed': comp = <QBioMedDrugDiscovery />; break;
            case 'mol-sim': comp = <MolecularSimulationToolkit />; break;
            case 'qnet-viz': comp = <QuantumNetworkVisualizer />; break;
            case 'vqe-toolkit': comp = <VQEToolkit />; break;
            case 'quantum-swine-intelligence': comp = <QuantumSwineIntelligence onOpenApp={props.onLaunchApp} />; break;
            case 'global-swine-foresight': comp = <GlobalSwineForesight />; break;
            case 'philippine-swine-resilience': comp = <PhilippineSwineResilience />; break;
            case 'pighaven-consumer-trust': comp = <PigHavenConsumerTrust />; break;
            case 'generic-solver': comp = <GenericQuantumSolver />; break;
            case 'qkd-sim': comp = <QKDSimulator />; break;
            case 'q-denoise': comp = <QDeNoiseProcessor />; break;
            case 'der-grid-optimizer': comp = <DERGridOptimizer />; break;
            case 'qos-kernel-manager': comp = <QOSKernelConsole />; break;
            default: break; 
        }
        return { ...app, component: comp };
    });
  }, [props.marketApps, props.onLaunchApp]);

  return useMemo<Record<number, FaceData>>(() => {
    const getContentForId = (id: string): React.ReactNode => {
        const connectedApp = props.connectedAppId ? hydratedApps.find(a => a.id === props.connectedAppId) : null;
        switch (id) {
            case 'agentq-core': return <AgentQCore systemHealth={props.systemHealth} isRecalibrating={props.isRecalibrating} isUpgrading={props.isUpgrading} activeDataStreams={props.activeDataStreams} onMaximizeSubPanel={props.onLaunchApp} />;
            case 'agentq-self-evo': return <AgentQSelfTrainingEvolution isRecalibrating={props.isRecalibrating} isUpgrading={props.isUpgrading} systemHealth={props.systemHealth} activeDataStreams={props.activeDataStreams} />;
            case 'quantum-reinforcement-learning': return <QuantumReinforcementLearning />;
            case 'agent-q-chat': return <AgentQ {...props.agentQProps} embedded={true} isOpen={true} />;
            case 'system-diagnostic': return <SystemDiagnostic onClose={props.onCloseSystemDiagnostic} />;
            case 'qcos-core-gateway': return <QCOSGateway codebase={props.codebase} onApplyPatch={props.onApplyPatch} editCode={props.editCode} onMaximizeSubPanel={props.onLaunchApp} />;
            case 'data-ingestion': return <QuantumDataIngestion onMaximize={() => {}} />; 
            case 'chips-quantum-network': return <ChipsQuantumInternetNetwork apps={hydratedApps} onInstallApp={props.onInstallApp} onToggleAgentQ={props.toggleAgentQ} onDeployApp={props.onPublishToExchange} />;
            case 'chips-browser-sdk': return <CHIPSBrowserSDK initialApp={undefined} onToggleAgentQ={() => {}} apps={hydratedApps} onInstallApp={props.onInstallApp} />;
            case 'chips-app-store': return <CHIPSAppStore apps={hydratedApps} onInstall={props.onInstallApp} onLaunch={props.onLaunchApp} />;
            case 'quantum-app-exchange': return <QuantumAppExchange apps={hydratedApps} onInstall={props.onInstallApp} onLaunch={props.onLaunchApp} onDeployApp={props.onPublishToExchange} uriAssignments={props.uriAssignments} onGenerateApp={props.onGenerateApp} onUpdateApp={props.onUpdateApp} onDebugApp={props.onDebugApp} onSimulate={props.onSimulateApp} />;
            case 'universe-simulator': return <UniverseSimulator qubitCount={240} onApplyPatch={props.onApplyPatch} onExportToCreator={props.onOpenAppCreator} connectedApp={connectedApp} />;
            case 'agi-singularity-interface': return <AGISingularityInterface qubitStability={props.systemHealth.qubitStability} />;
            case 'chips-dev-platform': return (
                <Suspense fallback={<div className="h-full flex items-center justify-center text-cyan-500">Loading Dev Platform...</div>}>
                    <ChipsDevPlatform onAiAssist={props.editCode} onDeploy={props.onPublishToExchange} />
                </Suspense>
            );
            case 'chips-back-office': return <CHIPSBackOffice uriAssignments={props.uriAssignments} marketApps={hydratedApps} onApplyPatch={props.onApplyPatch} onMaximizeSubPanel={props.onLaunchApp} onAiAssist={props.editCode} onDeploy={props.onPublishToExchange} systemHealth={props.systemHealth} />;
            case 'security-monitor': return <SecurityMonitorAndSimulator />;
            case 'chips-economy': return <ChipsEconomy />;
            case 'qpu-health': return <QPUHealth systemHealth={props.systemHealth} />;
            case 'neural-programming': return <NeuralProgrammingPanel />;
            case 'quantum-large-language-model': return <QuantumLargeLanguageModel />;
            case 'quantum-machine-learning': return <QuantumMachineLearning />;
            case 'quantum-cognitive-architecture': return <QuantumCognitiveArchitecture onApplyPatch={props.onApplyPatch} />;
            case 'quantum-deep-learning': return <QuantumDeepLearning onApplyPatch={props.onApplyPatch} />;
            case 'quantum-engineering-design': return <QuantumEngineeringDesign />;
            case 'q-vox': return <QuantumVoiceChat />;
            case 'qmc-finance': return <QuantumMonteCarloFinance />;
            case 'q-biomed': return <QBioMedDrugDiscovery />;
            case 'mol-sim': return <MolecularSimulationToolkit />;
            case 'qnet-viz': return <QuantumNetworkVisualizer />;
            case 'vqe-toolkit': return <VQEToolkit />;
            case 'der-grid-optimizer': return <DERGridOptimizer />;
            case 'qos-kernel-manager': return <QOSKernelConsole />;
            case 'quantum-swine-intelligence': return <QuantumSwineIntelligence onOpenApp={props.onLaunchApp} />;
            case 'global-swine-foresight': return <GlobalSwineForesight />;
            case 'philippine-swine-resilience': return <PhilippineSwineResilience />;
            case 'pighaven-consumer-trust': return <PigHavenConsumerTrust />;
            case 'generic-solver': return <GenericQuantumSolver />;
            case 'qkd-sim': return <QKDSimulator />;
            case 'q-denoise': return <QDeNoiseProcessor />;
            default:
                const installedApp = hydratedApps.find(a => a.id === id && a.status === 'installed');
                if (installedApp) {
                    if (installedApp.isCustom && installedApp.uiStructure && installedApp.code) {
                        return <DeployedAppWrapper structure={installedApp.uiStructure} code={installedApp.code} />;
                    }
                    if (installedApp.component) {
                        return installedApp.component;
                    }
                }
                return <div className="p-4 text-cyan-700">Content for {id} is not initialized.</div>;
        }
    };
    const result: Record<number, FaceData> = {};
    Object.keys(panelMetadata).forEach((key) => {
      const faceKey = parseInt(key, 10);
      result[faceKey] = {
        ...panelMetadata[faceKey],
        panels: panelMetadata[faceKey].panels.map(panel => ({
          ...panel,
          content: getContentForId(panel.id)
        }))
      };
    });
    return result;
  }, [panelMetadata, props, hydratedApps]);
};