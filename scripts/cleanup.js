
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

// Files to be deleted from the root directory
const filesToDelete = [
  // Entry points & Configs moved to src/ or redundant
  'index.tsx',
  'App.tsx',
  'types.ts',
  'constants.ts',
  'index-1.html',
  'index-1.tsx',
  'index (1).tsx',
  'App-1.tsx',
  'App (1).tsx',
  'metadata-1.json',
  'package (1).json',
  'gitignore (1).txt',
  '[full_path_of_file_1].txt',
  '[full_path_of_file_2].txt',

  // All Root Components (Moved to src/components/)
  'AGISingularityInterface.tsx',
  'AgiTrainingSimulationRoadmap.tsx',
  'AICommandConsole.tsx',
  'ApplicationChips.tsx',
  'ChatLogPanel.tsx',
  'ChimeraCoreStatus.tsx',
  'PigHavenConsumerTrust.tsx',
  'QuantumNeuroNetworkVisualizer.tsx',
  'QuantumNetworkVisualizer.tsx',
  'QubitSimulator.tsx',
  'AdvancedQuantumInterface.tsx',
  'CHIPSNetworkPanel.tsx',
  'QuantumDeepLearning (1).tsx',
  'QuantumSystemSimulator.tsx',
  'QuantumSystemSimulator-1.tsx',
  'PredictiveTaskOrchestrationPanel-1.tsx',
  'AppForgeLivePreviewPanel.tsx',
  'AchievementBoard.tsx',
  'AISIOS.tsx',
  'AppMarket.tsx',
  'CoreAccessButton.tsx',
  'CubeNavigator.tsx',
  'TesseractCoreFace.tsx',
  'AgentQTraining.tsx',
  'Button-1.tsx',
  'QuantumDeepLearning.tsx',
  'ChipsQuantumInternetPanel.tsx',
  'AnimatedBackground.tsx',
  'CHIPSBackOffice.tsx',
  'AgentQAppCreator.tsx',
  'MemoryMatrix.tsx',
  'SystemHealthChart.tsx',
  'SemanticIntegrityCheck.tsx',
  'SemanticDriftPanel.tsx',
  'SecurityMonitorAndSimulator.tsx',
  'RolePermissionsPanel.tsx',
  'ResourceSteward.tsx',
  'QubitStateVisualizer.tsx',
  'ViewNavigator.tsx',
  'VideoAnalysis.tsx',
  'VQEToolkit.tsx',
  'UtilityHubPanel.tsx',
  'UniverseSimulator.tsx',
  'ToastContainer.tsx',
  'TextToAppInterface.tsx',
  'SystemLog.tsx',
  'SystemDiagnostic.tsx',
  'SyntaxHighlighter.tsx',
  'SpecializedTrainingInputPanel.tsx',
  'PublicDeploymentPanel.tsx',
  'PublicDeploymentOptimizationHub.tsx',
  'QOSKernelConsole.tsx',
  'OnboardingFlow.tsx',
  'AgenticAddressBar.tsx',
  'ChipsBrowserInstaller.tsx',
  'QuantumEngineeringDesign.tsx',
  'ChipsMail.tsx',
  'ChipsEconomy.tsx',
  'EditorWorkspace.tsx',
  'GlobalAbundanceEngine.tsx',
  'GlassPanel.tsx',
  'HolographicCube.tsx',
  'PhilippineSwineResilience.tsx',
  'NeuralProgrammingPanel.tsx',
  'MolecularSimulationToolkit.tsx',
  'ModelRegistry.tsx',
  'MergedEvolutionPanel.tsx',
  'LoginScreen.tsx',
  'LiveIndicator.tsx',
  'KpiDisplay.tsx',
  'QCOSUserIdentityNodeRegistry.tsx',
  'QCOSSystemEvolutionInterface.tsx',
  'QCOSGateway.tsx',
  'QBioMedDrugDiscovery.tsx',
  'QANExecutionSimulator.tsx',
  'CollapsibleSection.tsx',
  'CircuitVisualizer.tsx',
  'ChipsQuantumInternetNetwork.tsx',
  'CubeFace.tsx',
  'DecoherenceDriftMonitor.tsx',
  'DeployAppModal.tsx',
  'CHIPSAppStore.tsx',
  'Icons.tsx',
  'HolographicPreviewRenderer.tsx',
  'DeployedAppWrapper.tsx',
  'QuantumReinforcementLearning.tsx',
  'QPUHealth.tsx',
  'IAIKernelStatus.tsx',
  'DistributedCognitiveArchitecture.tsx',
  'QLangCoreEvolutionPanel.tsx',
  'ImageAnalysis.tsx',
  'GenericQuantumSolver.tsx',
  'QKDSimulator.tsx',
  'QDeNoiseProcessor.tsx',
  'GlobalSwineForesight.tsx',
  'AdminChat.tsx',
  'EditorView.tsx',
  'FullScreenSwitcher.tsx',
  'QuantumAppExchange.tsx',
  'QuantumVoiceChat.tsx',
  'QuantumToWebGatewayPanel.tsx',
  'QuantumToWebGatewayEnhancedPanel.tsx',
  'QuantumMemoryDataMatrix.tsx',
  'MonacoEditorWrapper.tsx',
  'LivePreviewFrame.tsx',
  'LoadingSkeleton.tsx',
  'DERGridOptimizer.tsx',
  'AgentQCore.tsx',
  'AgentQ.tsx',
  'AgentQSelfTrainingEvolution.tsx',
  'QuantumSpeedometer.tsx',
  'QuantumProtocolSimulator.tsx',
  'QuantumDataSearchPanel.tsx',
  'CHIPSStoreAdmin.tsx',
  'CHIPSGatewayAdmin.tsx',
  'CHIPSBrowserSDK.tsx',
  'CHIPSBrowser.tsx',
  'ChipsDevPlatform.tsx',
  'AnomalyLog.tsx',
  'PowerMetrics.tsx',
  'PredictiveTaskOrchestrationPanel.tsx',
  'QuantumProgrammingInterface.tsx',
  'ApplicationHub.tsx',
  'QuantumExecutionFlow.tsx',
  'AgentQEnhancedInsights.tsx',
  'AgentQMasteryInterface.tsx',
  'QuantumLargeLanguageModel.tsx',
  'QubitStabilityChart.tsx',
  'QuantumMonteCarloFinance.tsx'
];

// Directories to be removed (recursively)
const dirsToDelete = [
  'components',
  'context',
  'hooks',
  'services',
  'utils',
  'qllm'
];

console.log('Starting QCOS Project Cleanup...');

// Delete Files
filesToDelete.forEach(file => {
  const filePath = path.join(rootDir, file);
  if (fs.existsSync(filePath)) {
    try {
      fs.unlinkSync(filePath);
      console.log(`Deleted file: ${file}`);
    } catch (err) {
      console.error(`Error deleting ${file}: ${err.message}`);
    }
  }
});

// Delete Directories
dirsToDelete.forEach(dir => {
    const dirPath = path.join(rootDir, dir);
    if (fs.existsSync(dirPath)) {
        try {
            fs.rmSync(dirPath, { recursive: true, force: true });
            console.log(`Deleted directory: ${dir}`);
        } catch (err) {
            console.error(`Error deleting directory ${dir}: ${err.message}`);
        }
    }
});

console.log('Cleanup complete. Project structure optimized for src/ architecture.');
