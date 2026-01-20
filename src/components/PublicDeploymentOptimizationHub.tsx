
import React, { useState, useEffect } from 'react';
import GlassPanel from './GlassPanel';
import {
  GlobeIcon,
  CheckCircle2Icon,
  AlertTriangleIcon,
  ClockIcon,
  ArrowTopRightOnSquareIcon,
} from './Icons';
import { AppDefinition, URIAssignment } from '../types';

interface PublicDeploymentOptimizationHubProps {
  apps: AppDefinition[];
  uriAssignments: URIAssignment[];
  onTogglePublic: (appId: string) => void;
}

type OptimizationStatus = 'Optimized' | 'Monitoring' | 'Action Recommended' | 'Initializing';

const PublicDeploymentOptimizationHub: React.FC<PublicDeploymentOptimizationHubProps> = ({ apps, uriAssignments, onTogglePublic }) => {
  const [optimizationStatuses, setOptimizationStatuses] = useState<Record<string, OptimizationStatus>>({});

  useEffect(() => {
    // Initialize or update random statuses for apps when they appear
    const newStatuses: Record<string, OptimizationStatus> = {};
    apps.forEach(app => {
        if (!optimizationStatuses[app.id]) {
            const statuses: OptimizationStatus[] = ['Optimized', 'Monitoring', 'Action Recommended', 'Initializing'];
            newStatuses[app.id] = statuses[Math.floor(Math.random() * statuses.length)];
        }
    });
    if (Object.keys(newStatuses).length > 0) {
        setOptimizationStatuses(prev => ({...prev, ...newStatuses}));
    }
  }, [apps, optimizationStatuses]);

  // Returns the appropriate icon for the QNN optimization status
  const getOptimizationIcon = (status: OptimizationStatus) => {
    switch (status) {
      case 'Optimized':
        return <CheckCircle2Icon className="w-4 h-4 text-green-400 mr-1" />;
      case 'Monitoring':
        return <ClockIcon className="w-4 h-4 text-yellow-400 mr-1" />;
      case 'Action Recommended':
        return <AlertTriangleIcon className="w-4 h-4 text-red-400 mr-1" />;
      case 'Initializing':
        return <ClockIcon className="w-4 h-4 text-gray-400 mr-1" />;
      default:
        return null;
    }
  };

  const navigateToSystemEvolution = (appName: string) => {
    console.log(`Simulating navigation to QCOS System Evolution for app: ${appName}`);
    alert(`Opening QCOS System Evolution details for ${appName}`);
  };

  const installedApps = apps.filter(app => app.status === 'installed');

  return (
    <GlassPanel title="Public Deployment & Optimization Hub">
      <div className="p-4 space-y-4 text-sm h-full overflow-y-auto">
        <p className="text-cyan-200 mb-4">
          Manage public access for your CHIPS applications via the <span className="text-cyan-400 font-semibold">Quantum-to-Web Gateway</span> and monitor their <span className="text-cyan-400 font-semibold">QNN optimization status</span>.
        </p>

        {installedApps.map((app) => {
            const assignment = uriAssignments.find(a => a.appName === app.name);
            const isPublic = !!assignment;
            const publicUrl = assignment?.https_url;
            const optimizationStatus = optimizationStatuses[app.id] || 'Initializing';

            return (
              <div
                key={app.id}
                className="flex flex-col md:flex-row items-start md:items-center justify-between p-3 bg-black/20 rounded-lg border border-cyan-700/50 hover:border-cyan-500/50 transition-colors duration-200"
              >
                <div className="flex items-center flex-grow mb-2 md:mb-0 w-full md:w-auto">
                  <app.icon className="w-6 h-6 text-cyan-300 mr-3" />
                  <span className="font-medium text-cyan-100 flex-grow break-all">{app.name}</span>
                </div>
    
                <div className="flex flex-col md:flex-row items-start md:items-center space-y-2 md:space-y-0 md:space-x-4 w-full md:w-auto">
                  {/* Public Gateway Toggle */}
                  <div className="flex items-center">
                    <GlobeIcon className="w-5 h-5 text-cyan-400 mr-2" />
                    <label htmlFor={`toggle-${app.id}`} className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        id={`toggle-${app.id}`}
                        className="sr-only peer"
                        checked={isPublic}
                        onChange={() => onTogglePublic(app.id)}
                      />
                      <div className="w-11 h-6 bg-gray-600 rounded-full peer peer-focus:ring-2 peer-focus:ring-cyan-500 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:border-gray-300 after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-cyan-600"></div>
                      <span className="ml-3 text-sm font-medium text-gray-300">Public</span>
                    </label>
                  </div>
    
                  {/* Public URL Display */}
                  {isPublic && publicUrl && (
                    <div className="flex items-center md:min-w-[150px]">
                      <a
                        href={publicUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-cyan-400 text-xs hover:underline flex items-center whitespace-nowrap overflow-hidden text-ellipsis"
                        title={`Open ${publicUrl}`}
                      >
                        {publicUrl.replace('https://', '')}
                        <ArrowTopRightOnSquareIcon className="w-3 h-3 ml-1 flex-shrink-0" />
                      </a>
                    </div>
                  )}
    
                  {/* QNN Optimization Status Indicator */}
                  <button
                    onClick={() => navigateToSystemEvolution(app.name)}
                    className={`flex items-center px-2 py-1 rounded-md text-xs font-semibold whitespace-nowrap ${
                      optimizationStatus === 'Optimized' ? 'bg-green-700/30 text-green-300' :
                      optimizationStatus === 'Monitoring' ? 'bg-yellow-700/30 text-yellow-300' :
                      optimizationStatus === 'Action Recommended' ? 'bg-red-700/30 text-red-300' :
                      'bg-gray-700/30 text-gray-300'
                    } hover:bg-white/10 transition-colors duration-200`}
                    title={`View QNN optimization details for ${app.name} in QCOS System Evolution`}
                  >
                    {getOptimizationIcon(optimizationStatus)}
                    {optimizationStatus}
                  </button>
                </div>
              </div>
            );
        })}
      </div>
    </GlassPanel>
  );
};

export default PublicDeploymentOptimizationHub;
