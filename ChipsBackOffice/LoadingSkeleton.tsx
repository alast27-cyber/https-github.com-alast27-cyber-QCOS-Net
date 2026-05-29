
import React from 'react';

interface LoadingSkeletonProps {
    className?: string;
    lines?: number;
}

const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({ className = "h-4 w-full", lines = 1 }) => {
    return (
        <div className="space-y-2 animate-pulse">
            {Array.from({ length: lines }).map((_, i) => (
                <div 
                    key={i} 
                    className={`bg-cyan-900/30 rounded relative overflow-hidden ${className}`}
                >
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-500/10 to-transparent animate-shimmer"></div>
                </div>
            ))}
        </div>
    );
};

export default LoadingSkeleton;
