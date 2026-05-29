import React from 'react';
import { UIStructure } from '../types';
import * as Icons from './Icons';

interface HolographicPreviewRendererProps {
    structure: UIStructure;
    state: { [key: string]: any };
    onAction: (action: string) => void;
}

const HolographicPreviewRenderer: React.FC<HolographicPreviewRendererProps> = ({ structure, state, onAction }) => {
    const renderNode = (node: UIStructure | string, index: number): React.ReactNode => {
        if (typeof node === 'string') {
            // Check for state injection
            if (node.startsWith('STATE:')) {
                const key = node.replace('STATE:', '');
                return state[key] !== undefined ? String(state[key]) : '';
            }
            // Check for expression injection
            if (node.startsWith('{') && node.endsWith('}')) {
                 const key = node.slice(1, -1);
                 return state[key] !== undefined ? String(state[key]) : node;
            }
            return node;
        }

        const { component, props, children } = node;
        
        // Resolve Props
        const resolvedProps: any = { key: index };
        if (props) {
            Object.entries(props).forEach(([key, value]) => {
                if (typeof value === 'string' && value.startsWith('STATE:')) {
                    const stateKey = value.replace('STATE:', '');
                    resolvedProps[key] = state[stateKey];
                } else if (key.startsWith('on') && typeof value === 'string') {
                    // Handle event binding
                    const actionName = value.replace('()', '').trim();
                    resolvedProps[key] = () => onAction(actionName);
                } else {
                    resolvedProps[key] = value;
                }
            });
        }

        // Handle Standard HTML Elements
        if (component.match(/^[a-z]/)) {
             return React.createElement(
                component,
                resolvedProps,
                children ? children.map((child, i) => renderNode(child, i)) : null
            );
        }

        // Handle Icons
        if (Icons[component as keyof typeof Icons]) {
            return React.createElement(
                Icons[component as keyof typeof Icons] as React.FC<any>,
                resolvedProps
            );
        }

        // Fallback for unknown components (render as div with warning)
        return (
            <div key={index} className="border border-red-500 p-1 text-red-500 text-xs">
                Unknown Component: {component}
            </div>
        );
    };

    return <>{renderNode(structure, 0)}</>;
};

export default HolographicPreviewRenderer;