
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
            if (node.startsWith('STATE:')) {
                const key = node.replace('STATE:', '');
                return state[key] !== undefined ? String(state[key]) : '';
            }
            if (node.startsWith('{') && node.endsWith('}')) {
                 const key = node.slice(1, -1);
                 return state[key] !== undefined ? String(state[key]) : node;
            }
            return node;
        }

        const { component, props, children } = node;
        const resolvedProps: any = { key: index };
        if (props) {
            Object.entries(props).forEach(([key, value]) => {
                if (typeof value === 'string' && value.startsWith('STATE:')) {
                    const stateKey = value.replace('STATE:', '');
                    resolvedProps[key] = state[stateKey];
                } else if (key.startsWith('on') && typeof value === 'string') {
                    const actionName = value.replace('()', '').trim();
                    resolvedProps[key] = () => onAction(actionName);
                } else {
                    resolvedProps[key] = value;
                }
            });
        }

        if (component.match(/^[a-z]/)) {
             return React.createElement(
                component,
                resolvedProps,
                children ? children.map((child, i) => renderNode(child, i)) : null
            );
        }

        if (Icons[component as keyof typeof Icons]) {
            return React.createElement(
                Icons[component as keyof typeof Icons] as React.FC<any>,
                resolvedProps
            );
        }

        return <div key={index}>Unknown Area: {component}</div>;
    };

    return <>{renderNode(structure, 0)}</>;
};

export default HolographicPreviewRenderer;
