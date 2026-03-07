
import React from 'react';
import { AppDefinition } from '../types';

interface ApplicationChipsProps {
  apps: AppDefinition[];
  onChipSelect: (id: string) => void;
  radius: number;
}

const ApplicationChips: React.FC<ApplicationChipsProps> = ({ apps, onChipSelect, radius }) => {
  if (apps.length === 0) return null;
  
  const angleStep = 360 / apps.length;

  return (
    <div className="absolute inset-0 w-full h-full flex items-center justify-center" style={{ transformStyle: 'preserve-3d' }}>
      <div className="relative w-0 h-0" style={{ transformStyle: 'preserve-3d' }}>
        {apps.map((app, index) => {
          const angle = index * angleStep - 90; // Start from top
          const x = radius * Math.cos(angle * Math.PI / 180);
          const y = radius * Math.sin(angle * Math.PI / 180);

          return (
            <button
              key={app.id}
              onClick={() => onChipSelect(app.id)}
              className="app-chip group"
              style={{
                transform: `translate(${x}px, ${y}px) translateZ(5px)`,
              }}
              aria-label={`Launch ${app.name}`}
            >
              <app.icon className="w-6 h-6 text-cyan-300 transition-transform group-hover:scale-110" />
              <div className="chip-tooltip">{app.name}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default ApplicationChips;
