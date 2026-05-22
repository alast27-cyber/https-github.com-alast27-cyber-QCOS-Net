import React, { useState, useEffect, useRef } from 'react';
import { ChevronDownIcon } from './Icons';

interface CollapsibleSectionProps {
  title: string;
  children: React.ReactNode;
  /** Delay in milliseconds for auto-collapse. Set to 0 or a negative number to disable. */
  autoCollapseDelay?: number;
  isOpen?: boolean;
  onToggle?: () => void;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({ title, children, autoCollapseDelay = 20000, isOpen, onToggle }) => {
  const [internalIsOpen, setInternalIsOpen] = useState(false);
  const timerRef = useRef<number | null>(null);

  const open = isOpen !== undefined ? isOpen : internalIsOpen;

  const handleToggle = () => {
    if (onToggle) {
      onToggle();
    } else {
      setInternalIsOpen(prev => !prev);
    }
  };

  const clearTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };
  
  const startTimer = () => {
    clearTimer();
    if (autoCollapseDelay > 0) {
      timerRef.current = window.setTimeout(() => {
        if (onToggle) {
            // If controlled, we can't directly set state.
            // A more advanced implementation might have an onTimeout callback.
            // For now, we assume auto-collapse is for uncontrolled components.
        } else {
            setInternalIsOpen(false);
        }
      }, autoCollapseDelay);
    }
  };

  useEffect(() => {
    if (open) {
      startTimer();
    } else {
      clearTimer();
    }
    return clearTimer;
  }, [open, autoCollapseDelay]);

  return (
    <div className="border-b border-blue-500/20 last:border-b-0">
      <button 
        className="flex justify-between items-center w-full p-3 text-left text-cyan-300 hover:bg-black/20 transition-colors"
        onClick={handleToggle}
        aria-expanded={open}
        title={`Toggle ${title} section`}
      >
        <span className="text-sm font-semibold">{title}</span>
        <ChevronDownIcon className={`w-5 h-5 transition-transform duration-300 ${open ? 'rotate-180' : ''}`} />
      </button>
      <div 
        className={`grid transition-all duration-500 ease-in-out ${open ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'}`}
      >
        <div className="overflow-hidden">
          <div className={`p-3 pt-0 transition-all duration-300 ease-in-out delay-100 ${open ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-2'}`}>
            {children}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CollapsibleSection;