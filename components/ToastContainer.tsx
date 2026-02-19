import React, { useEffect } from 'react';
import { useToast, Toast } from '../context/ToastContext';
import { CheckCircle2Icon, AlertTriangleIcon, XCircleIcon, XIcon, InformationCircleIcon } from './Icons';

const ToastItem: React.FC<{ toast: Toast; onDismiss: (id: string) => void }> = ({ toast, onDismiss }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onDismiss(toast.id);
    }, toast.duration);
    return () => clearTimeout(timer);
  }, [toast, onDismiss]);

  const icons = {
    success: <CheckCircle2Icon className="w-5 h-5 text-green-400" />,
    error: <XCircleIcon className="w-5 h-5 text-red-400" />,
    warning: <AlertTriangleIcon className="w-5 h-5 text-yellow-400" />,
    info: <InformationCircleIcon className="w-5 h-5 text-blue-400" />
  };

  const bgColors = {
    success: 'bg-green-950/90 border-green-500/50 shadow-[0_0_10px_theme(colors.green.900)]',
    error: 'bg-red-950/90 border-red-500/50 shadow-[0_0_10px_theme(colors.red.900)]',
    warning: 'bg-yellow-950/90 border-yellow-500/50 shadow-[0_0_10px_theme(colors.yellow.900)]',
    info: 'bg-blue-950/90 border-blue-500/50 shadow-[0_0_10px_theme(colors.blue.900)]'
  };

  return (
    <div className={`flex items-center p-3 rounded-lg border backdrop-blur-md mb-2 w-80 animate-fade-in-right transition-all duration-300 pointer-events-auto ${bgColors[toast.type]}`}>
      <div className="mr-3 flex-shrink-0">
        {icons[toast.type]}
      </div>
      <div className="flex-grow text-sm text-white font-medium">{toast.message}</div>
      <button onClick={() => onDismiss(toast.id)} className="ml-2 text-white/70 hover:text-white transition-colors">
        <XIcon className="w-4 h-4" />
      </button>
    </div>
  );
};

const ToastContainer: React.FC = () => {
  const { toasts, removeToast } = useToast();

  return (
    <div className="fixed top-20 right-6 z-[100] flex flex-col items-end pointer-events-none">
        {toasts.map((toast) => (
            <ToastItem key={toast.id} toast={toast} onDismiss={removeToast} />
        ))}
    </div>
  );
};

export default ToastContainer;