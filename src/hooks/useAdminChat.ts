import { useState, useCallback } from 'react';
import { Message } from '../utils/agentUtils';

export const useAdminChat = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [consoleMessages, setConsoleMessages] = useState<Message[]>([]);
  const [chatMessages, setChatMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState<'console' | 'chat'>('console');

  const onToggle = useCallback(() => setIsOpen(prev => !prev), []);

  const onSendMessage = useCallback((text: string, file: File | null) => {
    if (!text.trim() && !file) return;

    setIsLoading(true);
    const newMessage: Message = { 
        sender: 'user', 
        text: text.trim(), 
        timestamp: Date.now(),
        attachment: file ? { name: file.name } : undefined
    };
    
    if (mode === 'console') {
        setConsoleMessages(prev => [...prev, newMessage]);
        // Simulate Kernel Response
        setTimeout(() => {
            setConsoleMessages(prev => [...prev, { 
                sender: 'system', 
                text: `[KERNEL] Command processed: ${text.substring(0, 20)}${text.length > 20 ? '...' : ''}\nStatus: OK\nResponse: Logic gates synchronized.`, 
                timestamp: Date.now() 
            }]);
            setIsLoading(false);
        }, 1000);
    } else {
        setChatMessages(prev => [...prev, newMessage]);
        // Simulate Admin Response
        setTimeout(() => {
            setChatMessages(prev => [...prev, { 
                sender: 'ai', 
                text: `Operator priority acknowledged. System vitals are nominal. How can I assist further with your current project?`, 
                timestamp: Date.now() 
            }]);
            setIsLoading(false);
        }, 1500);
    }
  }, [mode]);

  return {
    isOpen,
    onToggle,
    mode,
    setMode,
    consoleMessages,
    chatMessages,
    isLoading,
    onSendMessage,
    adminChatProps: { 
        isOpen,
        onToggle,
        mode,
        setMode,
        consoleMessages,
        chatMessages,
        isLoading,
        onSendMessage
    }
  };
};