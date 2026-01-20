
import { useState, useEffect, useRef, useCallback } from 'react';

// Added proper TypeScript definitions for the Web Speech API
declare global {
  interface SpeechRecognitionErrorEvent extends Event {
    readonly error: string;
  }

  interface SpeechRecognitionEvent extends Event {
    readonly results: SpeechRecognitionResultList;
    readonly resultIndex: number;
  }

  interface SpeechRecognitionResultList {
    readonly length: number;
    item(index: number): SpeechRecognitionResult;
    [index: number]: SpeechRecognitionResult;
  }

  interface SpeechRecognitionResult {
    readonly isFinal: boolean;
    readonly length: number;
    item(index: number): SpeechRecognitionAlternative;
    [index: number]: SpeechRecognitionAlternative;
  }

  interface SpeechRecognitionAlternative {
    readonly transcript: string;
    readonly confidence: number;
  }

  interface SpeechRecognition extends EventTarget {
    continuous: boolean;
    interimResults: boolean;
    lang: string;
    start(): void;
    stop(): void;
    onresult: (event: SpeechRecognitionEvent) => void;
    onerror: (event: SpeechRecognitionErrorEvent) => void;
    onstart: () => void;
    onend: () => void;
  }

  interface SpeechRecognitionConstructor {
    prototype: SpeechRecognition;
    new (): SpeechRecognition;
  }

  interface Window {
    SpeechRecognition: SpeechRecognitionConstructor;
    webkitSpeechRecognition: SpeechRecognitionConstructor;
  }
}

interface Command {
  command: string | string[];
  callback: (spoken: string) => void;
}

// Check for browser support
const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
const isSupported = !!SpeechRecognitionAPI;

export type ListeningState = 'idle' | 'listening' | 'error' | 'permission_denied';

export const useVoiceCommands = (commands: Command[]) => {
  const [listeningState, setListeningState] = useState<ListeningState>('idle');
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const commandsRef = useRef(commands);
  commandsRef.current = commands; // Keep commands up to date
  
  // Refs to track state without triggering re-renders inside callbacks
  const isListeningIntentRef = useRef(false);
  const permissionDeniedRef = useRef(false);
  const errorTimeoutRef = useRef<number | null>(null);

  // Ref to hold the permission status object for cleanup
  const permissionStatusRef = useRef<PermissionStatus | null>(null);

  useEffect(() => {
    if (!isSupported) return;

    let isMounted = true;

    // --- Proactive Permission Check ---
    const handlePermissionChange = () => {
        const status = permissionStatusRef.current;
        if (!status) return;
        
        if (status.state === 'denied') {
            if (isMounted) setListeningState('permission_denied');
            permissionDeniedRef.current = true;
            if (isListeningIntentRef.current && recognitionRef.current) {
                isListeningIntentRef.current = false;
                try { recognitionRef.current.stop(); } catch(e) {}
            }
        }
    };

    if (typeof navigator.permissions?.query === 'function') {
        navigator.permissions.query({ name: 'microphone' } as any).then(status => {
            if (!isMounted) return; // Prevent leak if unmounted before promise resolves
            
            permissionStatusRef.current = status;
            handlePermissionChange(); // Check initial state
            status.addEventListener('change', handlePermissionChange);
        }).catch(err => {
            console.warn("Could not query microphone permission status", err);
        });
    }

    // --- Initialize Recognition Engine ---
    const recognition = new SpeechRecognitionAPI();
    recognitionRef.current = recognition;
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
        if (!isMounted) return;
        if (permissionDeniedRef.current) return;
        setListeningState('listening');
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        if (!isMounted) return;
        // console.error('Speech recognition error:', event.error); // Suppress log spam
        if (event.error === 'not-allowed') {
            setListeningState('permission_denied');
            permissionDeniedRef.current = true;
        } else if (event.error !== 'no-speech') {
            setListeningState('error');
        }
    };

    recognition.onend = () => {
        if (!isMounted) return;
        if (permissionDeniedRef.current) return;
        
        // Restart recognition automatically if the intent is to be listening (keep-alive)
        if (isListeningIntentRef.current) {
            try {
                recognition.start();
            } catch (e) {
                // Ignore if already started
            }
        } else {
            setListeningState('idle');
        }
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
        if (!isMounted) return;

        // Debounce / watchdog to restart if stuck
        if (errorTimeoutRef.current) window.clearTimeout(errorTimeoutRef.current);
        errorTimeoutRef.current = window.setTimeout(() => {
            if (isMounted && isListeningIntentRef.current && recognitionRef.current) {
                 try { recognitionRef.current.stop(); } catch(e) {}
                 // It will restart in onend
            }
        }, 5000);

        const transcript = Array.from(event.results)
            .map(result => result[0])
            .map(result => result.transcript)
            .join('');

        const finalTranscript = Array.from(event.results)
            .filter(result => result.isFinal)
            .map(result => result[0].transcript)
            .join('')
            .trim()
            .toLowerCase();

        if (finalTranscript) {
            console.log('Voice Command Input:', finalTranscript);
            for (const { command, callback } of commandsRef.current) {
                const commandsArray = Array.isArray(command) ? command : [command];
                const matchedCommand = commandsArray.find(c => finalTranscript.includes(c.toLowerCase()));
                if (matchedCommand) {
                    console.log(`Command Executed: "${matchedCommand}"`);
                    callback(finalTranscript);
                    break; 
                }
            }
        }
    };

    // Cleanup on unmount
    return () => {
        isMounted = false;
        isListeningIntentRef.current = false;
        
        if (recognitionRef.current) {
            try { recognitionRef.current.stop(); } catch(e) {}
            recognitionRef.current = null;
        }
        
        if (permissionStatusRef.current) {
            permissionStatusRef.current.removeEventListener('change', handlePermissionChange);
            permissionStatusRef.current = null;
        }
        
        if (errorTimeoutRef.current) {
            window.clearTimeout(errorTimeoutRef.current);
        }
    };
  }, []);

  const toggleListening = useCallback(() => {
    if (permissionDeniedRef.current) return;
    
    if (isListeningIntentRef.current) {
        // Stop listening
        isListeningIntentRef.current = false;
        if (recognitionRef.current) {
            try { recognitionRef.current.stop(); } catch(e) {}
        }
        setListeningState('idle');
    } else {
        // Start listening
        isListeningIntentRef.current = true;
        if (recognitionRef.current) {
            try {
              recognitionRef.current.start();
            } catch (e) {
                console.warn("Speech recognition already active.");
            }
        }
    }
  }, []);
  
  return { listeningState, toggleListening, isSupported };
};
