import { useState, useEffect, useRef, useCallback } from 'react';

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

const SpeechRecognitionAPI = typeof window !== 'undefined' ? (window.SpeechRecognition || window.webkitSpeechRecognition) : null;
const isSupported = !!SpeechRecognitionAPI;

export type ListeningState = 'idle' | 'listening' | 'error' | 'permission_denied';

export const useVoiceCommands = (commands: Command[]) => {
  const [listeningState, setListeningState] = useState<ListeningState>('idle');
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const commandsRef = useRef(commands);
  commandsRef.current = commands;
  
  const isListeningIntentRef = useRef(false);
  const permissionDeniedRef = useRef(false);

  useEffect(() => {
    if (!isSupported) return;

    const recognition = new SpeechRecognitionAPI!();
    recognitionRef.current = recognition;
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
        setListeningState('listening');
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        if (event.error === 'not-allowed') {
            setListeningState('permission_denied');
            permissionDeniedRef.current = true;
        } else if (event.error !== 'no-speech') {
            setListeningState('error');
        }
    };

    recognition.onend = () => {
        if (isListeningIntentRef.current && !permissionDeniedRef.current) {
            try { recognition.start(); } catch(e) {}
        } else {
            setListeningState('idle');
        }
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
        const finalTranscript = Array.from(event.results)
            .filter(result => result.isFinal)
            .map(result => result[0].transcript)
            .join('')
            .trim()
            .toLowerCase();

        if (finalTranscript) {
            for (const { command, callback } of commandsRef.current) {
                const commandsArray = Array.isArray(command) ? command : [command];
                const matchedCommand = commandsArray.find(c => finalTranscript.includes(c.toLowerCase()));
                if (matchedCommand) {
                    callback(finalTranscript);
                    break; 
                }
            }
        }
    };

    return () => {
        isListeningIntentRef.current = false;
        if (recognitionRef.current) {
            try { recognitionRef.current.stop(); } catch(e) {}
        }
    };
  }, []);

  const toggleListening = useCallback(() => {
    if (!isSupported || permissionDeniedRef.current) return;
    
    if (isListeningIntentRef.current) {
        isListeningIntentRef.current = false;
        if (recognitionRef.current) {
            try { recognitionRef.current.stop(); } catch(e) {}
        }
        setListeningState('idle');
    } else {
        isListeningIntentRef.current = true;
        if (recognitionRef.current) {
            try { recognitionRef.current.start(); } catch(e) {}
        }
    }
  }, []);
  
  return { listeningState, toggleListening, isSupported };
};