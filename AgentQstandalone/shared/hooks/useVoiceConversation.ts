
import { useState, useEffect, useRef, useCallback } from 'react';

const SpeechRecognitionAPI = typeof window !== 'undefined' ? (window.SpeechRecognition || window.webkitSpeechRecognition) : null;
const isSupported = !!SpeechRecognitionAPI;

export type VoiceConversationState = 'idle' | 'listening' | 'processing' | 'speaking' | 'error' | 'disabled';

interface UseVoiceConversationProps {
    onSendMessage: (text: string) => void;
    isAgentSpeaking: boolean;
    isAgentLoading: boolean;
    enabled: boolean;
}

export const useVoiceConversation = ({ onSendMessage, isAgentSpeaking, isAgentLoading, enabled }: UseVoiceConversationProps) => {
    const [state, setState] = useState<VoiceConversationState>(isSupported ? 'idle' : 'disabled');
    const recognitionRef = useRef<any>(null);
    const isListeningIntentRef = useRef(false);
    const lastTranscriptRef = useRef('');

    const stopListening = useCallback(() => {
        if (recognitionRef.current) {
            try {
                recognitionRef.current.stop();
            } catch (e) {
                // Ignore error on stop
            }
        }
    }, []);

    const startListening = useCallback(() => {
        if (!isSupported || !enabled) return;
        if (recognitionRef.current) {
            try {
                recognitionRef.current.start();
            } catch (e) {
                // Ignore error on start
            }
        }
    }, [enabled]);

    useEffect(() => {
        if (!isSupported) return;

        const recognition = new SpeechRecognitionAPI();
        recognitionRef.current = recognition;
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onstart = () => setState('listening');
        recognition.onresult = (event: any) => {
            let finalTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) finalTranscript += event.results[i][0].transcript;
            }
            if (finalTranscript) {
                const text = finalTranscript.trim();
                if (text && text !== lastTranscriptRef.current) {
                    lastTranscriptRef.current = text;
                    onSendMessage(text);
                    stopListening();
                }
            }
        };

        recognition.onerror = (event: any) => {
            if (event.error === 'no-speech') return;
            if (event.error === 'not-allowed') {
                setState('disabled');
                return;
            }
            setState('error');
        };

        recognition.onend = () => {
            if (isListeningIntentRef.current && !isAgentSpeaking && !isAgentLoading && enabled) {
                setTimeout(startListening, 100);
            } else if (!isListeningIntentRef.current) {
                setState('idle');
            }
        };

        return () => {
            if (recognitionRef.current) recognitionRef.current.stop();
        };
    }, [onSendMessage, enabled, isAgentSpeaking, isAgentLoading, startListening, stopListening]);

    useEffect(() => {
        if (!enabled) {
            isListeningIntentRef.current = false;
            stopListening();
            // Wrap in setTimeout to avoid synchronous setState inside useEffect
            setTimeout(() => setState('idle'), 0);
            return;
        }

        if (isAgentSpeaking) {
            // Use setTimeout to avoid synchronous setState inside useEffect
            setTimeout(() => setState('speaking'), 0);
            stopListening();
        } else if (isAgentLoading) {
            // Use setTimeout to avoid synchronous setState inside useEffect
            setTimeout(() => setState('processing'), 0);
            stopListening();
        } else if (enabled) {
            isListeningIntentRef.current = true;
            startListening();
        }
    }, [isAgentSpeaking, isAgentLoading, enabled, startListening, stopListening]);

    const toggleVoiceMode = useCallback(() => {
        if (!isSupported) return;
        if (isListeningIntentRef.current) {
            isListeningIntentRef.current = false;
            stopListening();
        } else {
            isListeningIntentRef.current = true;
            startListening();
        }
    }, [startListening, stopListening]);

    return { state, toggleVoiceMode, isSupported };
};
