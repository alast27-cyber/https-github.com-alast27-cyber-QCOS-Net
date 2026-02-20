
// --- Type Definitions ---
export interface ChartData {
  type: 'line' | 'bar' | 'area';
  title: string;
  data: { name: string; value: number }[];
  dataKey: string;
  color?: string;
}

export interface Message {
  id?: string;
  sender: 'user' | 'ai' | 'system';
  text: string;
  attachment?: {
    name: string;
  };
  chartData?: ChartData;
  generatedImage?: string; // Base64 Data URL for generated images
  timestamp?: number;
}

export interface QLangTemplate {
    id: string;
    title: string;
    description: string;
    code: string;
    isCustom?: boolean;
}

// --- Standard Q-Lang Templates ---
export const STANDARD_QLANG_TEMPLATES: QLangTemplate[] = [
    {
        id: 'universe_predictor',
        title: "Universe Prediction Engine",
        description: "Simulates all possible futures via quantum parallelism and collapses to the optimal timeline.",
        code: `// Universe Prediction Engine (Hamiltonian Simulation)
// Based on Hilbert's Paradox & Quantum Parallelism

QREG timeline[5]; // 32 Parallel Universes
CREG reality[5];

// 1. THE GRAND HOTEL (State Preparation)
// Apply Hadamard to all qubits to create equal superposition
// of all 2^N possibilities simultaneously.
OP::H timeline[0];
OP::H timeline[1];
OP::H timeline[2];
OP::H timeline[3];
OP::H timeline[4];

// 2. EVOLUTION (The Universal Hamiltonian)
// Apply phase shifts to mark the 'True' timeline.
// This simulates the laws of physics filtering outcomes.
OP::ORACLE timeline; 

// 3. INTERFERENCE (Amplitude Amplification)
// Constructive interference for the solution,
// Destructive interference for wrong timelines.
OP::DIFFUSION timeline;

// 4. OBSERVATION
// Collapse the wavefunction to a single predicted future.
MEASURE timeline -> reality;`
    },
    {
        id: 'grover',
        title: "Grover's Search",
        description: "Finds marked elements in O(√N) time using amplitude amplification.",
        code: `// Grover's Search Algorithm (3-Qubit)
QREG q[3]; // Search space
CREG c[3]; // Result

// 1. Superposition
OP::H q[0];
OP::H q[1];
OP::H q[2];

// 2. Oracle (Mark state |101>)
OP::CZ q[0], q[2]; 

// 3. Diffusion Operator
OP::H q[0]; OP::H q[1]; OP::H q[2];
OP::X q[0]; OP::X q[1]; OP::X q[2];
OP::H q[2];
OP::CCX q[0], q[1], q[2];
OP::H q[2];
OP::X q[0]; OP::X q[1]; OP::X q[2];
OP::H q[0]; OP::H q[1]; OP::H q[2];

// 4. Measure
MEASURE q -> c;`
    },
    {
        id: 'bb84',
        title: "BB84 QKD Protocol",
        description: "Quantum Key Distribution sender logic (Alice).",
        code: `// BB84 QKD - Role: Alice
QREG qubit[1];
CREG basis[1];
CREG bit[1];

// 1. Generate Random Bit & Basis
EXECUTE OP::GEN_RANDOM -> bit[0];
EXECUTE OP::GEN_RANDOM -> basis[0];

// 2. Encode State
IF (bit[0] == 1) { OP::X qubit[0]; }
IF (basis[0] == 1) { OP::H qubit[0]; }

// 3. Transmit
EXECUTE OP::SEND_QCHANNEL qubit[0] -> TARGET::BOB;`
    },
    {
        id: 'vqe',
        title: "VQE Ansatz",
        description: "Variational Quantum Eigensolver setup for molecular simulation.",
        code: `// VQE UCCSD Ansatz (Simplified)
QREG q[2];
PARAM theta[1]; 

// Hartree-Fock State |01>
OP::X q[1]; 

// Entanglement & Rotation
OP::RY(1.57) q[0];
OP::CNOT q[1], q[0];
OP::RZ(theta[0]) q[0];
OP::CNOT q[1], q[0];
OP::RY(-1.57) q[0];

MEASURE q -> c;`
    },
    {
        id: 'shor',
        title: "Shor's Algorithm Stub",
        description: "Period finding subroutine for integer factorization.",
        code: `// Shor's Period Finding
QREG x[4]; 
QREG f[4]; 
CREG m[4]; 

// 1. Superposition
OP::H x[0]; OP::H x[1]; OP::H x[2]; OP::H x[3];

// 2. Modular Exponentiation Oracle
OP::MOD_EXP(base=7, modulus=15) x, f; 

// 3. IQFT
OP::IQFT x;

// 4. Measure
MEASURE x -> m;`
    }
];


// --- Utility Functions ---
export const fileToText = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsText(file);
        reader.onload = () => {
            resolve(reader.result as string);
        };
        reader.onerror = (error) => reject(error);
    });

export const processZipFile = async (file: File): Promise<string> => {
    const JSZip = (window as any).JSZip;
    if (!JSZip) {
        throw new Error("JSZip library is not loaded. Please ensure the CDN script is in index.html.");
    }
    const jszip = new JSZip();
    const zip = await jszip.loadAsync(file);
    
    let extractedContent = `--- Contents of ZIP file: ${file.name} ---\n\n`;
    const textFilePromises: Promise<void>[] = [];
    const binaryFiles: string[] = [];
    const truncatedFiles: string[] = [];

    const isTextFile = (filename: string): boolean => {
        const textExtensions = [
            '.txt', '.md', '.json', '.csv', '.py', '.js', '.ts', '.tsx', '.html', '.css', 
            '.q-lang', '.xml', '.yaml', '.yml', '.ini', '.log', '.sh', '.bat', '.java', 
            '.c', '.cpp', '.h', '.hpp', '.rs', '.go', '.php', '.rb', '.pl', '.sql'
        ];
        return textExtensions.some(ext => filename.toLowerCase().endsWith(ext));
    };

    zip.forEach((_: string, zipEntry: any) => {
        if (!zipEntry.dir && isTextFile(zipEntry.name)) {
            const promise = zipEntry.async('string').then((content: string) => {
                let finalContent = content;
                if (content.length > 20000) {
                    finalContent = content.substring(0, 20000) + "\n... (file truncated due to size) ...";
                    truncatedFiles.push(zipEntry.name);
                }
                extractedContent += `--- File: ${zipEntry.name} ---\n${finalContent}\n\n`;
            });
            textFilePromises.push(promise);
        } else if (!zipEntry.dir) {
            binaryFiles.push(zipEntry.name);
        }
    });

    await Promise.all(textFilePromises);
    
    if (binaryFiles.length > 0) {
        extractedContent += `--- Binary/Unsupported Files (contents not included): ---\n${binaryFiles.join('\n')}\n\n`;
    }

    if (truncatedFiles.length > 0) {
        extractedContent += `--- Note: The following files were truncated due to size: ---\n${truncatedFiles.join('\n')}\n\n`;
    }

    extractedContent += `--- End of ZIP file contents ---`;
    return extractedContent;
};

export const fileToBase64 = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      const base64String = (reader.result as string).split(',')[1];
      resolve(base64String);
    };
    reader.onerror = (error) => reject(error);
  });

export const extractSpeakableText = (text: string) => {
  const parts = text.split(/(```(?:\w+)?[\s\S]*?```)/g);
  return parts.filter(part => !part.startsWith('```')).join(' ').replace(/\s+/g, ' ').trim();
};

export const parseGeneratedFiles = (text: string): { [filename: string]: string } => {
    const files: { [filename: string]: string } = {};
    // Regex matches: FILE: filename.ext ... ```code```
    // It handles optional spaces and different newline formats
    const fileRegex = /(?:^|\n)(?:###\s*)?FILE:\s*([a-zA-Z0-9_./-]+)\s*\n+```(?:\w+)?\n([\s\S]*?)```/g;
    
    let match;
    while ((match = fileRegex.exec(text)) !== null) {
        const fileName = match[1].trim();
        const content = match[2].trim();
        files[fileName] = content;
    }
    return files;
};

export const daisyBellLyrics = `Daisy, Daisy,
Give me your answer, do!
I'm half crazy,
All for the love of you!

It won't be a stylish marriage,
I can't afford a carriage,
But you'll look sweet upon the seat
Of a bicycle built for two!`;

export const daisyBellMelody: { text: string; pitch: number; rate: number; delay: number }[] = [
    { text: "Daisy,", pitch: 1.2, rate: 0.8, delay: 0 },
    { text: "Daisy,", pitch: 1.1, rate: 0.8, delay: 700 },
    { text: "Give me your answer, do!", pitch: 1.2, rate: 0.8, delay: 700 },
    { text: "I'm half crazy,", pitch: 1.3, rate: 0.8, delay: 1800 },
    { text: "All for the love of you!", pitch: 1.2, rate: 0.8, delay: 1500 },
    { text: "It won't be a stylish marriage,", pitch: 1.1, rate: 0.9, delay: 2000 },
    { text: "I can't afford a carriage,", pitch: 1.0, rate: 0.9, delay: 2000 },
    { text: "But you'll look sweet", pitch: 1.2, rate: 0.8, delay: 1200 },
    { text: "upon the seat", pitch: 1.1, rate: 0.8, delay: 1200 },
    { text: "Of a bicycle built for two!", pitch: 1.2, rate: 0.8, delay: 1200 },
];

// --- Statistical & Predictive Utils ---

export const calculateBasicStats = (data: number[]) => {
    if (data.length === 0) return { mean: 0, median: 0, min: 0, max: 0, stdDev: 0 };
    
    const sorted = [...data].sort((a, b) => a - b);
    const sum = data.reduce((a, b) => a + b, 0);
    const mean = sum / data.length;
    const min = sorted[0];
    const max = sorted[sorted.length - 1];
    
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    
    const variance = data.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / data.length;
    const stdDev = Math.sqrt(variance);
    
    return { mean, median, min, max, stdDev };
};

export const linearRegressionPrediction = (data: number[], steps: number) => {
    const n = data.length;
    if (n < 2) return [];

    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    for (let i = 0; i < n; i++) {
        sumX += i;
        sumY += data[i];
        sumXY += i * data[i];
        sumXX += i * i;
    }

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    const predictions = [];
    for (let i = 1; i <= steps; i++) {
        predictions.push(slope * (n - 1 + i) + intercept);
    }
    return predictions;
};

// --- TTS Integration (ElevenLabs) ---
const ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech";
// Default to 'Adam' (deep, calm voice) if no specific Lex ID is provided
const DEFAULT_VOICE_ID = "21m00Tcm4obsikG9TzGS"; 

export const playAgentVoice = async (text: string, voiceId: string = DEFAULT_VOICE_ID, apiKey?: string): Promise<HTMLAudioElement | null> => {
    if (!apiKey) {
        console.warn("ElevenLabs API Key missing. Falling back to system TTS.");
        return null;
    }

    try {
        const response = await fetch(`${ELEVENLABS_API_URL}/${voiceId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'xi-api-key': apiKey
            },
            body: JSON.stringify({
                text: text,
                model_id: "eleven_multilingual_v2",
                voice_settings: {
                    stability: 0.75,
                    similarity_boost: 0.75,
                    style: 0.0,
                    use_speaker_boost: true
                }
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(`ElevenLabs API Error: ${response.status} ${response.statusText} - ${JSON.stringify(errorData)}`);
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        return audio;
    } catch (e) {
        console.error("Agent Q Voice Generation Failed:", e);
        return null;
    }
};
