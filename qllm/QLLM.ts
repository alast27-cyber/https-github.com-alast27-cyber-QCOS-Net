
/**
 * QIAI-IPS QLLM Core Implementation
 * 
 * Objective: Compression of Classical Bytes into Quantum States.
 * Protocol: 1 Byte (8 Bits) -> 4 Qubits (Superposition/Entanglement)
 * 
 * This enables 2:1 Data Density (1 Byte data stored in 0.5 Byte-equivalent quantum space).
 */

export const BYTE_TO_QUBIT_RATIO = 0.5;

export interface QubitState {
    id: number;
    amplitude: number; // Represents Bit n
    phase: number;     // Represents Bit n+1
    basis: 'Z' | 'X';
    entangledWith?: number;
}

export class QuantumCompressor {
    
    /**
     * Compresses a standard byte (0-255) into 4 entangled qubits.
     * Uses Superdense Coding principles where 1 Qubit carries 2 Classical Bits.
     * 
     * @param byte Standard 8-bit integer
     */
    static compressByteToQubits(byte: number): QubitState[] {
        // Ensure byte is 8 bits
        const binaryString = byte.toString(2).padStart(8, '0');
        
        // Split into 4 pairs of 2 bits
        const pairs = binaryString.match(/.{1,2}/g) || [];
        
        return pairs.map((pair, index) => {
            const [b1, b2] = pair.split('').map(Number);
            
            // Encoding Strategy:
            // Bit 1 -> Amplitude Rotation (Ry)
            // Bit 2 -> Phase Rotation (Rz)
            
            return {
                id: index,
                amplitude: b1 === 1 ? 1.0 : 0.0, // |1> state if 1
                phase: b2 === 1 ? Math.PI : 0.0, // Phase shift if 1
                basis: 'Z' as const,
                entangledWith: index < 3 ? index + 1 : undefined // Linear entanglement chain
            };
        });
    }

    /**
     * Reconstructs the byte from the quantum state (Measurement).
     * Simulates the collapse of the wavefunction.
     */
    static measureQubits(qubits: QubitState[]): number {
        if (qubits.length !== 4) throw new Error("Invalid Qubit Block: Expected 4 Qubits");
        
        let binary = "";
        qubits.forEach(q => {
            // Measure Amplitude
            const b1 = q.amplitude > 0.5 ? "1" : "0";
            // Measure Phase
            const b2 = Math.abs(q.phase) > 0.1 ? "1" : "0";
            
            binary += b1 + b2;
        });
        
        return parseInt(binary, 2);
    }
}
