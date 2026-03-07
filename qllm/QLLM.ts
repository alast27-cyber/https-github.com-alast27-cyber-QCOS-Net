export const BYTE_TO_QUBIT_RATIO = 0.5;

export class QLLM_Interface {
    private isConnected: boolean = false;

    public connect() {
        console.log("Connecting to QLLM Core...");
        this.isConnected = true;
        return this.isConnected;
    }

    public encodeData(data: string): number[] {
        if (!this.isConnected) throw new Error("QLLM not connected.");
        return data.split('').map(char => char.charCodeAt(0) * BYTE_TO_QUBIT_RATIO);
    }

    public decodeData(qubits: number[]): string {
        if (!this.isConnected) throw new Error("QLLM not connected.");
        return qubits.map(q => String.fromCharCode(Math.round(q / BYTE_TO_QUBIT_RATIO))).join('');
    }

    public simulateEntanglement(qubitA: number, qubitB: number): [number, number] {
        const avg = (qubitA + qubitB) / 2;
        return [avg, avg];
    }
    
    public measure(state: number[]): number {
        const binary = state.map(q => q > 0.5 ? '1' : '0').join('');
        return parseInt(binary, 2);
    }
}
