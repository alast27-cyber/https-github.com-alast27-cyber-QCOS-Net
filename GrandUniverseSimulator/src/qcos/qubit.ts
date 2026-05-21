// /src/qcos/qubit.ts

export interface QubitState {
    id: number;
    state: number;
    phase: number;
}

export class QubitProtocol {
    private qubits: QubitState[] = [];

    constructor(count: number) {
        this.qubits = Array.from({ length: count }).map((_, i) => ({
            id: i,
            state: Math.random(),
            phase: Math.random() * 360
        }));
    }

    public getQubits(): QubitState[] {
        return this.qubits;
    }

    public updateQubits(): void {
        this.qubits = this.qubits.map(q => ({
            ...q,
            state: Math.max(0, Math.min(1, q.state + (Math.random() - 0.5) * 0.1)),
            phase: (q.phase + 5) % 360
        }));
    }
}
