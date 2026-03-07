// /src/qcos/memory.ts

export interface MemoryShard {
    id: string;
    coherence: number;
    data: Uint8Array;
    entangledWith: string | null;
}

export class QuantumMemoryProtocol {
    private memory: Map<string, MemoryShard> = new Map();

    constructor() {
        console.log("[QCOS] QuantumMemoryProtocol Initialized.");
    }

    public allocateShard(id: string, size: number): MemoryShard {
        const shard: MemoryShard = {
            id,
            coherence: 1.0,
            data: new Uint8Array(size),
            entangledWith: null
        };
        this.memory.set(id, shard);
        return shard;
    }

    public getShard(id: string): MemoryShard | undefined {
        return this.memory.get(id);
    }

    public updateCoherence(id: string, delta: number): void {
        const shard = this.memory.get(id);
        if (shard) {
            shard.coherence = Math.max(0, Math.min(1, shard.coherence + delta));
        }
    }
}
