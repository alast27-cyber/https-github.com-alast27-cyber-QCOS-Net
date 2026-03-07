// /src/qcos/isa.ts

export enum OpCode {
    H = 'H', // Hadamard
    CNOT = 'CNOT', // Controlled-NOT
    MEASURE = 'MEASURE',
    RESET = 'RESET'
}

export interface Instruction {
    op: OpCode;
    target: string;
    control?: string;
}

export class InstructionSetArchitecture {
    private program: Instruction[] = [];

    public addInstruction(instr: Instruction): void {
        this.program.push(instr);
    }

    public execute(): void {
        console.log("[QCOS] Executing ISA Program:", this.program);
        this.program = []; // Clear program after execution
    }

    public executeHandshake(protocol: string): void {
        console.log(`[QCOS] Executing ${protocol} Handshake...`);
    }

    public getThroughput(): number {
        return 800 + Math.random() * 200;
    }
}
