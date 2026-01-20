CHIPS Response Packet Status Codes (INT16)

The Execution Status field in the Response Packet provides specific context about the job's outcome, enabling the Quantum Authority Node (QAN) to triage execution problems automatically.

2xx Series: Success and Acknowledgment

These codes indicate that the quantum job was completed and a classical result was returned.

Code

Status Name

Description

200

OK

Standard Success. The Q-Lang script executed completely, and the final measurement and result encapsulation were successful.

201

PARTIAL_SUCCESS

Job completed with minor, non-critical warnings. The final result is valid, but the log (Execution Log field) should be checked for warnings (e.g., minor gate sequence warnings, higher-than-normal thermal variance).

4xx Series: Quantum Execution and Q-Lang Errors

These codes indicate a job failure due to issues with the quantum state, the script itself, or resource constraints.

Code

Status Name

Description

401

DECOHERENCE_ABORT

The QEM detected critical, runtime decoherence in the working qubits that exceeded the algorithm's tolerance threshold, leading to an immediate abort.

402

QUBIT_OVERFLOW

The Q-Lang script (as defined by Qubit Count in the Request Packet) requested more qubits than the target DQN could allocate at the time of execution.

403

TIMEOUT_EXCEEDED

The job's execution time exceeded the Execution Timeout (INT32) value specified in the original CHIPS Request Packet.

404

INVALID_QLANG

A fatal semantic or syntax error was found in the compiled Q-Lang payload during parsing by the Quantum-Lang Execution Engine (QLEE).

405

LOW_FIDELITY_REJECT

The job ran to completion, but the fidelity (accuracy) of the final measured state fell below the minimum acceptable threshold required by the target algorithm.

406

RESOURCE_LOCK_FAIL

The QEM was unable to lock all necessary classical or quantum registers due to a conflict or ongoing maintenance cycle.

5xx Series: DQN System and Hardware Errors

These codes indicate a severe failure originating from the DQN's internal operating system (QCOS) or physical hardware.

Code

Status Name

Description

500

QEM_CRASH

A critical failure in the Quantum Engine Manager (QEM) software, resulting in an unrecoverable crash and job termination.

501

THERMAL_QUARANTINE

The DQN entered a thermal safety state (e.g., cryostat warming up, cooling system failure) and aborted execution to protect the physical hardware.

502

EKS_SYNC_LOSS

The DQN lost synchronization with the Quantum Authority Node's (QAN) Entangled Key State (EKS) and entered a Quarantine State, aborting the pending job.

503

HARDWARE_FAULT

A physical fault was detected in the quantum processor (e.g., laser failure, control electronics error).