EPR-Based EKS Distribution (EED) Protocol

The EED Protocol is the Quantum Key Distribution (QKD) mechanism used to establish and synchronize the Entangled Key State (EKS) between the Quantum Authority Node (QAN) and a Decentralized Quantum Node (DQN). The EKS is the root of trust, providing the shared, quantum-secured state used for CHIPS packet signing and verification.

This protocol utilizes Entangled Pairs (EPR pairs), ensuring security is based on the No-Cloning Theorem and the instantaneous state collapse upon measurement.

1. Protocol Goals and Key State ($K_{EKS}$)

Goal: To establish a secure, shared bit string (the $K_{EKS}$ master key) whose integrity is guaranteed by the laws of physics.

Key State: The EKS is a stream of Bell states, typically $| \Phi^+ \rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$, where one qubit ($Q_A$) is held by the QAN and the other ($Q_B$) is sent to the DQN.

Final EKS ($K_{EKS}$): The classical bit string resulting from the sequential measurement of these entangled pairs.

2. EED Protocol Flow (Quantum Channel)

The establishment process relies on a dedicated Quantum Channel for qubit transmission and a secure Classical Channel for synchronization and sifting.

Stage 1: Preparation and Entanglement

QAN Prepares Pairs: The QAN prepares a large number ($N$) of EPR pairs ($Q_{Ai}, Q_{Bi}$).

QAN Retains $Q_A$: The QAN keeps $Q_A$ in its local quantum memory.

QAN Distributes $Q_B$: The QAN transmits $Q_B$ to the DQN over the quantum network backbone (fiber or free-space entanglement links).

Stage 2: Measurement and Sifting (Classical Channel)

The classical channel (CHIPS control packets) is used to establish the correlation between the QAN and DQN measurements.

DQN Measurement: The DQN measures all received $Q_{Bi}$ qubits in the computational basis ($\{Z\}$ basis) and records the classical bit string, $S_{DQN}$.

QAN Measurement: The QAN measures its retained $Q_{Ai}$ qubits in the same basis and records $S_{QAN}$.

Basis Verification: The QAN and DQN publicly exchange the sequence index of the qubits they successfully measured (the "sifting" process). They discard all pairs where a qubit was lost or decohered during transit.

Raw Key Generation: The remaining bits form the raw keys $R_{QAN}$ and $R_{DQN}$. Due to entanglement, $R_{QAN} = R_{DQN}$ (ideally).

3. Security and Final Key Agreement

Stage 3: Error Estimation and Privacy Amplification

Error Estimation (Quantum Bit Error Rate - QBER): The QAN and DQN publicly compare a small, randomly selected subset of their raw keys (the test sample). The percentage of differing bits is the QBER.

Eavesdropping Threshold: If the QBER exceeds a pre-defined threshold ($\tau$, e.g., 5%), it indicates that an eavesdropper ($\text{EVE}$) has likely performed a measurement on the channel, collapsing the state and introducing errors. The protocol is immediately aborted and the link is quarantined.

Error Correction: The remaining bits are processed using classical error correction techniques (e.g., Cascade protocol) to reconcile any residual classical errors introduced by measurement imperfections.

Privacy Amplification: A universal hash function is applied to the reconciled key to distill the final secret key ($K_{EKS}$) and reduce any potential information $\text{EVE}$ may have gained.

4. Final EKS State Synchronization

The final, distilled secret bit string is the $K_{EKS}$.

Synchronization: This key is now simultaneously active on both nodes.

EKS Reference: The QAN assigns a time-based index or sequence number to this key ($EKS_{ref}$) and disseminates it to the DQN via a secured classical control packet.

Usage: The DQN and QAN use this $K_{EKS}$ bit string as the shared secret for generating the $\text{HMAC}_{\text{Dilithium-2}}$ signatures used in the CHIPS protocol for packet integrity. A new EKS is typically established after a fixed time interval or a set number of transactions.