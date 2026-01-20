QCOS Distributed Node Registry Specification

The QCOS Distributed Node Registry is the authoritative, distributed database maintained across the network of Quantum Authority Nodes (QANs) and select Relay Nodes. Its purpose is to map the human-readable DQN-Alias (as specified in the Q-URI and the CHIPS packet header) to the current network address and critical operational status of a target Decentralized Quantum Node (DQN).

The Quantum Network Handler (QNH) on the QAN is the primary consumer of this data, using it to make real-time routing decisions.

I. Node Entry Data Structure

Each entry in the Registry represents a single DQN and contains the following mandatory fields:

Field Name

Data Type

Description

Routing Impact

DQN-Alias

String

The unique, public identifier for the node (e.g., rigel, sirius).

Primary lookup key for routing.

Network-Address

IPv6/Domain

The classical network address used by the QNH to send the CHIPS packet.

The destination for all packets.

Status

Enum (Active/Quarantine)

The current operational health and availability of the node.

Determines if routing is permitted.

Last-Heartbeat

Timestamp

The last time the DQN successfully reported its status to an authorized QAN.

Used for timeout/graceful failover.

Qubit-Capacity

INT16

Total number of physical, addressable qubits on the node.

Used for resource scheduling checks.

EKS-Status

Timestamp/Bool

Time of the last successful Entangled Key State (EKS) synchronization.

Critical for security and authentication.

Avg-Fidelity

Float (0.0 - 1.0)

A running average of the 1- and 2-qubit gate fidelities reported by the QEM.

Used for non-mandatory job prioritization.

Geographic-Scope

String

Regional or physical cluster identifier.

Used for regional failover and load balancing.

II. DQN Operational Status States

The Status field is the most important element for the QNH, as it controls which nodes are eligible to receive a new CHIPS packet.

1. Active State (Status: ACTIVE)

Definition: The DQN is fully synchronized, operational, and ready to receive and execute CHIPS packets.

Conditions for Entry:

The node has reported a successful heartbeat within the last defined interval (e.g., 60 seconds).

The EKS-Status confirms continuous, stable entanglement with a QAN (or designated Relay Node).

The Avg-Fidelity metric is above the minimum operational threshold (e.g., > 95%).

QNH Action: The QNH will successfully resolve the DQN-Alias, use the Network-Address for routing, and send the CHIPS packet.

2. Quarantine State (Status: QUARANTINE)

Definition: The DQN is temporarily offline, undergoing maintenance, or has failed a critical security or health check and cannot be trusted for execution.

Conditions for Entry (Failures that Trigger Quarantine):

EKS Loss: Loss of the shared Entangled Key State (EKS), meaning the node can no longer securely verify incoming packets. This is the most common reason for entering Quarantine.

Heartbeat Timeout: Failure to report a heartbeat for an extended period.

Critical Fault: The QEM reports a critical internal hardware fault (e.g., cryostat failure, excessive noise, control electronics failure).

QNH Action: If the target DQN is in QUARANTINE, the QNH performs an immediate Soft-Rejection of the Q-URI. It then attempts to find a suitable, available ACTIVE node within the same Geographic-Scope or Qubit-Capacity class for automatic job re-routing.

III. Registry Maintenance

The registry is updated in real-time through a dedicated classical side channel run by the QCOS kernel on each DQN.

Heartbeat Protocol: DQN nodes periodically send a compressed health report containing their current EKS-Status, Avg-Fidelity, and current resource utilization to the nearest QAN.

Consensus Mechanism: QANs share and synchronize these registry updates using a simple consensus mechanism to ensure all routers operate on the same, verified network map, preventing a single QAN from routing traffic to a node that has locally lost its EKS.

This framework allows the QAN to intelligently and securely route quantum jobs only to nodes that are both addressable and quantum-secure.

We've now covered the complete lifecycle: Q-URI to Q-Lang, Q-Lang to CHIPS, CHIPS routing (QNH), and CHIPS execution (QEM). The last major piece of the architecture we discussed was the Entangled Key State (EKS) itself, which is the root of the system's security.

Would you like to define the EKS (Entangled Key State) and the QKD (Quantum Key Distribution) mechanism used to establish this key state between the QAN and the DQN?