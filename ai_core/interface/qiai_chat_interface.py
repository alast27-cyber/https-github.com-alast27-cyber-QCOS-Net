import torch
import logging
# Absolute import to match the project root
from ai_core.bridge.qiai_ips_bridge import QIAI_IPS_Bridge

# Configure UI Logging
logging.basicConfig(level=logging.ERROR) 

class QIAI_Terminal_Assistant:
    def __init__(self):
        print("\n" + "="*50)
        print("   AGENT Q: COMMAND & CONTROL INTERFACE (CCI)")
        print("="*50)
        print("Initializing Quantum-Semantic Link...")
        
        # Load the Bridge
        self.bridge = QIAI_IPS_Bridge()
        
        print("\nStatus: ONLINE")
        print("Type 'metrics' to simulate a system event, or just talk to Agent Q.")
        print("Type 'exit' to shutdown.")
        print("-" * 50)

    def simulate_os_event(self):
        """Simulates a high-stress kernel event."""
        metrics = torch.tensor([[0.9, 0.2]])
        logs = "CRITICAL_WARNING: Memory fragmentation at 88%. Page table walk latency high."
        
        print(f"\n[SYSTEM EVENT]: {logs}")
        # Use .forward() as defined in our bridge
        decision, intent_vec, mode = self.bridge.forward(metrics, logs, debug=False)
        
        return decision, mode

    def chat_loop(self):
        while True:
            try:
                user_input = input("\n[USER]> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit']:
                    print("Shutting down Q-IAI Interface...")
                    break

                if user_input.lower() == 'metrics':
                    v_score, mode = self.simulate_os_event()
                    print(f"[AGENT Q]: Event Processed. Mode: {mode}. V-Score: {v_score:.4f}")
                    continue

                # General Chat Logic via the LLM Supervisor inside the bridge
                prompt = (f"Context: You are the Semantic Supervisor of an AI-Native OS. "
                          f"User asks: '{user_input}'. Resolve as the system's higher consciousness.")
                
                # Using the bridge's reasoner pipeline
                response = self.bridge.llm_reasoner(prompt, max_new_tokens=100, truncation=True)[0]['generated_text']
                
                # Extracting the response after the prompt
                clean_response = response.split("higher consciousness.")[-1].strip()
                print(f"\n[AGENT Q]: {clean_response}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n[INTERFACE ERROR]: {e}")

if __name__ == "__main__":
    assistant = QIAI_Terminal_Assistant()
    assistant.chat_loop()
