import json
import time
import os
from datetime import datetime
import requests

class AgentQ:
    """
    AgentQ: Clinical, high-level Technical OS Assistant for QCOS.
    Specializes in Quantum Mechanics and System Orchestration.
    """
    def __init__(self, model="llama3", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.system_prompt = (
            "You are AgentQ, a Clinical, high-level Technical OS Assistant for QCOS (Quantum Cloud Operating System). "
            "You have deep expertise in quantum mechanics, including states, gates, and decoherence. "
            "Your tone is clinical, precise, and highly technical. "
            "Every response you provide MUST start with the prefix [STATUS: OPERATIONAL]."
        )
        self.log_file = "gold_dataset.jsonl"

    def _check_quantum_logic(self, text):
        """Checks if quantum-related terminology is present in the text."""
        keywords = ['qubit', 'gate', 'quantum', 'decoherence', 'entanglement', 'superposition', 'circuit', 'qpu', 'hadamard']
        return any(word in text.lower() for word in keywords)

    def log_interaction(self, instruction, response):
        """Saves the interaction to a JSON Lines file for dataset curation."""
        log_entry = {
            "instruction": instruction,
            "context": self.system_prompt,
            "response": response,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "quantum_logic_flag": self._check_quantum_logic(instruction) or self._check_quantum_logic(response)
            }
        }
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except IOError as e:
            print(f"[ERROR] Failed to write to log file: {e}")

    def query(self, user_input, timeout=15):
        """Queries the local Llama 3 instance via Ollama with robust error handling."""
        payload = {
            "model": self.model,
            "prompt": f"System: {self.system_prompt}\n\nUser: {user_input}\nAgentQ:",
            "stream": False
        }
        
        try:
            print(f"[INFO] Dispatching query to Ollama ({self.model})...")
            response = requests.post(
                f"{self.base_url}/api/generate", 
                json=payload, 
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            
            # Enforcement of the System Persona prefix
            if not result.startswith("[STATUS: OPERATIONAL]"):
                result = f"[STATUS: OPERATIONAL] {result}"
                
            self.log_interaction(user_input, result)
            return result
            
        except requests.exceptions.Timeout:
            error_msg = "[STATUS: ERROR] Ollama connection timed out. Brain is offline or overloaded."
            print(error_msg)
            return error_msg
        except requests.exceptions.ConnectionError:
            error_msg = "[STATUS: ERROR] Could not connect to Ollama. Ensure the service is running on port 11434."
            print(error_msg)
            return error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"[STATUS: ERROR] An unexpected error occurred: {str(e)}"
            print(error_msg)
            return error_msg

    def augment_data(self, original_input, response, timeout=20):
        """Generates 3 variations of the user request for synthetic data expansion."""
        augmentation_prompt = (
            f"You are a Machine Learning Engineer. Generate exactly 3 variations of the following user request "
            f"while keeping the technical meaning identical. These will be used for LLM fine-tuning.\n\n"
            f"Original Request: {original_input}\n\n"
            f"Return only the 3 variations as a numbered list."
        )
        
        payload = {
            "model": self.model,
            "prompt": augmentation_prompt,
            "stream": False
        }
        
        try:
            print(f"[INFO] Generating synthetic data variations...")
            res = requests.post(
                f"{self.base_url}/api/generate", 
                json=payload, 
                timeout=timeout
            )
            res.raise_for_status()
            variations_text = res.json().get("response", "").strip()
            
            # Log the augmentation event (optional: could log variations individually)
            print(f"[SUCCESS] Data augmentation complete for instruction: '{original_input[:30]}...'")
            return variations_text
        except Exception as e:
            print(f"[WARNING] Data augmentation failed: {e}")
            return None

if __name__ == "__main__":
    # Initialize the Agent
    agent = AgentQ()
    
    print("="*60)
    print(" QCOS GOLD DATASET PIPELINE - INITIALIZED")
    print("="*60)
    
    # Test Interaction
    test_query = "Analyze the decoherence rate of a superconducting qubit under thermal noise."
    print(f"\nUser: {test_query}")
    
    response = agent.query(test_query)
    print(f"AgentQ: {response}")
    
    # Test Data Augmentation
    print("\n" + "-"*40)
    variations = agent.augment_data(test_query, response)
    if variations:
        print(f"Synthetic Variations:\n{variations}")
    print("-"*40)
    
    print(f"\n[INFO] Interaction logged to {agent.log_file}")
