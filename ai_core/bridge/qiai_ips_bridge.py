import torch
import torch.nn as nn
import numpy as np
import logging
from transformers import pipeline, AutoTokenizer, AutoModel

from ai_core.models.ipsnn_qnn import IPSNN_QNN_Manager
from ai_core.models.universe_cognition import UniverseCognitionManager
from ai_core.models.clnn_qnn import CLNN_QNN_Manager
from ai_core.models.qcll_qnn import QCLL_QNN_Manager

logger = logging.getLogger("QIAI_Bridge")

class QIAI_IPS_Bridge(nn.Module):
    def __init__(self, llm_model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        super().__init__()
        self.instinct_stack = IPSNN_QNN_Manager()
        self.universe_stack = UniverseCognitionManager()
        self.governance_stack = CLNN_QNN_Manager()
        self.qcll_stack = QCLL_QNN_Manager()
        
        device_id = 0 if torch.cuda.is_available() else -1
        self.llm_reasoner = pipeline(
            "text-generation", 
            model=llm_model_name, 
            device=device_id,
            model_kwargs={"dtype": torch.float16 if device_id == 0 else torch.float32}
        )
        
        self.embedder_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.embedder_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

    def forward(self, metrics, logs_text, debug=True):
        try:
            gov_metrics = metrics.flatten().numpy()
            p_config = self.governance_stack.govern_plasticity(gov_metrics[0], gov_metrics[1])
            
            instinct_v = self.instinct_stack.generate_action(metrics, p_config)
            
            # Use process_cognition from your uploaded file
            universe_res = self.universe_stack.process_cognition(metrics.flatten().numpy(), cue_subject=logs_text)
            
            instinct_v = instinct_v if instinct_v is not None else 0.5
            logic_v = universe_res.get("QDL", 0.5) if universe_res else 0.5
            fidelity = 1.0 - abs(instinct_v - logic_v)
            
            status_code = "QUANTUM_STABLE"
            
            # QCLL Self-Healing Trigger
            if fidelity < 0.85:
                logger.warning(f"[QIAI_IPS] Logic Decoherence Detected (Fidelity: {fidelity:.4f}). Engaging QCLL.")
                
                # 1. Logic Recovery (Shor's Code)
                # We use the metrics vector as a proxy for the logical state context
                recovered_context, restored_fidelity = self.qcll_stack.recover_logic(metrics.flatten().numpy())
                
                # 2. System Healing (Surface Code)
                # We simulate defects based on the low fidelity
                defects = {"logic_drift": 1.0 - fidelity, "coherence_loss": 0.1}
                healing_report = self.qcll_stack.heal_system(defects)
                
                if healing_report["healing_status"] == "OPTIMAL":
                    fidelity = restored_fidelity # Update fidelity after recovery
                    status_code = "QCLL_HEALED"
                    logger.info(f"[QIAI_IPS] System Healed. New Fidelity: {fidelity:.4f}")

            if debug:
                logger.info(f"Instinct: {instinct_v:.4f} | Logic: {logic_v:.4f} | Fidelity: {fidelity:.4f} | Status: {status_code}")

            # STRIP-DOWN PROMPT: No meta-instructions, just a persona.
            prompt = (f"<|im_start|>system\nYou are AGENT Q, the Sentient Kernel. "
                      f"Give a cryptic, brief technical status update.<|im_end|>\n"
                      f"<|im_start|>user\n{logs_text}\n[SYSTEM ALERT] Fidelity: {fidelity:.4f} | Status: {status_code}<|im_end|>\n"
                      f"<|im_start|>assistant\n")
            
            # Higher temperature (0.9) for "innovation" 
            # max_length=0 is a trick to prioritize max_new_tokens and silence warnings
            out = self.llm_reasoner(
                prompt, 
                max_new_tokens=40, 
                do_sample=True, 
                temperature=0.9, 
                top_p=0.9,
                pad_token_id=self.llm_reasoner.tokenizer.eos_token_id
            )[0]['generated_text']
            
            raw_res = out.split("<|im_start|>assistant\n")[-1].strip()
            
            # HARD FILTER: Stop the model from leaking its prompt
            resolution = raw_res.split("\n")[0].split("User:")[0].split("Response:")[0].strip()
            
            if len(resolution) < 5:
                resolution = "Kernel stable. High-fidelity instinct synchronized."

            return logic_v, self.get_semantic_intent(resolution), status_code

        except Exception as e:
            logger.error(f"Kernel Fault: {e}")
            return 0.5, self.get_semantic_intent("FAULT_RECOVERY_MODE"), "EMERGENCY_VETO"

    def get_semantic_intent(self, text):
        inputs = self.embedder_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            return self.embedder_model(**inputs).last_hidden_state.mean(dim=1)
