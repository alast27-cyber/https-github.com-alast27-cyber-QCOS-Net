class ContextualReasoningEngine:
    def __init__(self, llm_bridge):
        self.bridge = llm_bridge
        self.context_window = []

    def update_context(self, event):
        self.context_window.append(event)
        if len(self.context_window) > 10:
            self.context_window.pop(0)

    def reason(self, prompt):
        context_str = "\n".join(self.context_window)
        full_prompt = f"Context:\n{context_str}\n\nQuery: {prompt}"
        # Simulate LLM call
        decision = self.bridge.query(full_prompt)
        return decision
