"""
Part 7: Using Large Language Model (LLM) APIs & Local Open Models
Demonstrates:
1. OpenAI API (GPT-4o / GPT-4o-mini) using the modern v1.x OpenAI client.
2. Anthropic Claude API (claude-3-5-sonnet) using official messages endpoint.
3. Ollama Local Models (Llama-3, Mistral, Phi-3) via local REST API (http://localhost:11434).
4. Three core LLM tasks:
   - Text Generation
   - Document Summarization
   - Contextual Question Answering
5. Resilient Mock & Fallback Mode: Seamless execution even without active API keys.
"""

import os
import json
import urllib.request
import urllib.error

# Optional imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ----------------------------------------------------------------------
# 1. UNIFIED LLM INTERFACE
# ----------------------------------------------------------------------
class LLMClient:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        if OPENAI_AVAILABLE and self.openai_key:
            self.openai_client = OpenAI(api_key=self.openai_key)
        else:
            self.openai_client = None

    def call_openai(self, system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
        """Invokes OpenAI Chat Completions API."""
        if not self.openai_client:
            return self._mock_llm_response("OpenAI (gpt-4o-mini)", system_prompt, user_prompt)

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[OpenAI API Error: {e}] -> falling back to simulator")
            return self._mock_llm_response("OpenAI", system_prompt, user_prompt)

    def call_anthropic(self, system_prompt: str, user_prompt: str, model: str = "claude-3-5-sonnet-20241022") -> str:
        """Invokes Anthropic Claude API."""
        if not self.anthropic_key:
            return self._mock_llm_response("Anthropic (Claude 3.5 Sonnet)", system_prompt, user_prompt)

        try:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": model,
                "max_tokens": 300,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}]
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return data["content"][0]["text"]
        except Exception as e:
            print(f"[Anthropic API Error: {e}] -> falling back to simulator")
            return self._mock_llm_response("Anthropic", system_prompt, user_prompt)

    def call_ollama(self, prompt: str, model: str = "llama3") -> str:
        """Invokes local open-source models running via Ollama (Free & Local)."""
        url = f"{self.ollama_base_url}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return data.get("response", "")
        except Exception:
            return self._mock_llm_response(f"Ollama Local ({model})", "Local Assistant", prompt)

    def _mock_llm_response(self, provider: str, system: str, user: str) -> str:
        """Provides realistic mock responses tailored to the specific task when API keys or local server are absent."""
        u_lower = user.lower()
        if "skip" in u_lower or "resnet" in u_lower:
            return (
                f"[{provider} Generated Response]\n"
                "Skip connections (residual shortcuts) formulate the layer mapping as H(x) = F(x) + x.\n"
                "During backpropagation, the gradient is computed as dH/dx = dF/dx + 1. The '+ 1' term "
                "acts as an unobstructed identity highway that allows error gradients to flow directly "
                "to earlier layers without being diminished by repeated matrix multiplications, effectively "
                "eliminating the vanishing gradient problem in ultra-deep networks."
            )
        elif "summar" in u_lower or "abstract" in u_lower:
            return (
                f"[{provider} Generated Response]\n"
                "Summary: The authors propose a residual learning framework that reforms deep layers as "
                "residual functions with reference to layer inputs. Empirical evaluations demonstrate "
                "that these residual networks optimize effectively and achieve superior accuracy "
                "as depth increases significantly."
            )
        elif "lstm" in u_lower or "gru" in u_lower or "difference" in u_lower:
            return (
                f"[{provider} Generated Response]\n"
                "Answer: The primary architectural difference lies in gate structure and state management:\n"
                "1. LSTM maintains two separate states (Cell state c_t and Hidden state h_t) managed by 3 gates "
                "(Forget, Input, and Output).\n"
                "2. GRU merges the cell and hidden state into a single h_t and uses only 2 gates (Reset and Update).\n"
                "As a result, GRU has ~25% fewer parameters and trains faster, while LSTM often performs "
                "better on longer, complex dependency tasks."
            )
        else:
            return (
                f"[{provider} Generated Response]\n"
                "Response: Deep learning architectures rely on gradient-based backpropagation, layer normalization, "
                "and regularization to achieve strong generalization across unseen test distributions."
            )



# ----------------------------------------------------------------------
# 2. DEMONSTRATION OF THREE CORE LLM TASKS
# ----------------------------------------------------------------------
def run_llm_tasks():
    print("=" * 65)
    print(" EXPERIMENTING WITH LLM APIs (OpenAI, Anthropic, Ollama)")
    print("=" * 65)

    client = LLMClient()
    print("API Key Configuration Status:")
    print(f"  * OPENAI_API_KEY:    {'Configured' if client.openai_key else 'Not set (Simulation mode enabled)'}")
    print(f"  * ANTHROPIC_API_KEY: {'Configured' if client.anthropic_key else 'Not set (Simulation mode enabled)'}")
    print(f"  * OLLAMA Endpoint:   {client.ollama_base_url}")

    # Task 1: Text Generation
    print("\n" + "-" * 55)
    print(" TASK 1: TEXT GENERATION (Explanation & Code)")
    print("-" * 55)
    gen_system = "You are an expert AI researcher who explains deep learning concepts concisely."
    gen_user = "Explain why Skip Connections in ResNet solve the vanishing gradient problem."
    print(f"Prompt: '{gen_user}'\n")
    gen_output = client.call_openai(gen_system, gen_user)
    print(gen_output)

    # Task 2: Summarization
    print("\n" + "-" * 55)
    print(" TASK 2: TEXT SUMMARIZATION")
    print("-" * 55)
    research_text = """
    Deep neural networks with hundreds of layers are difficult to train due to vanishing gradients.
    We introduce a residual learning framework where layers fit a residual mapping H(x) = F(x) + x
    rather than unreferenced mappings. Extensive experiments on ImageNet demonstrate that these
    residual networks are easier to optimize and gain accuracy from considerably increased depth.
    """
    sum_system = "You are a professional research paper summarizer. Summarize in 2 crisp sentences."
    sum_user = f"Summarize this abstract:\n{research_text}"
    print(f"Input Text: {research_text.strip()}\n")
    sum_output = client.call_anthropic(sum_system, sum_user)
    print(sum_output)

    # Task 3: Question Answering (Contextual)
    print("\n" + "-" * 55)
    print(" TASK 3: QUESTION ANSWERING")
    print("-" * 55)
    qa_user = "Question: What is the primary difference between an LSTM and a GRU in sequence modeling?"
    print(f"Query: {qa_user}\n")
    qa_output = client.call_ollama(qa_user, model="llama3")
    print(qa_output)

    print("\n" + "=" * 65)
    print(" [INFO] To run live API calls with your own credentials:")
    print("  1. OpenAI:    Set $env:OPENAI_API_KEY='your-key-here'")
    print("  2. Anthropic: Set $env:ANTHROPIC_API_KEY='your-key-here'")
    print("  3. Ollama:    Install from https://ollama.com and run 'ollama run llama3'")
    print("=" * 65)


if __name__ == "__main__":
    run_llm_tasks()
