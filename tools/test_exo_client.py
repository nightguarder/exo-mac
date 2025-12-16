import os
from openai import OpenAI
##
# How to Run
#   1. Start Exo:
#   bash start_brain.sh
#   2. Run Test:
#   .venv/bin/python test_exo_client.py

# Configuration
# Default to localhost for Exo, but allow env var override
EXO_API_URL = os.getenv("EXO_URL", "http://localhost:52415/v1")
DEFAULT_MODEL = "qwen-2.5-32b-instruct" # Adjust based on what your Exo instance serves

class ExoClient:
    def __init__(self):
        # We use the OpenAI standard client, but point it to local Exo
        self.client = OpenAI(
            base_url=EXO_API_URL,
            api_key="exo" # Dummy key
        )
        print(f"🔗 Connected to Exo Brain at {EXO_API_URL}")

    def generate_text(self, system_prompt, user_prompt, temperature=0.7):
        """Generic text generation wrapper"""
        try:
            response = self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=2000 
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Exo Error: {e}")
            return None

if __name__ == "__main__":
    client = ExoClient()
    system = "You are a helpful assistant."
    user = "What is the capital of France?"
    print(f"User: {user}")
    response = client.generate_text(system, user)
    print(f"Assistant: {response}")
