import sys
import os

# Ensure we can import exo
sys.path.append(os.getcwd())

try:
    from exo import models
    print("✅ Successfully imported exo.models")
except Exception as e:
    print(f"❌ Failed to import exo.models: {e}")
    sys.exit(1)

model_id = "qwen-2.5-32b-instruct"

if model_id in models.model_cards:
    print(f"✅ Found '{model_id}' in model_cards")
    card = models.model_cards[model_id]
    print(f"   Config: {card}")
    
    # Check repo resolution
    engine = "MLXDynamicShardInferenceEngine"
    repo = models.get_repo(model_id, engine)
    print(f"   Resolved Repo for {engine}: {repo}")
    
    if repo == "Qwen/Qwen2.5-32B-Instruct":
        print("✅ Repo ID matches expected value.")
    else:
        print(f"❌ Repo ID mismatch! Expected 'Qwen/Qwen2.5-32B-Instruct', got '{repo}'")

else:
    print(f"❌ '{model_id}' NOT found in model_cards!")
    print("Available keys:", list(models.model_cards.keys()))
