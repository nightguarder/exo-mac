import asyncio
import aiohttp
import json
import sys

BASE_URL = "http://localhost:52415"

async def test_completion(model: str):
    print(f"Testing completion for model: {model}")
    url = f"{BASE_URL}/v1/completions"
    payload = {
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "temperature": 0
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    print(f"Error: {response.status}")
                    print(await response.text())
                    return
                
                data = await response.json()
                print("Response:", json.dumps(data, indent=2))
                assert "choices" in data
                assert len(data["choices"]) > 0
                assert "text" in data["choices"][0]
                print("Legacy completion test PASSED")
        except Exception as e:
            print(f"Failed to connect: {e}")

async def test_completion_stream(model: str):
    print(f"Testing STREAMING completion for model: {model}")
    url = f"{BASE_URL}/v1/completions"
    payload = {
        "model": model,
        "prompt": "Count to 5: 1, 2,",
        "max_tokens": 10,
        "temperature": 0,
        "stream": True
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    print(f"Error: {response.status}")
                    print(await response.text())
                    return
                
                print("Stream started...")
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line: continue
                    if line == "data: [DONE]":
                        print("\nStream DONE")
                        break
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        sys.stdout.write(data["choices"][0]["text"])
                        sys.stdout.flush()
                print("\nStreaming completion test PASSED")
        except Exception as e:
            print(f"Failed to connect: {e}")

async def test_parameters(model: str):
    print(f"Testing PARAMETERS (stop, temp, max_tokens) for model: {model}")
    url = f"{BASE_URL}/v1/completions"
    
    # Test 1: Stop sequence
    # Qwen3-0.6B-Instruct is a thinking model that starts with <think>.
    # If we set stop=["<think>"], it should return empty or very short text.
    payload = {
        "model": model,
        "prompt": "Why is the sky blue?",
        "max_tokens": 20,
        "temperature": 0,
        "stop": ["<think>"]
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                text = data["choices"][0]["text"]
                print(f"Stop test output: '{text}'")
                if "<think>" in text:
                    print(f"Stop test FAILED: Found '<think>' in output '{text}'")
                elif len(text) > 5: # Allow some whitespace or partial
                    print(f"Stop test WARNING: Output '{text}' is longer than expected for immediate stop")
                else:
                    print("Stop test PASSED")
        except Exception as e:
            print(f"Stop test failed with error: {e}")



    # Test 2: Max tokens
    payload = {
        "model": model,
        "prompt": "Write a very long story about a cat.",
        "max_tokens": 5,
        "temperature": 0
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                usage = data["usage"]["completion_tokens"]
                text = data["choices"][0]["text"]
                print(f"Max tokens usage: {usage}, text len: {len(text)}")
                # Allow small tolerance
                if usage > 6: 
                     print(f"Max tokens test FAILED: {usage} > 5")
                else:
                     print("Max tokens test PASSED")
        except Exception as e:
            print(f"Max tokens test failed with error: {e}")

    # Test 3: Temperature consistency
    print("Testing Temperature Consistency...")
    payload["max_tokens"] = 10
    payload["temperature"] = 0
    
    res1 = ""
    res2 = ""
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                res1 = data["choices"][0]["text"]
            
            async with session.post(url, json=payload) as response:
                data = await response.json()
                res2 = data["choices"][0]["text"]
                
            if res1 == res2:
                print("Temperature=0 consistency test PASSED")
            else:
                print(f"Temperature=0 consistency test FAILED: '{res1}' != '{res2}'")
        except Exception as e:
             print(f"Temp test failed: {e}")

async def test_list_prompt(model: str):
    print(f"Testing List Prompt for model: {model}")
    url = f"{BASE_URL}/v1/completions"
    payload = {
        "model": model,
        "prompt": ["Say A", "Say B"],
        "max_tokens": 5,
        "temperature": 0
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"List prompt response: {json.dumps(data, indent=2)}")
                    print("List prompt test PASSED (Server accepted list)")
                else:
                    print(f"List prompt test FAILED: {response.status}")
        except Exception as e:
            print(f"List prompt failed: {e}")

async def place_instance(model: str):
    print(f"Placing instance for model: {model}")
    url = f"{BASE_URL}/place_instance"
    # Using specific sharding and instance_meta from discovered enums
    payload = {
        "model_id": model,
        "sharding": "Pipeline",
        "instance_meta": "MlxRing",
        "min_nodes": 1
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    print(f"Failed to place instance: {response.status}")
                    print(await response.text())
                    return False
                print("Instance placement requested successfully.")
                data = await response.json()
                print("Placement response:", json.dumps(data, indent=2))
                return True
        except Exception as e:
            print(f"Failed to connect to place instance: {e}")
            return False

async def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/test_completion.py <model_name>")
        # Auto-detect mode
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{BASE_URL}/v1/models") as resp:
                    if resp.status == 200:
                        models_data = await resp.json()
                        models = models_data.get('data', [])
                        if models:
                            # Pick the smallest model by storage size
                            models.sort(key=lambda x: x.get('storage_size_megabytes', 9999999))
                            target_model = models[0]
                            model_id = target_model['id']
                            print(f"Auto-detected smallest model: {model_id} ({target_model.get('storage_size_megabytes')} MB)")
                            
                            # Place instance
                            if await place_instance(model_id):
                                print("Waiting 10s for instance to form...")
                                await asyncio.sleep(10)
                                await test_completion(model_id)
                                await test_completion_stream(model_id)
                                await test_parameters(model_id)
                            else:
                                print("Skipping tests due to placement failure.")
                        else:
                            print("No models found running.")
                    else:
                        print(f"Failed to fetch models: {resp.status}")
            except Exception as e:
                print(f"Could not connect to Exo to list models: {e}")
        return

    model = sys.argv[1]
    # If user provides model, we assume they want to use it directly, or we should place it too?
    # Let's try to place it just in case.
    await place_instance(model)
    await asyncio.sleep(5)
    await test_completion(model)
    await test_completion_stream(model)

if __name__ == "__main__":
    asyncio.run(main())
