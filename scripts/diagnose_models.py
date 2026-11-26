import os
import requests

def main():
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("❌ FIREWORKS_API_KEY is not set in the environment.")
        print("Please run this script in the terminal where your API key is exported:")
        print("  python3 -m scripts.diagnose_models")
        return

    print("🔑 FIREWORKS_API_KEY found (length: {})".format(len(api_key)))
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 1. Fetch available models
    print("\n📡 Querying Fireworks AI available models...")
    available_model_ids = []
    try:
        response = requests.get(
            "https://api.fireworks.ai/inference/v1/models",
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            models_data = response.json()
            models_list = models_data.get("data", [])
            print("✅ Successfully fetched model list. Found {} models.".format(len(models_list)))
            
            # Print all available models
            print("\n📦 Active Models in your Account:")
            for m in sorted(models_list, key=lambda x: x.get("id", "")):
                m_id = m.get("id")
                available_model_ids.append(m_id)
                print(f"  - {m_id}")
        else:
            print(f"❌ Failed to fetch models. Status: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Exception querying models: {e}")

    # 2. Add Kimi to tests
    test_models = ["accounts/fireworks/models/kimi-k2p5"]
    for m_id in available_model_ids:
        if m_id not in test_models:
            test_models.append(m_id)
            
    # Include generic ones just in case
    for generic in [
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "accounts/fireworks/models/llama-v3p1-8b-instruct"
    ]:
        if generic not in test_models:
            test_models.append(generic)
    
    print("\n⚡ Testing Chat Completion Endpoint for models...")
    for model in test_models[:5]: # test up to 5 models
        print(f"\n--- Testing model: {model} ---")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Say hello!"}],
            "max_tokens": 10
        }
        try:
            res = requests.post(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            if res.status_code == 200:
                print(f"✅ SUCCESS! Response: {res.json()['choices'][0]['message']['content'].strip()}")
                print(f"💡 Recommended configuration: use '{model}'")
                break
            else:
                print(f"❌ FAILED. Status: {res.status_code} - {res.text}")
        except Exception as e:
            print(f"❌ Exception: {e}")

if __name__ == "__main__":
    main()
