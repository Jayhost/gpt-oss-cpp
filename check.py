import urllib.request
import json

def check_config():
    urls = [
        "https://huggingface.co/TroyDoesAI/gpt-oss-4B/raw/main/config.json",
        "https://huggingface.co/mradermacher/gpt-oss-4B/raw/main/config.json"
    ]
    for url in urls:
        try:
            print(f"Trying {url}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                config = json.loads(response.read().decode('utf-8'))
                print("\n=== Model Config ===")
                print(f"  rope_theta: {config.get('rope_theta', 'NOT FOUND')}")
                print(f"  hidden_act: {config.get('hidden_act', 'NOT FOUND')}")
                print(f"  hidden_size: {config.get('hidden_size')}")
                print(f"  num_experts: {config.get('num_experts', config.get('num_local_experts'))}")
                print(f"  num_heads: {config.get('num_attention_heads')}")
                print(f"  num_kv_heads: {config.get('num_key_value_heads')}")
                return
        except Exception as e:
            print(f"  Failed: {e}")
    print("\nCould not automatically fetch config. Please check HuggingFace manually.")

if __name__ == '__main__':
    check_config()