import os
import json
import argparse
import re
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Try to import BitsAndBytesConfig, but make it optional
try:
    from transformers import BitsAndBytesConfig
    HAS_QUANTIZATION = True
except ImportError:
    print("⚠️  BitsAndBytesConfig not available. Install with: pip install bitsandbytes")
    HAS_QUANTIZATION = False

# Extended model registry with more free models
MODEL_REGISTRY = {
    # Reliable open models (no auth required)
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
    "openchat": "openchat/openchat-3.5-0106",
    "mistral-openorca": "Open-Orca/Mistral-7B-OpenOrca",
    
    # Smaller models for faster inference
    "phi-2": "microsoft/phi-2",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "stablelm-zephyr": "stabilityai/stablelm-zephyr-3b",
    
    # Alternative reliable models
    "nous-hermes": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "neural-chat": "Intel/neural-chat-7b-v3-1",
    
    # Code models (good at following instructions)
    "codellama-7b": "codellama/CodeLlama-7b-Instruct-hf",
    
    # Llama models (may require auth)
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct", 
    "llama-2-7b": "meta-llama/Llama-2-7b-chat-hf",
}

def get_chat_template(model_name: str, prompt: str) -> str:
    """Format prompt according to model's chat template"""
    if "llama" in model_name.lower():
        return f"<s>[INST] {prompt} [/INST]"
    elif "mistral" in model_name.lower():
        return f"<s>[INST] {prompt} [/INST]"
    elif "zephyr" in model_name.lower():
        return f"<|system|>\nYou are a helpful AI assistant that generates natural language descriptions.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    elif "openchat" in model_name.lower():
        return f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
    elif "phi" in model_name.lower():
        return f"Instruct: {prompt}\nOutput:"
    else:
        return prompt

def get_improved_prompt_template(n: int, training_mode: bool, action: str, example_caption: str = None) -> str:
    """Enhanced prompt templates with better instruction clarity"""
    
    if training_mode:
        base_prompt = f"""Generate exactly {n} different ways to describe the human action "{action}".

Requirements:
- Use simple, clear language
- Each description should be one short sentence
- Focus on what the person is doing
- Use different sentence structures and vocabulary
- Start each with "A person" or "Someone"
- Keep the same core meaning as the action

Examples of good variations:
Action: "drink water" → "A person drinks water.", "Someone sips water.", "A person consumes liquid."

Action: {action}
Generate {n} variations:
1."""
        
    else:
        base_prompt = f"""Describe the human motion and movement for the action "{action}" in {n} different ways.

Focus on:
- The physical movements involved
- Different contexts or settings
- Various speeds or intensities
- Different subjects (person, individual, someone)
- Keep descriptions concise and clear

Action: {action}
Descriptions:
1."""
    
    return base_prompt

def load_tokenizer_and_model(alias: str, use_4bit: bool = True, use_8bit: bool = False):
    """Load model with optimizations for memory efficiency"""
    if alias not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model alias: '{alias}'. Valid options: {list(MODEL_REGISTRY.keys())}")

    model_id = MODEL_REGISTRY[alias]
    print(f"🔁 Loading model: {alias} → {model_id}")
    
    # Configure quantization for memory efficiency
    quant_config = None
    if HAS_QUANTIZATION:
        try:
            if use_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            elif use_8bit:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
        except Exception as e:
            print(f"⚠️  Quantization failed: {e}")
            quant_config = None
            use_4bit = False
            use_8bit = False
    else:
        print("📝 Running without quantization (install bitsandbytes for memory optimization)")
        use_4bit = False
        use_8bit = False
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True,
            use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.chat_template is None and "chat" in model_id.lower():
            print("⚠️  Model doesn't have chat template, using basic formatting")
            
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("💡 Try a different model or check your internet connection")
        return None, None

    # Load model
    try:
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True,
        }
        
        # Only add quantization if available
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
            
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model.eval()
        
        memory_info = ""
        if torch.cuda.is_available():
            memory_info = f" Memory usage: {torch.cuda.memory_allocated() / 1e9:.2f}GB"
        print(f"✅ Model loaded successfully.{memory_info}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("💡 Try using a smaller model or add --no-4bit flag")
        return tokenizer, None

    return tokenizer, model

def clean_and_parse_captions(raw_output: str, expected_count: int) -> List[str]:
    """Enhanced caption parsing with better cleaning"""
    # Split by common delimiters
    lines = re.split(r'\n|(?:\d+\.)', raw_output)
    
    captions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove numbering patterns
        line = re.sub(r'^\d+[\.\)\-\:]\s*', '', line)
        line = re.sub(r'^[\-\*\•]\s*', '', line)
        
        # Clean up quotes and extra punctuation
        line = line.strip('\'".,;:-')
        
        # Ensure proper sentence structure
        if line and len(line.split()) >= 3:
            # Capitalize first letter
            line = line[0].upper() + line[1:] if len(line) > 1 else line.upper()
            
            # Ensure it ends with period
            if not line.endswith('.'):
                line += '.'
            
            # Basic quality filters
            if (len(line) > 10 and len(line) < 150 and
                any(subject in line.lower() for subject in ['person', 'someone', 'individual', 'human']) and
                not line.lower().startswith('generate') and
                not line.lower().startswith('action:')):
                captions.append(line)
    
    # If we don't have enough captions, add simple fallbacks
    if len(captions) < expected_count:
        base_captions = [
            "A person performs the action.",
            "Someone executes the movement.",
            "An individual demonstrates the motion.",
            "A person carries out the activity.",
            "Someone completes the task."
        ]
        
        for base in base_captions:
            if len(captions) < expected_count:
                captions.append(base)
    
    return captions[:expected_count]

def generate_captions_with_retry(tokenizer, model, prompt: str, n_captions: int, 
                                max_new_tokens: int = 200, max_retries: int = 3) -> List[str]:
    """Generate captions with retry logic for failed generations"""
    
    for attempt in range(max_retries):
        try:
            # Tokenize input
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=512,
                truncation=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate with different temperature for each retry
            temperature = 0.7 + (attempt * 0.1)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    num_return_sequences=1
                )
            
            # Decode and extract generated text
            full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            generated_text = full_output[len(prompt):].strip()
            
            # Parse captions
            captions = clean_and_parse_captions(generated_text, n_captions)
            
            if len(captions) >= max(1, n_captions // 2):  # Accept if we got at least half
                return captions
                
        except Exception as e:
            print(f"Generation attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return [f"A person performs the action."] * n_captions
    
    return [f"A person performs the action."] * n_captions

def process_dataset_batch(tokenizer, model, data_entries: List[Dict], n_captions: int, 
                         training_mode: bool, batch_size: int = 4) -> List[List[str]]:
    """Process dataset in batches for better efficiency"""
    
    all_results = []
    model_name = model.config.name_or_path
    
    # Process in batches
    for i in tqdm(range(0, len(data_entries), batch_size), desc="Processing batches"):
        batch = data_entries[i:i + batch_size]
        batch_results = []
        
        for entry in batch:
            action = entry["action"]
            example_caption = entry.get("captions", [None])[0]
            
            # Create prompt
            prompt = get_improved_prompt_template(n_captions, training_mode, action, example_caption)
            
            # Apply chat template if available
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                try:
                    formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
                except:
                    formatted_prompt = get_chat_template(model_name, prompt)
            else:
                formatted_prompt = get_chat_template(model_name, prompt)
            
            # Generate captions
            captions = generate_captions_with_retry(tokenizer, model, formatted_prompt, n_captions)
            batch_results.append(captions)
        
        all_results.extend(batch_results)
        
        # Clear cache periodically
        if torch.cuda.is_available() and i % (batch_size * 4) == 0:
            torch.cuda.empty_cache()
    
    return all_results

def create_ntu_data_structure(ntu_labels: Dict[int, str]) -> Dict[str, Dict]:
    """Create compatible data structure from NTU labels"""
    data = {}
    for action_id, action_label in ntu_labels.items():
        data[str(action_id)] = {
            "action": action_label,
            "captions": [f"A person {action_label}."]  # Simple initial caption
        }
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate caption variations using HuggingFace models")
    parser.add_argument("--json_file", type=str, help="Path to input JSON file")
    parser.add_argument("--ntu_mode", action="store_true", help="Create NTU dataset from scratch")
    parser.add_argument("--model", default="zephyr-7b", choices=list(MODEL_REGISTRY.keys()),
                        help="Model to use for generation")
    parser.add_argument("--training", action="store_true", help="Use training-style prompts")
    parser.add_argument("-n", "--num-captions", type=int, default=10, help="Number of captions per action")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for processing")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    # Load model
    print(f"🚀 Loading {args.model}...")
    tokenizer, model = load_tokenizer_and_model(args.model, use_4bit=not args.no_4bit)
    
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Load or create data
    if args.ntu_mode:
        print("📊 Creating NTU dataset structure...")
        # NTU RGB+D action labels (first 10 for testing)
        ntu_labels = {
            0: "drink water", 1: "eat meal", 2: "brush teeth", 3: "brush hair", 4: "drop",
            5: "pick up", 6: "throw", 7: "sit down", 8: "stand up", 9: "clapping",
            10: "reading", 11: "writing", 12: "tear up paper", 13: "put on jacket", 14: "take off jacket"
        }
        data = create_ntu_data_structure(ntu_labels)
    else:
        if not args.json_file or not os.path.exists(args.json_file):
            print("❌ JSON file not found. Use --ntu_mode to create NTU dataset or provide valid --json_file")
            return
        
        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    
    print(f"📝 Processing {len(data)} actions with {args.model}...")
    
    # Convert to list format for processing
    data_entries = []
    for key, value in data.items():
        entry = {"id": key, "action": value["action"]}
        if "captions" in value:
            entry["captions"] = value["captions"]
        data_entries.append(entry)
    
    # Generate captions
    all_captions = process_dataset_batch(
        tokenizer, model, data_entries, args.num_captions, 
        args.training, args.batch_size
    )
    
    # Update data with new captions
    for i, (key, new_captions) in enumerate(zip(data.keys(), all_captions)):
        if "captions" not in data[key]:
            data[key]["captions"] = []
        data[key]["captions"].extend(new_captions)
        # Remove duplicates while preserving order
        data[key]["captions"] = list(dict.fromkeys(data[key]["captions"]))
    
    # Save results
    mode_suffix = "training" if args.training else "sampling"
    if args.output:
        output_path = args.output
    elif args.json_file:
        output_path = args.json_file.replace(".json", f"_{args.model}_{mode_suffix}_augmented.json")
    else:
        output_path = f"ntu_captions_{args.model}_{mode_suffix}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(data)} actions with augmented captions to {output_path}")
    
    # Print sample results
    print("\n📋 Sample results:")
    for i, (key, value) in enumerate(list(data.items())[:3]):
        print(f"\nAction {key}: {value['action']}")
        for j, caption in enumerate(value['captions'][:5], 1):
            print(f"  {j}. {caption}")

if __name__ == "__main__":
    main()

# Example usage:
# python refined_caption_generator.py --ntu_mode --model zephyr-7b -n 10
# python refined_caption_generator.py --json_file data.json --model mistral-7b --training -n 15