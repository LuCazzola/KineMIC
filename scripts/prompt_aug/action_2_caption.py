import os
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Friendly model alias mapping
MODEL_REGISTRY = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "falcon-7b": "tiiuae/falcon-7b-instruct"
}

def get_prompt_template(n, training_mode, action, example_caption):
    if training_mode:
        return (
            f"Generate {n} short and diverse natural language descriptions that express the same general human action as the given label.\n"
            f"Example: a valid caption for the action '{action}' is: '{example_caption}'\n"
            "Guidelines:\n"
            "1. Each caption must be simple and general, like the example.\n"
            "2. Avoid naming objects, tools, or body parts unless absolutely necessary.\n"
            "3. Focus only on rewording the core action, not describing how it's done.\n"
            "4. Use short, standalone sentences.\n"
            "5. Stick to generic subjects like 'a person' or 'someone'.\n\n"
            f"Action: {action}\n"
            "Captions:"
        )
    else:
        return (
            f"Generate {n} diverse and concise natural language descriptions of human motion based on the given action.\n"
            "The captions must follow these priorities:\n"
            "1. Focus explicitly on the *human motion* involved in the action.\n"
            "2. Use varied wording, phrasing, and context e.g., different subjects ('a person', 'a man', 'someone'), speeds ('quickly', 'slowly'), postures ('while seated', 'standing'), or environments.\n"
            "3. Each caption should be a short, standalone sentence.\n"
            f"Action: {action}\n"
            "Captions:"
        )


def load_tokenizer_and_model(alias):
    if alias not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model alias: '{alias}'. Valid options: {list(MODEL_REGISTRY)}")

    model_id = MODEL_REGISTRY[alias]
    print(f"🔁 Loading model: {alias} → {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=True,
        device_map="auto",
        torch_dtype="auto"
    )
    model.eval()
    return tokenizer, model


def generate_many_captions(tokenizer, model, data_entries, n_captions, training_mode, batch_size=8, max_new_tokens=150, max_input_length=512):
    prompts = []
    for entry in data_entries:
        action = entry["action_label"]
        example_caption = entry["captions"][0] if entry["captions"] else f"A person {action}."
        prompt = get_prompt_template(n_captions, training_mode, action, example_caption)
        prompts.append(prompt)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    outputs = []
    pbar = tqdm(range(0, len(prompts), batch_size), desc="Generating", unit="batch")
    for i in pbar:
        batch_input_ids = input_ids[i:i+batch_size]
        batch_attention_mask = attention_mask[i:i+batch_size]
        generated_ids = model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        outputs.extend(decoded)

    results = []
    for output in outputs:
        generated = output.split("Captions:")[-1].strip().split("\n")
        captions = [cap.strip("1234567890. -") for cap in generated if cap.strip()]
        results.append(captions)

    return results


def main(json_path, model_alias, training_mode, n_captions):
    tokenizer, model = load_tokenizer_and_model(model_alias)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🚀 Generating {n_captions} captions per action for {len(data)} actions...")
    all_captions = generate_many_captions(tokenizer, model, list(data.values()), n_captions, training_mode)

    for key, (entry, new_captions) in zip(data.keys(), zip(data.values(), all_captions)):
        entry["captions"].extend(new_captions)

    mode_suffix = "training" if training_mode else "sampling"
    out_path = json_path.replace(".json", f"_{mode_suffix}_augmented.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved augmented data to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, default="NTU60" help="Path to the input JSON file")
    parser.add_argument("--model", default="mistral-7b",
                        help="Model alias: mistral-7b, mixtral-8x7b, falcon-7b, llama-2")
    parser.add_argument("--training", action="store_true", help="Use training-style prompts instead of sampling-style")
    parser.add_argument("-n", "--num-captions", type=int, default=3, help="Number of captions to generate per action")
    args = parser.parse_args()

    json_file = os.path.join(args.dataset, 'action_captions.json')

    main(args.json_file, args.model, args.training, args.num_captions)
