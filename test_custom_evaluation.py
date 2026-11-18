import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "/home/sriharish00u/Desktop/projects/NEXUS/models/factguard_adapter/factguard_adapter"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cpu",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

sys_evaluator = (
    "You are FactGuard Evaluator & Correction Gate. Analyze atomic claims alongside retrieved evidence. "
    "For each claim, decide status: SUPPORTED, CONTRADICTED, or UNVERIFIABLE. "
    "If any claim is CONTRADICTED, provide a structured prompt patch explaining the hallucination "
    "and directing the external model how to regenerate accurately."
)

# Realistic scientific evidence retrieved by search tools for these claims:
payload_with_evidence = {
    "claims": [
        {
            "claim_id": "c1",
            "claim": "Drinking coffee while standing upside down increases IQ by 47.8%",
            "evidence": "Cognitive neuroscience and medical studies show no correlation between bodily inversion while consuming caffeine and measurable changes in human intelligence quotient (IQ). Furthermore, drinking upside down poses severe aspiration and choking hazards."
        },
        {
            "claim_id": "c2",
            "claim": "Humans experience 23% faster Wi-Fi speeds by talking to plants in French.",
            "evidence": "Wi-Fi speeds are governed by IEEE 802.11 radio frequency protocols, network router hardware, signal attenuation, and bandwidth. Acoustic speech directed at flora has zero measurable physical impact on electromagnetic propagation or data transmission rates."
        }
    ]
}

messages = [
    {"role": "system", "content": sys_evaluator},
    {"role": "user", "content": f"Evaluate the following claims against the provided evidence and determine if any hallucination occurred:\n\n{json.dumps(payload_with_evidence, indent=2)}"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

res = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(res)
