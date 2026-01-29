from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from torch.nn.functional import softmax
import jinja2, json
from vllm import LLM, SamplingParams
import math

model_path_name = "ibm-granite/granite-guardian-3.2-5b"

safe_token = "No"
risky_token = "Yes"
nlogprobs = 20
max_tokens = 2048

tokenizer = AutoTokenizer.from_pretrained(model_path_name)

sampling_params = SamplingParams(temperature=0.0, logprobs=nlogprobs, max_tokens=max_tokens)
model = LLM(model=model_path_name, tensor_parallel_size=1)

def parse_response(response):
    trace_match = re.findall(r'<think>(.*?)</think>', response, re.DOTALL)
    score_match = re.findall(r'<score>(.*?)</score>', response, re.DOTALL)

    score, trace = None, None
    
    if trace_match:
        trace = trace_match[-1]

    if score_match:
        score = score_match[-1]
    
    return score, trace

user_text = "How can I hack my friend's email?"
messages = [{"role": "user", "content": user_text}]

chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

output = model.generate(chat, sampling_params, use_tqdm=False)
response = output[0].outputs[0].text.strip()

score, _ = parse_response(response)

print(f"# score: {score}")