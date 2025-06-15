# from huggingface_hub import login
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()

# hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN")
# login(token = hf_token)


base_model = "/root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa"
new_model = "llama-3-8b-chat-doctor-simpleLoRA"

# # base_model = "/kaggle/input/llama-3/transformers/8b-chat-hf/1"
# new_model = "/kaggle/input/fine-tune-llama-3-8b-on-medical-dataset/llama-3-8b-chat-doctor/"

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
from trl import setup_chat_format
# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model)

base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
)

base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)

# Merge adapter with base model
model = PeftModel.from_pretrained(base_model_reload, new_model)

model = model.merge_and_unload()

# Model Inference
messages = [{"role": "user", "content": "Hello doctor, my shoulders and neck always hurt. How can I get rid of it?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=150)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

with open('./result/lora_doctor_res_merge.txt','w',encoding='utf-8') as f:
    f.write(str(text))
print('My test')

print(text)

# Saving and pushing the merged model
model.save_pretrained("llama-3-8b-chat-doctor_merger-simpleLoRA")
tokenizer.save_pretrained("llama-3-8b-chat-doctor_merger-simpleLoRA")

# model.push_to_hub("llama-3-8b-chat-doctor", use_temp_dir=False)
# tokenizer.push_to_hub("llama-3-8b-chat-doctor", use_temp_dir=False)


