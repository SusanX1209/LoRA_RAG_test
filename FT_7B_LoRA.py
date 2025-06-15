
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format

# 130MB
import wandb
print(dir(wandb))

run = wandb.init(
    project='Fine-tune Llama 3 8B on Medical Dataset', 
    job_type="training", 
    anonymous="allow"
)

base_model = "/root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa"

dataset_name = "./dataset/ai-medical-chatbot"
new_model = "llama-3-8b-chat-doctor-simpleLoRA-alldata-3ep-8bs"

torch_dtype = torch.float16
attn_implementation = "eager"

# # QLoRA config
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch_dtype,
#     bnb_4bit_use_double_quant=True,
# )

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    # quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation=attn_implementation
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
model, tokenizer = setup_chat_format(model, tokenizer)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)

#Importing the dataset
dataset = load_dataset("parquet", data_files="./dataset/dialogues.parquet", split="train")

# dataset = dataset.shuffle(seed=65).select(range(1000)) # Only use 1000 samples for quick demo
dataset = dataset.shuffle(seed=65) # Only use 1000 samples for quick demo

def format_chat_template(row):
    row_json = [{"role": "user", "content": row["Patient"]},
               {"role": "assistant", "content": row["Doctor"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc=4,
)

print(dataset['text'][3])

dataset = dataset.train_test_split(test_size=0.1)
training_arguments = TrainingArguments(
    output_dir=new_model,
    # per_device_train_batch_size=8,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    # gradient_accumulation_steps=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    # learning_rate=2e-4,
    learning_rate=2e-5,
    max_grad_norm=1.0,
    fp16=False,
    # fp16=True,
    bf16=False,
    group_by_length=True,
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length=512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

trainer.train()

wandb.finish()
model.config.use_cache = True

messages = [
    {
        "role": "user",
        # "content": "Hello doctor, I have bad acne. How do I get rid of it?"
        "content": "Hello doctor, my shoulders and neck always hurt. How can I get rid of it?"
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
with open('./result/lora_doctor_res.txt','w',encoding='utf-8') as f:
    f.write(text)
print('My test')
print(text.split("assistant")[1])
print('End')
trainer.model.save_pretrained(new_model)

# trainer.model.push_to_hub(new_model, use_temp_dir=False)

