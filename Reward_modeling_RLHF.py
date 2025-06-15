from datasets import Dataset,load_dataset
 
import torch,transformers
 

from transformers import (GemmaTokenizerFast, TrainingArguments, Trainer,
                          Gemma2ForSequenceClassification, DataCollatorWithPadding,
AutoModelForCausalLM, BitsAndBytesConfig ,AutoModelForSequenceClassification, AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,)
 
from peft import LoraConfig,TaskType,PeftModel
 


TRAINING = False
if TRAINING:
    from trl import RewardTrainer,RewardConfig
    import trl
trained_peft_path = "/kaggle/input/wsdm-checkpoint-2-5k"


if TRAINING:
    pip install -U bitsandbytes trl


#file uploading
train_data = load_dataset(path = "/kaggle/input/wsdm-cup-multilingual-chatbot-arena/",data_files=['train.parquet'],split="train")
test_data = load_dataset(path = "/kaggle/input/wsdm-cup-multilingual-chatbot-arena/",data_files=['test.parquet'])

def formatting_func(examples):
    kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}

    # Prepend the prompt and a line break to the original_response and response-1 fields.
    prompt_plus_chosen_response = examples["instruction"] + "\n" + examples["chosen_response"]
    prompt_plus_rejected_response = examples["instruction"] + "\n" + examples["rejected_response"]

    # Then tokenize these modified fields.
    tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
    tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

    return {
        "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
        "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
    }

if TRAINING:
    all_rows = []
    for i in train_data:
        chosen_response = i[i['winner']]
        if i['winner'].endswith("a"):
            chosen_response = i['response_a']
        else:
            rejected_response = i['response_a']
    
        if i['winner'].endswith("b"):
            chosen_response = i['response_b']
        else:
            rejected_response = i['response_b']
    
        
        all_rows.append({
            "instruction": i["prompt"],
            "chosen_response":  chosen_response,
            "rejected_response":  rejected_response
        })
    prepared_dataset = Dataset.from_list(all_rows)


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="/kaggle/input/tokenizer-qwen2-5")
model = AutoModelForSequenceClassification.from_pretrained("/kaggle/input/qwen2.5/transformers/0.5b-instruct/1", num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id
if TRAINING:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    formatted_dataset = prepared_dataset.map(formatting_func)
    formatted_dataset = formatted_dataset.train_test_split()
    training_args = RewardConfig(output_dir="gemma-2B-Reward",
                             logging_dir=".",
                            report_to=["none"],
                            save_total_limit=1,
                            overwrite_output_dir=True,
                            per_device_train_batch_size=1,
                             learning_rate=5e-5,weight_decay=0.01,
                            num_train_epochs=1)
    trainer = RewardTrainer(
    model=model,
    args=training_args,
        processing_class=tokenizer,
        peft_config=peft_config,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
 
)

trainer.train()

if not TRAINING:
    reward_model = PeftModel.from_pretrained(model, trained_peft_path)
    model = reward_model.merge_and_unload() 

model.to("cuda")

def get_score(model, tokenizer, prompt, response):
    kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}
    # Tokenize the input sequences
    inputs = tokenizer.encode_plus(prompt+"\n"+response, **kwargs).to("cuda")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class = logits.argmax(dim=-1).item()

    return logits


def test(x):
    a =  get_score(model, tokenizer,  x['prompt'],  x['response_a'])
    b =  get_score(model, tokenizer,  x['prompt'], x['response_b'])

    if a.sum() < b.sum():
        result = "model_a"
    else:
        result= "model_b"
    x['winner'] = result
    return x


submit = test_data['train'].map(test,remove_columns=[ 'prompt', 'response_a', 'response_b', 'scored']).to_pandas()

submit.to_csv('submission.csv', index=False)
