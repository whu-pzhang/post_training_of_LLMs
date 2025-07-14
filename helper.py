import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


def generate_responses(model,
                       tokenizer,
                       user_message,
                       system_message=None,
                       max_new_tokens=100):
    # Format chat using tokenizer's chat template
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})

    # We assume the data are all single-turn conversation
    messages.append({"role": "user", "content": user_message})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Recommended to use vllm, sglang or TensorRT
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids,
                                skip_special_tokens=True).strip()

    return response


def test_model_with_questions(model,
                              tokenizer,
                              questions,
                              system_message=None,
                              title="Model Output"):
    print(f"\n=== {title} ===")
    for i, question in enumerate(questions, 1):
        response = generate_responses(model, tokenizer, question,
                                      system_message)
        print(
            f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")


def load_model_and_tokenizer(model_name, use_gpu=False):

    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if use_gpu:
        model.to("cuda")

    if not tokenizer.chat_template:
        tokenizer.chat_template = """{% for message in messages %}
                {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
                {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
                {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>
                {% endif %}
                {% endfor %}"""

    # Tokenizer config
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def display_dataset(dataset):
    # Visualize the dataset
    rows = []
    for i in range(3):
        example = dataset[i]
        user_msg = next(m['content'] for m in example['messages']
                        if m['role'] == 'user')
        assistant_msg = next(m['content'] for m in example['messages']
                             if m['role'] == 'assistant')
        rows.append({
            'User Prompt': user_msg,
            'Assistant Response': assistant_msg
        })

    # Display as table
    df = pd.DataFrame(rows)
    pd.set_option('display.max_colwidth',
                  None)  # Avoid truncating long strings
    print(df.to_markdown(index=False, tablefmt="grid"))


if __name__ == "__main__":
    # Example usage
    model_name = "Qwen/Qwen3-0.6B-Base"
    model, tokenizer = load_model_and_tokenizer(model_name=model_name,
                                                use_gpu=True)
    questions = [
        "Give me an 1-sentence introduction of LLM.", "Calculate 1+1-1",
        "What's the difference between thread and process?"
    ]
    test_model_with_questions(model, tokenizer, questions)
