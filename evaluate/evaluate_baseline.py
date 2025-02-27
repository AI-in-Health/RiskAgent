import pandas as pd 
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import os
import requests

def load_local_model(model_path, device_map):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.bfloat16
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def chat_generate(messages, model, tokenizer=None, api_key=None, model_type="llama3", max_new_tokens=4, temperature=0.6, top_p=0.9, do_sample=True):
    if model_type.lower() == "gpt":
        API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": model,  
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens
        }
        response = requests.post(API_ENDPOINT, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
    
    else:  
        if model_type.lower() == "llama2":
            if tokenizer.chat_template is None:
                tokenizer.chat_template = """<s>[INST] {% if messages[0]['role'] == 'system' %}<<SYS>>{{ messages[0]['content'] }}<</SYS>>

{% endif %}{{ messages[1]['content'] }} [/INST]"""
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
        
        return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)


def process_with_model(test_df, model, tokenizer=None, api_key=None, model_type="llama3"):
    results = []
    
    if tokenizer and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = row['question']
        options = [
            row['option_a'],
            row['option_b'],
            row['option_c'],
            row['option_d']
        ]

        messages = [
            {
                "role": "system", 
                "content": "You are a medical expert taking a multiple choice quiz. Always respond with only a single letter (A/B/C/D)."
            },
            {
                "role": "user", 
                "content": f"""Patient Information: {row['patient_info']}

Question: {question}

Options:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

Select your answer (A/B/C/D only)"""
            }
        ]

        response = chat_generate(
            messages=messages,
            model=model,
            tokenizer=tokenizer,
            api_key=api_key,
            model_type=model_type,
            max_new_tokens=4,
            temperature=0.6,
            top_p=0.9,
            do_sample=True
        )

        results.append({
            'input_id': row['input_id'],
            'response': response,
            'correct_answer': row['correct_answer']
        })
        
    return pd.DataFrame(results)

def calculate_accuracy(results_df, output_file):

    results_df['pred'] = results_df['response'].str.extract(r'([A-D])', expand=False)
    correct = (results_df['pred'] == results_df['correct_answer']).sum()
    total = len(results_df)
    accuracy = correct / total

    print(f"Correct predictions: {correct}")
    print(f"Total samples: {total}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    txt_file = output_file.rsplit('.', 1)[0] + '_accuracy.txt'
    
    with open(txt_file, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"=================\n")
        f.write(f"Correct predictions: {correct}\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    print(f"Accuracy results saved to {txt_file}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Run LLaMA evaluation')
    parser.add_argument('--model_path', default='', type=str, help='Path to model')
    parser.add_argument('--split', type=str, choices=['train', 'test', 'val'], default='test', required=True)
    parser.add_argument('--device_map', type=str, default='auto', help='Device to run model on')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data file')
    parser.add_argument('--model_type', type=str, choices=['llama3', 'llama2', 'gpt'], required=True)
    parser.add_argument('--api_key', type=str, help='OpenAI API key (required for GPT models)')
    parser.add_argument('--model_card', type=str, default='gpt-4', help='OpenAI model card name (e.g., gpt-4, gpt-3.5-turbo)')

    args = parser.parse_args()

    if args.model_type == 'gpt' and not args.api_key:
        raise ValueError("API key is required for GPT models")


    model = None
    tokenizer = None
    
    if args.model_type in ['llama3', 'llama2']:
        if not args.model_path:
            raise ValueError("Model path is required for local models")
        print("Loading model...")
        model, tokenizer = load_local_model(args.model_path, args.device_map)
        tokenizer.pad_token = tokenizer.eos_token
        print("Model loaded!")
    else:  
        model = args.model_card  
    
    data = pd.read_excel(args.data_path)
    split_data = data[data['split'] == args.split].sample(frac=1)
    
    results_df = process_with_model(
        split_data, 
        model, 
        tokenizer=tokenizer, 
        api_key=args.api_key, 
        model_type=args.model_type
    )
    final_results = calculate_accuracy(results_df, args.output_file)
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    final_results.to_excel(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()