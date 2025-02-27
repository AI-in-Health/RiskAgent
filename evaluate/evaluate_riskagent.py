from openai import OpenAI, AzureOpenAI
import time
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import codecs
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os
import ast
from uuid import uuid4
import requests
from typing import Dict
import argparse
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry
import urllib3
import logging



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

def load_data(file_path):
    return pd.read_excel(file_path)

def save_json_array(lst, file_path):
    with open(file_path, 'w', encoding='utf-8') as wf:
        json.dump(lst, wf, ensure_ascii=False, indent=4)

def extract_reflection_hint(reflection_output):

    if "ANALYSIS:" in reflection_output:
        analysis = reflection_output.split("ANALYSIS:")[1].strip()
    elif "Analysis:" in reflection_output:
        analysis = reflection_output.split("Analysis:")[1].strip()
    else:
        return reflection_output.strip()
        
    analysis = analysis.replace("Stage_", "Stage ") 
    analysis = analysis.replace("[ERROR]", "error:")  
    
    if "error:" in analysis.lower():
        return analysis.strip()
        
    if "could be improved" in analysis.lower() or "should be" in analysis.lower():
        return analysis.strip()
        
    if "all stages processed correctly" in analysis.lower():
        return ""
        
    return analysis.strip()


def analyze_reflection(reflection_text):
        reflection_text = reflection_text.lower()
        
        if 'stage_1' in reflection_text or 'tool selection' in reflection_text:
            return 'env1'
        elif 'stage_2' in reflection_text or 'parameter' in reflection_text:
            return 'env2'
        elif 'stage_3' in reflection_text or 'interpretation' in reflection_text:
            return 'env3'
        elif 'stage_4' in reflection_text or 'answer' in reflection_text:
            return 'env4'
        return 'all'

def calculate_from_url(calc_id, params: Dict) -> Dict:
    try:
        request_body = {
            "UOMSYSTEM": False,
            "webLanguage": "english",
            **params,
            "meta": {
                "event_id": str(uuid4()),
                "user_id": str(uuid4()),
                "source": "web",
                "units": "US",
                "inputs_unit_toggled": []
            }
        }
        
        api_url = f'https://www.mdcalc.com/api/v1/calc/{calc_id}/calculate'
        headers = {
            'User-Agent': 'Mozilla/5.0',
            "X-Requested-With": "XMLHttpRequest",
        }
        
        logging.info(f"Sending API request to: {api_url}")

        response = requests.post(api_url, json=request_body, headers=headers, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Calculation Error: {response.status_code} - {response.text}")
        return response.json()
    except requests.Timeout:
        logging.error("API request timeout")
        return {'error': 'API request timeout'}
    except Exception as e:
        logging.error(f"API request error: {str(e)}")
        return {'error': str(e)}


def load_local_model(model_path, device_map):
    """
    Load a local model and tokenizer
    Args:
        model_path: Path to model or model name on HuggingFace
        device_map: Device configuration for model loading
    Returns:
        tuple: (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.bfloat16
    )
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
            
    return model, tokenizer

class MedicalQAPipeline:
    def __init__(self, model_type="azure", deployment=None, api_key=None, 
                 api_base=None, api_version=None, wait_time=0.1,
                 model_path=None, device_map='auto', output_dir="outputs",
                 tool_lib_path="final_release/data/tool_library.xlsx"):
        """
        Initialize the pipeline
        Args:
            model_type: Type of model to use ("azure", "openai", "local", or "huggingface")
            deployment: Model deployment name or card
            api_key: API key for cloud services
            api_base: API base URL for Azure
            api_version: API version for Azure
            wait_time: Wait time between requests
            model_path: Path to local model or HuggingFace model name
            device_map: Device configuration for local models
            output_dir: Directory to save outputs
            tool_lib_path: Path to tool library excel file
        """
        self.model_type = model_type
        self.deployment = deployment
        self.wait_time = wait_time
        self.results = {}
        self.stats = {}
        self.checkpoint_path = None
        self.output_dir = output_dir
        
        # Load tool library
        try:
            self.tool_lib = pd.read_excel(tool_lib_path)
            logging.info(f"Successfully loaded tool library from {tool_lib_path}")
        except Exception as e:
            logging.error(f"Error loading tool library: {str(e)}")
            raise
        
        if model_type in ["azure", "openai"]:
            if model_type == "azure":
                self.client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=api_base
                )
            else:
                self.client = OpenAI(api_key=api_key)
        else:  # local or huggingface model
            if not model_path:
                raise ValueError("model_path is required for local or huggingface models")
            self.model, self.tokenizer = load_local_model(model_path, device_map)


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    @sleep_and_retry
    @limits(calls=6000, period=60)
    def chat_generate(self, messages, temperature=0.6, max_tokens=256):
        """
        Generate chat response using specified model
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
        Returns:
            str: Generated response
        """
        try:
            if self.model_type in ["azure", "openai"]:
                if self.model_type == "azure":
                    time.sleep(self.wait_time)
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=60 if self.model_type == "azure" else 30
                )
                return response.choices[0].message.content
            
            else:  # local or huggingface model
                try:
                    # Prepare input using chat template
                    if self.model_type == "llama2" and self.tokenizer.chat_template is None:
                        self.tokenizer.chat_template = """<s>[INST] {% if messages[0]['role'] == 'system' %}<<SYS>>{{ messages[0]['content'] }}<</SYS>>

{% endif %}{{ messages[1]['content'] }} [/INST]"""
                    
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.model.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs,
                            max_new_tokens=max_tokens,
                            pad_token_id=self.tokenizer.pad_token_id,
                            do_sample=True,
                            temperature=temperature,
                            top_p=0.9,
                        )
                    
                    response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                    return response.strip()
                    
                except Exception as e:
                    logging.error(f"Local model error: {str(e)}")
                    if "CUDA" in str(e):
                        torch.cuda.empty_cache()  # Clear CUDA cache on error
                    raise e

        except Exception as e:
            logging.error(f"API error ({self.model_type}): {str(e)}")
            if "timeout" in str(e).lower():
                logging.warning("Request timeout, retrying...")
            if "rate" in str(e).lower():
                logging.warning("Rate limit reached, waiting before retry...")
                time.sleep(2)
            if "content_filter" in str(e).lower():
                logging.warning(f"Content filter error, returning FILTERED")
                return "FILTERED"
            raise e

    def get_question(self, row):
        stem = row['patient_info']
        question = row['question']
        options = [
            row['option_a'],
            row['option_b'],
            row['option_c'],
            row['option_d']
        ]

        options_str = " ".join([f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
        return f"{stem} \n{question} \n{options_str}"

    def get_question_options(self, row):
        question = row['question']
        options = [
            row['option_a'],
            row['option_b'],
            row['option_c'],
            row['option_d']
        ]
        answer = row['correct_answer']
    
        return question, options, answer

    def run_env1(self, row, data_meta, k = 5, reflection_hint = ''):

        def get_tool_name(tool_id, tool_info):
            return tool_info[tool_info['cal_id'] == tool_id]['name'].values[0]

        def get_tool_description(tool_id, tool_info):
            return tool_info[tool_info['cal_id'] == tool_id]['description'].values[0]


        try:
            tool_info = data_meta[['cal_id', 'name', 'description', 'input_schema']]
            
            top_k = row['relevant_tools'].replace('[', '').replace(']', '').replace(' ', '').split(',')[:k]
            
            avail_tools = [
                f"Tool_{t}. {get_tool_name(int(t), tool_info)}: {get_tool_description(int(t), tool_info)}"
                for t in top_k
            ]

            avail_tools_str = '\n'.join(avail_tools)

        
            messages = [
                {
                    "role": "system",
                    "content": "You are a medical professional expert in selecting appropriate clinical assessment tools."
                },
                {
                    "role": "user",
                    "content": f"""Select the most appropriate assessment tool for the following case and question:
{self.get_question(row)}
Available Tools:
{avail_tools_str}
Please respond only one tool that fits the question best in this format:
Tool_xx. [tool name]
Analysis: Brief justification in 2-3 sentences.  
"""
                }
            ]

            if reflection_hint:
                messages = [
                {
                    "role": "system",
                    "content": "You are a medical professional expert in selecting appropriate clinical assessment tools."
                }, {
        "role": "system",
        "content": f"Previous attempt feedback: {reflection_hint}"
    }, {
                    "role": "user",
                    "content": f"""Select the most appropriate assessment tool for the following case and question:
{self.get_question(row)}
Available Tools:
{avail_tools_str}
Please respond only one tool that fits the question best in this format:
Tool_xx. [tool name]
Analysis: Brief justification in 2-3 sentences.  
"""
                }
            ]
            response = self.chat_generate(messages, max_tokens=256)

            tool_id = self.extract_tool_id(response)

            if not tool_id:
                return {
                    'status': 'error',
                    'error_type': 'invalid_tool_id',
                    'selected_tool': None,
                    'full_response': response,
                    'is_correct': False,
                    'full_input1': messages,
                    'full_output1': response,
                    'message': 'Failed to extract tool ID from response'
                }

            if int(tool_id) not in data_meta['cal_id'].tolist():
                return {
                    'status': 'error',
                    'error_type': 'tool_not_in_dataset',
                    'selected_tool': tool_id,
                    'full_response': response,
                    'is_correct': False,
                    'full_input1': messages,
                    'full_output1': response,
                    'message': 'Selected tool not in dataset'
                }

            is_correct = str(tool_id) == str(row['cal_id'])
            
            return {
                'status': 'success',
                'selected_tool': tool_id,
                'full_response': response,
                'is_correct': is_correct,
                'full_input1': messages,
                'full_output1': response,
                'message': 'Wrong tool selected' if not is_correct else 'Success'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error_type': 'exception',
                'selected_tool': None,
                'full_response': None,
                'is_correct': False,
                'full_input1': None,
                'full_output1': None,
                'message': f'Exception occurred: {str(e)}'
            }

    def run_env2(self, row, data_meta, env1_result, reflection_hint = ''):

        def are_dicts_equal(dict1, dict2):
            try:
                return json.dumps(dict1, sort_keys=True) == json.dumps(dict2, sort_keys=True)
            except (TypeError, ValueError):
                return False

        try:
            if env1_result['status'] == 'error' and env1_result['error_type'] in ['invalid_tool_id', 'tool_not_in_dataset']:
                return {
                    'status': 'error',
                    'error_type': 'env1_failed',
                    'parsed_params': None,
                    'full_response': None,
                    'is_valid': False,
                    'is_correct':False,
                    'full_input2': None,
                    'full_output2': None,
                    'message': f"ENV1 failed: {env1_result['message']}"
                }

            tool_id = env1_result['selected_tool']
            patient_info = row['patient_info']
            input_schema = data_meta[data_meta['cal_id'] == int(tool_id)]['input_schema'].values[0]

            messages = [
                {
                    "role": "system",
                    "content": "You are a medical professional expert in parsing accurate parameters from patient information."
                },
                {
                    "role": "user",
                    "content": f"""Analyze the medical case and output parameters based on the schema.

Patient case:
{patient_info}

RULES:
1. Output format MUST be: {{"name": value}}
2. Use EXACT "name" fields from schema as keys
3. Include ALL fields from schema
4. Use ONLY values defined in schema options
5. Do NOT include the unit in the values

Schema:
{input_schema}"""
                }
            ]
            if reflection_hint:
                messages =  [
                {
                    "role": "system",
                    "content": "You are a medical professional expert in parsing accurate parameters from patient information."
                },{
        "role": "system",
        "content": f"Previous attempt feedback: {reflection_hint}"
    },{
                    "role": "user",
                    "content": f"""Analyze the medical case and output parameters based on the schema.

Patient case:
{patient_info}

RULES:
1. Output format MUST be: {{"name": value}}
2. Use EXACT "name" fields from schema as keys
3. Include ALL fields from schema
4. Use ONLY values defined in schema options
5. Do NOT include the unit in the values

Schema:
{input_schema}"""
                }
            ]
            response = self.chat_generate(messages)
            
            
            parsed_params = self.parse_response(response)

            correct_input_params = json.loads(
            row['inputs_raw'].replace("'", '"')
        )
            is_correct = True if are_dicts_equal(parsed_params, correct_input_params) else False

            if parsed_params is None:
                return {
                    'status': 'error',
                    'error_type': 'parsing_failed',
                    'parsed_params': None,
                    'full_response': response,
                    'is_valid': False,
                    'is_correct':False,
                    'full_input2': messages,
                    'full_output2': response,
                    'message': 'Failed to parse parameters from response'
                }

            is_valid = self.validate_schema(parsed_params, input_schema)

            return {
                'status': 'success',
                'parsed_params': parsed_params,
                'full_response': response,
                'is_valid': is_valid,
                'is_correct':is_correct,
                'full_input2': messages,
                'full_output2': response,
                'message': 'Invalid schema' if not is_valid else 'Success'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error_type': 'exception',
                'parsed_params': None,
                'full_response': None,
                'is_valid': False,
                'is_correct':False,
                'full_input2': None,
                'full_output2': None,
                'message': f'Exception occurred: {str(e)}'
            }

    def run_env3(self, row, env2_result, calc_result=None, reflection_hint = ''):
        try:
            question, _, _ = self.get_question_options(row)

            if env2_result['status'] == 'error':
                return {
                    'status': 'error',
                    'error_type': 'env2_failed',
                    'interpretation': f"Error: Previous stage failed - {env2_result['message']}",
                    'calc_result': None,
                    'full_input3': None,
                    'full_output3': None,
                    'message': f"ENV2 failed: {env2_result['message']}"
                }

            if calc_result is None or 'error' in calc_result:
                error_msg = "No calculation result available" if calc_result is None else calc_result['error']
                return {
                    'status': 'error',
                    'error_type': 'calc_failed',
                    'interpretation': f"Error: Calculator failed - {error_msg}",
                    'calc_result': calc_result,
                    'full_input3': None,
                    'full_output3': None,
                    'message': f"Calculator failed: {error_msg}"
                }

            messages = [
                {
                    "role": "system",
                    "content": "You are a medical expert specialized in clinical assessment and data interpretation."
                },
                {
                    "role": "user",
                    "content": f"""Based on the calculator's output:
{json.dumps(calc_result, indent=2)}
Please conclude the results and answer the question:
{question}"""
                }
            ]

            if reflection_hint:
                messages = [
                {
                    "role": "system",
                    "content": "You are a medical expert specialized in clinical assessment and data interpretation."
                },{
        "role": "system",
        "content": f"Previous attempt feedback: {reflection_hint}"
    }, {
                    "role": "user",
                    "content": f"""Based on the calculator's output:
{json.dumps(calc_result, indent=2)}
Please conclude the results and answer the question:
{question}"""
                }
            ]


            response = self.chat_generate(messages)

            return {
                'status': 'success',
                'interpretation': response,
                'calc_result': calc_result,
                'full_input3': messages,
                'full_output3': response,
                'message': 'Success'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error_type': 'exception',
                'interpretation': None,
                'calc_result': None,
                'full_input3': None,
                'full_output3': None,
                'message': f'Exception occurred: {str(e)}'
            }

    def run_env4(self, row, env3_result, reflection_hint = ''):
        try:
            q, options, correct_answer = self.get_question_options(row)
            if env3_result['status'] == 'error':


                messages = [
                {
                    "role": "system",
                    "content": "You are an experienced medical professional specializing in clinical decision-making."
                },
                {
                    "role": "user",
                    "content": f"""Patient Information: {row['patient_info']}   

Question: {q}

Select the best answer from:
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

Respond with format: Finish[A/B/C/D]"""
                }
            ]

            
            elif reflection_hint:
                messages = [
                {
                    "role": "system",
                    "content": "You are an experienced medical professional specializing in clinical decision-making."
                },{
        "role": "system",
        "content": f"Previous attempt feedback: {reflection_hint}"
    },{
                    "role": "user",
                    "content": f"""Based on the analysis:
{env3_result['interpretation']}

Select the best answer from:
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

Respond with format: Finish[A/B/C/D]"""
                }
            ]

            else:
                messages = [
                {
                    "role": "system",
                    "content": "You are an experienced medical professional specializing in clinical decision-making."
                },
                {
                    "role": "user",
                    "content": f"""Based on the analysis:
{env3_result['interpretation']}

Select the best answer from:
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

Respond with format: Finish[A/B/C/D]"""
                }
            ]

            response = self.chat_generate(messages,max_tokens=10)
            selected_option = self.extract_option(response)

            if not selected_option:
                return {
                    'status': 'error',
                    'error_type': 'invalid_option',
                    'selected_option': None,
                    'full_response': response,
                    'is_correct': False,
                    'full_input4': messages,
                    'full_output4': response,
                    'message': 'Failed to extract valid option from response'
                }

            return {
                'status': 'success',
                'selected_option': selected_option,
                'full_response': response,
                'is_correct': selected_option == correct_answer,
                'full_input4': messages,
                'full_output4': response,
                'message': 'Success'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error_type': 'exception',
                'selected_option': None,
                'full_response': None,
                'is_correct': False,
                'full_input4': None,
                'full_output4': None,
                'message': f'Exception occurred: {str(e)}'
            }

    def run_env5(self, row, env1_result, env2_result, env3_result, env4_result):
        messages = [
            {
                "role": "system",
                "content": "You are an expert clinical decision assistant. Analyze the case processing stages to identify the root cause of any errors affecting the final answer."
            },
            {
                "role": "user",
                "content": f"""Validate the following clinical case analysis stages for correctness:

Case Question: {self.get_question(row)}

Model's Answer: {env4_result['full_output4']}

1. Tool Selection:
Input: {env1_result.get('full_input1', 'N/A')}
Output: {env1_result.get('full_response', 'N/A')}

2. Parameter Extraction:
Input: {env2_result.get('full_input2', 'N/A')}
Output: {env2_result.get('full_response', 'N/A')}

3. Result Interpretation:
Input: {env3_result.get('full_input3', 'N/A')}
Output: {env3_result.get('full_output3', 'N/A')}

4. Answer Selection:
Input: {env4_result.get('full_input4', 'N/A')}
Output: {env4_result.get('full_output4', 'N/A')}

Required Output Format:
RESULT: Reflect[RIGHT/WRONG]
ANALYSIS:
"All stages processed correctly" / Stage_X: [ERROR] <error description>

Instructions:
- Response start with Reflect[RIGHT] or Reflect[WRONG] with ANALYSIS strictly follow the required format
- If error found, report only the earliest error stage
"""
            }
        ]

        response = self.chat_generate(messages, max_tokens=128)
        return {
            'output': response,
            'is_correct': 'Reflect[RIGHT]' in response
        }

    def extract_tool_id(self, text):
        patterns = [
            r'Tool_(\d+)', r'Tool (\d+)', r'Tool#(\d+)', r'Tool-(\d+)',
            r'Tool:\s*(\d+)', r'Tool\s*(\d+)', r'tool\s*(\d+)', r'(\d{4,})'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def extract_option(self, text):
        # match the following format:
        # Finish[A] / Finish[B] / Finish[C] / Finish[D]
        # Finish A) / Finish B) / Finish C) / Finish D)
        # Finish A / Finish B / Finish C / Finish D
        patterns = [
            r'Finish\s*\[([A-D])\]',         
            r'Finish\s*([A-D])\)',            
            r'Finish\s*([A-D])(?:\s|$)',      
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE) 
            if match:
                return match.group(1).upper()  
                
        return None

    def parse_response(self, response_str):
        try:
            start = response_str.find('{')
            end = response_str.rfind('}')
            if start == -1 or end == -1:
                return None
            
            dict_str = response_str[start:end+1]
            dict_str = dict_str.replace("'", '"')
            
            try:
                return json.loads(dict_str)
            except json.JSONDecodeError:
                return ast.literal_eval(dict_str)
            
        except Exception as e:
            return None

    def validate_schema(self, parsed_dict, input_schema_str):
        try:
            input_schema = ast.literal_eval(input_schema_str)
        except Exception:
            return False

        if parsed_dict is None:
            return False

        schema_dict = {item['name']: item for item in input_schema}
        required_names = set(schema_dict.keys())
        parsed_names = set(parsed_dict.keys())

        if required_names != parsed_names:
            return False

        for name, value in parsed_dict.items():
            schema_item = schema_dict[name]
            if schema_item['type'] == 'textbox':
                try:
                    float(value)
                except (ValueError, TypeError):
                    return False
            elif schema_item['type'] in ['radio', 'toggle', 'dropdown']:
                valid_values = [opt['value'] for opt in schema_item['options']]
                if value not in valid_values:
                    return False

        return True



    def save_checkpoint(self):
    
        try:
            checkpoint_data = {
                'results': self.results,
                'stats': self.stats,
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            }
            
            checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{timestamp}.json')
            
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=4)
            
            logging.info(f"Checkpoint saved to: {checkpoint_path}")
            
            self.checkpoint_path = checkpoint_path
            
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path) as f:
                data = json.load(f)
                self.results = data['results']
                self.stats = data['stats']
            return True
        return False


    def run_single_env(self, env_name, row, data, env_results, reflection_hint, max_env_retries=1):
        
        data_meta = self.tool_lib
        for retry in range(max_env_retries):
            if retry > 0:
                logging.info(f"Retrying {env_name} (attempt {retry + 1}/{max_env_retries})")
            
            if env_name == 'env1':
                result = self.run_env1(row, data_meta, reflection_hint=reflection_hint)
              
                is_valid = result['status'] == 'success'
            
            elif env_name == 'env2':
                if env_results.get('env1', {}).get('status') != 'success':
                    return {
                        'status': 'error',
                        'error_type': 'env1_failed',
                        'message': 'ENV1 did not complete successfully'
                    }, False
                    
                result = self.run_env2(row, data_meta, env_results['env1'], reflection_hint=reflection_hint)
               
                is_valid = result['status'] == 'success' and result.get('is_valid', False)
            
            elif env_name == 'env3':
                if env_results.get('env2', {}).get('status') != 'success':
                    return {
                        'status': 'error',
                        'error_type': 'env2_failed',
                        'message': 'ENV2 did not complete successfully'
                    }, False
                    
                calc_result = None
                if env_results['env2']['status'] == 'success' and env_results['env2'].get('is_valid'):
                    try:
                        calc_result = calculate_from_url(row['cal_id'], env_results['env2']['parsed_params'])
                    except Exception as e:
                        logging.error(f"Calculator API error: {str(e)}")
                        calc_result = {'error': str(e)}
                
                result = self.run_env3(row, env_results['env2'], calc_result, reflection_hint=reflection_hint)
                is_valid = result['status'] == 'success' and 'error' not in calc_result.get('error', '')
            
            elif env_name == 'env4':
                if env_results.get('env3', {}).get('status') != 'success':
                    return {
                        'status': 'error',
                        'error_type': 'env3_failed',
                        'message': 'ENV3 did not complete successfully'
                    }, False
                    
                result = self.run_env4(row, env_results['env3'], reflection_hint=reflection_hint)
                is_valid = (result['status'] == 'success' and 
                        result.get('selected_option') is not None and 
                        result.get('selected_option') in ['A', 'B', 'C', 'D'])
            
            result['retries'] = retry + 1 if retry > 0 else 0
            
            logging.info(f"{env_name} result: {result['status']} - {result['message']} - Valid: {is_valid}")
            
            if is_valid or retry == max_env_retries - 1:
                return result, is_valid
            
            if retry < max_env_retries - 1:
                reflection_hint = f"Previous {env_name} attempt failed: {result.get('message', 'Unknown error')}"
                if env_name == 'env4' and result.get('selected_option'):
                    reflection_hint += f" Selected option was: {result.get('selected_option')}"
        
        return result, False


    def run_pipeline(self, data_path, split='test', save_dir='', max_retries=3):
        if save_dir:
            self.output_dir = save_dir
        data = pd.read_excel(data_path)
        subset = data[data['split'] == split]
        subset = subset.sample(frac=1).iloc[:5]

        for idx, row in tqdm(subset.iterrows(), total=len(subset)):
            input_id = row['input_id']
            if str(input_id) in self.results:
                logging.info(f"Skipping already processed input_id: {input_id}")
                continue

            logging.info(f"Processing input_id: {input_id}")
            
            retry_count = 0
            env_results = {}
       
            best_env_results = {}
            reflection_hint = ""
            
            if str(input_id) not in self.results:
                self.results[str(input_id)] = {'rounds': []}
                
            while retry_count < max_retries:
              
                env1_result, env1_success = self.run_single_env('env1', row, data, env_results, reflection_hint)
                env_results['env1'] = env1_result
            
                if env1_success:
                    best_env_results['env1'] = env1_result
                
                if not env1_success:
                    logging.info(f"ENV1 failed for {input_id}, retrying...")
                    retry_count += 1
                    reflection_hint = f"Previous env1 attempt failed: {env1_result.get('message', 'Unknown error')}"
                    continue
                    
            
                env2_result, env2_success = self.run_single_env('env2', row, data, env_results, reflection_hint)
                env_results['env2'] = env2_result
               
                if env2_success:
                    best_env_results['env2'] = env2_result
                
                if not env2_success:
                    logging.info(f"ENV2 failed for {input_id}, retrying...")
                    retry_count += 1
                    reflection_hint = f"Previous env2 attempt failed: {env2_result.get('message', 'Unknown error')}"
                    continue
                    
               
                env3_result, env3_success = self.run_single_env('env3', row, data, env_results, reflection_hint)
                env_results['env3'] = env3_result
            
                if env3_success:
                    best_env_results['env3'] = env3_result
                
                if not env3_success:
                    logging.info(f"ENV3 failed for {input_id}, retrying...")
                    retry_count += 1
                    reflection_hint = f"Previous env3 attempt failed: {env3_result.get('message', 'Unknown error')}"
                    continue
                    
                # 执行 env4
                env4_result, env4_success = self.run_single_env('env4', row, data, env_results, reflection_hint)
                env_results['env4'] = env4_result
                # 保存最新有效结果
                if env4_success:
                    best_env_results['env4'] = env4_result
                
                if not env4_success:
                    logging.info(f"ENV4 failed for {input_id}, retrying...")
                    retry_count += 1
                    reflection_hint = f"Previous env4 attempt failed: {env4_result.get('message', 'Unknown error')}"
                    continue
                    
           
                env5_result = self.run_env5(row, env_results['env1'], env_results['env2'], 
                                        env_results['env3'], env_results['env4'])
                
                env_results['env5'] = env5_result
          
                best_env_results['env5'] = env5_result
                
                logging.info(f"ENV5 result for {input_id}: Correct={env5_result['is_correct']}, Analysis={env5_result['output']}")
                
             
                current_round = {
                    'env1': env_results.get('env1'),
                    'env2': env_results.get('env2'),
                    'env3': env_results.get('env3'),
                    'env4': env_results.get('env4'),
                    'env5': env5_result,
                    'retry_count': retry_count
                }
                self.results[str(input_id)]['rounds'].append(current_round)
                
             
                if env5_result['is_correct']:
                    logging.info(f"Pipeline completed successfully for {input_id} after {retry_count} retries")
                    break
                    
            
                problem_env = analyze_reflection(env5_result['output'])
                reflection_hint = extract_reflection_hint(env5_result['output'])
                
            
                if problem_env == 'env1' or problem_env == 'all':
               
                    env_results = {}
                elif problem_env == 'env2':
                 
                    env_results.pop('env2', None)
                    env_results.pop('env3', None)
                    env_results.pop('env4', None)
                    env_results.pop('env5', None)
                elif problem_env == 'env3':
                   
                    env_results.pop('env3', None)
                    env_results.pop('env4', None)
                    env_results.pop('env5', None)
                elif problem_env == 'env4':
                 
                    env_results.pop('env4', None)
                    env_results.pop('env5', None)
                    
                retry_count += 1
                
                if retry_count >= max_retries:
                    logging.warning(f"Maximum retries ({max_retries}) reached for {input_id}")
                    
        
            def get_env_error(env_results, env_name):
                env_result = env_results.get(env_name, {})
                if not env_result:
                    return None
                return None if env_result.get('status') == 'success' else env_result.get('message', f'Error in {env_name}')
            
     
            self.results[str(input_id)].update({
                'final_env_results': {
                    'env1': best_env_results.get('env1'),
                    'env2': best_env_results.get('env2'),
                    'env3': best_env_results.get('env3'),
                    'env4': best_env_results.get('env4'),
                    'env5': best_env_results.get('env5')
                },
                'total_retries': retry_count,
                'error_log': {
                    'env1_error': get_env_error(best_env_results, 'env1'),
                    'env2_error': get_env_error(best_env_results, 'env2'),
                    'env3_error': get_env_error(best_env_results, 'env3'),
                    'env4_error': get_env_error(best_env_results, 'env4'),
                    'env5_error': None if best_env_results.get('env5', {}).get('is_correct', False) else 'Reflection indicates errors',
                    'final_status': 'success' if best_env_results.get('env5', {}).get('is_correct', False) else f'failed_after_{retry_count}_retries'
                }
            })
                
            if idx % 5 == 0:
                self.save_checkpoint()
                
        self.save_checkpoint()
        self.save_results()
        return self.results


    def save_results(self):
        """
        Save pipeline results to files
        Only saves checkpoint, final results and env4 accuracy
        """
        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save final results
        results_path = os.path.join(output_dir, "final_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)
        
        # Calculate and save ENV4 accuracy
        total_cases = 0
        correct_cases = 0
        
        for case_id, case_data in self.results.items():
            try:
                # Add safety checks for nested dictionary access
                if (case_data 
                    and 'final_env_results' in case_data 
                    and case_data['final_env_results'] 
                    and 'env4' in case_data['final_env_results'] 
                    and case_data['final_env_results']['env4']):
                    
                    total_cases += 1
                    if case_data['final_env_results']['env4'].get('is_correct', False):
                        correct_cases += 1
                        
            except Exception as e:
                logging.warning(f"Error processing case {case_id}: {str(e)}")
                continue
        
        accuracy = correct_cases / total_cases if total_cases > 0 else 0
        
        # Save accuracy results
        accuracy_path = os.path.join(output_dir, "pred_accuracy.txt")
        with open(accuracy_path, 'w', encoding='utf-8') as f:
            f.write(f"ENV4 Evaluation Results\n")
            f.write(f"====================\n")
            f.write(f"Total cases: {total_cases}\n")
            f.write(f"Correct cases: {correct_cases}\n")
            f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        
        logging.info(f"Results saved to directory: {output_dir}")
        logging.info(f"Total cases processed: {total_cases}")
        logging.info(f"Correct cases: {correct_cases}")
        logging.info(f"Final accuracy: {accuracy:.4f}")



    

    



    

    
def main():
    """
    Main function to run the medical QA pipeline
    Supports Azure, OpenAI, and local/HuggingFace models
    """
    parser = argparse.ArgumentParser(description="Medical QA Pipeline")
    
    # Required arguments
    parser.add_argument("--data_path", required=True, help="Path to input data Excel file")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save outputs")
    parser.add_argument("--split", default="test", choices=["test", "train", "val"], help="Data split to process")
    
    # Model type and configuration
    parser.add_argument("--model_type", 
                       choices=["azure", "openai", "llama2", "llama3", "local"], 
                       required=True, 
                       help="Type of model to use")
    
    # Local/HuggingFace model arguments
    parser.add_argument("--model_path", 
                       help="Path to local model or HuggingFace model name")
    parser.add_argument("--device_map", 
                       default="auto", 
                       help="Device mapping for model deployment")
    
    # Cloud API arguments
    parser.add_argument("--api_key", 
                       help="API key for Azure/OpenAI")
    parser.add_argument("--api_base", 
                       help="API base URL for Azure")
    parser.add_argument("--api_version", 
                       default="2023-05-15", 
                       help="API version for Azure")
    parser.add_argument("--deployment", 
                       help="Model deployment name for Azure or model card for OpenAI")
    
    # Additional arguments
    parser.add_argument("--checkpoint", 
                       help="Path to checkpoint file to resume from")
    parser.add_argument("--wait_time", 
                       type=float, 
                       default=0.1, 
                       help="Wait time between API calls")
    parser.add_argument("--tool_lib_path",
                       default="data/tool_library.xlsx",
                       help="Path to tool library excel file")
    args = parser.parse_args()
    
    # Validate arguments based on model type
    if args.model_type in ["azure", "openai"]:
        if not args.api_key:
            raise ValueError(f"API key is required for {args.model_type}")
        if args.model_type == "azure" and not args.api_base:
            raise ValueError("API base URL is required for Azure")
        if not args.deployment:
            raise ValueError(f"Deployment/model name is required for {args.model_type}")
    
    elif args.model_type in ["llama2", "llama3", "local"]:
        if not args.model_path:
            raise ValueError("Model path is required for local/HuggingFace models")
    
    # Initialize pipeline based on model type
    if args.model_type in ["azure", "openai"]:
        pipeline = MedicalQAPipeline(
            model_type=args.model_type,
            deployment=args.deployment,
            api_key=args.api_key,
            api_base=args.api_base,
            api_version=args.api_version,
            wait_time=args.wait_time,
            output_dir=args.output_dir,
            tool_lib_path=args.tool_lib_path
        )
    else:
        pipeline = MedicalQAPipeline(
            model_type=args.model_type,
            model_path=args.model_path,
            device_map=args.device_map,
            output_dir=args.output_dir,
            tool_lib_path=args.tool_lib_path
        )
    
    # Load checkpoint if specified
    if args.checkpoint and os.path.exists(args.checkpoint):
        pipeline.load_checkpoint(args.checkpoint)
        logging.info(f"Loaded checkpoint from {args.checkpoint}")
    
    # Run pipeline
    try:
        results = pipeline.run_pipeline(
            data_path=args.data_path,
            split=args.split,
            save_dir=args.output_dir
        )
        logging.info("Pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise
    
    return results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main()