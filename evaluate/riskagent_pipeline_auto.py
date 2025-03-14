import logging
import json
import re
import time
import ast
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import AzureOpenAI, OpenAI
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry
from uuid import uuid4

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def validate_tool_ids(tool_ids: List[str], tool_info: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """validate tool ids"""
    valid_ids = []
    invalid_ids = []
    valid_tool_ids = set(tool_info['cal_id'].astype(str))
    
    for tool_id in tool_ids:
        if tool_id in valid_tool_ids:
            valid_ids.append(tool_id)
        else:
            invalid_ids.append(tool_id)
            
    return valid_ids, invalid_ids

def validate_parameters(params: Dict, schema: Any) -> Tuple[bool, List[str]]:
    """Validate parameters against schema
    
    Args:
        params: Dictionary of parameter values
        schema: Schema definition (string or dict)
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
    """
    errors = []
    
    try:
        # Convert schema string to dictionary
        if isinstance(schema, str):
            try:
                schema = ast.literal_eval(schema)
            except (ValueError, SyntaxError) as e:
                return False, [f"Invalid schema format: {str(e)}"]
        
        if not isinstance(schema, list):
            return False, ["Schema must be a list of parameter definitions"]
            
        if not isinstance(params, dict):
            return False, ["Parameters must be a dictionary"]
            
        # Create schema lookup dictionary
        schema_dict = {item['name']: item for item in schema}
        required_names = set(schema_dict.keys())
        parsed_names = set(params.keys())
        
        # Check required parameters
        missing_params = required_names - parsed_names
        if missing_params:
            errors.append(f"Missing required parameters: {', '.join(missing_params)}")
            
        # Check extra parameters
        extra_params = parsed_names - required_names
        if extra_params:
            errors.append(f"Unexpected parameters: {', '.join(extra_params)}")
            
        # Validate each parameter's value
        for name, value in params.items():
            if name not in schema_dict:
                continue
                
            schema_item = schema_dict[name]
            
            # Check value type
            if schema_item['type'] == 'textbox':
                try:
                    float(value)
                except (ValueError, TypeError):
                    errors.append(f"Parameter '{name}' must be a number")
                    
            # Check option type
            elif schema_item['type'] in ['radio', 'toggle', 'dropdown']:
                valid_values = [str(opt['value']) for opt in schema_item['options']]
                if str(value) not in valid_values:
                    errors.append(
                        f"Invalid value for '{name}'. Must be one of: {', '.join(valid_values)}"
                    )
            
            # Check required
            if schema_item.get('required', True) and (value is None or value == ''):
                errors.append(f"Parameter '{name}' is required")
                
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
        
    return len(errors) == 0, errors

def extract_tool_ids(text: str) -> List[str]:
    """extract tool ids from ENV1 output"""
    pattern = r'Tool_(\d+)'
    return re.findall(pattern, text, re.IGNORECASE)

def parse_params(response_str: str) -> Dict:
    """parse parameters from ENV2 output"""
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
        logging.error(f"Parameter parsing error: {str(e)}")
        return None

def calculate_from_url(calc_id: str, params: Dict) -> Dict:
    """Calculate risk score from MDCalc API"""
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
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest',
            'Origin': 'https://www.mdcalc.com',
            'Referer': 'https://www.mdcalc.com/'
        }
        
        # logging.info(f"Sending API request to: {api_url}")
        # logging.info(f"Request params: {params}")

        response = requests.post(
            api_url, 
            json=request_body, 
            headers=headers, 
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            # logging.info(f"API response: {result}")
            return result
        else:
            logging.error(f"API error: {response.status_code}")
            return {
                'score': 'N/A',
                'interpretation': f'API Error: Status {response.status_code}'
            }
            
    except requests.Timeout:
        logging.error("API request timeout")
        return {
            'score': 'N/A',
            'interpretation': 'Request timeout'
        }
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return {
            'score': 'N/A',
            'interpretation': f'Error: {str(e)}'
        }
    
def retry_with_backoff(func, max_retries=3, initial_delay=1):
    """general retry function, using exponential backoff"""
    def wrapper(*args, **kwargs):
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    return result
            except Exception as e:
                last_exception = e
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(delay)
                delay *= 2  # exponential backoff
                continue
            
        raise ValueError(f"Failed after {max_retries} attempts. Last error: {str(last_exception)}")
    
    return wrapper

class RiskAgentPipeline:
    def __init__(
        self,
        model_type: str = "azure",
        model_path: Optional[str] = None,
        device_map: str = "auto",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        tool_library: str = "../data/tool_library.xlsx",
        tool_mode: str = "auto",  # API, local, auto
        verbose: bool = False
    ):
        
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        """initialize pipeline"""
        self.verbose = verbose
        log_level = logging.INFO if verbose else logging.WARNING
        # logging.basicConfig(
        #     level=log_level,
        #     format='%(asctime)s - %(levelname)s - %(message)s'
        # )

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',  
            handlers=[logging.StreamHandler()]
        )
        
        # Validate tool_mode
        self.tool_mode = tool_mode.lower()
        if self.tool_mode not in ["API", "local", "auto"]:
            raise ValueError(f"Unsupported tool mode: {tool_mode}. Must be one of: API, local, auto")
        
        self.model_type = model_type.lower()
        if self.model_type not in ["azure", "openai", "llama3"]:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        if self.model_type in ["azure", "openai"] and not api_key:
            raise ValueError(f"API key required for {model_type}")
            
        if self.model_type == "llama3" and not model_path:
            raise ValueError("Model path required for llama3")
            
        # initialize API client
        if model_type == "azure":
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=api_endpoint
            )
            self.deployment = deployment
            
        elif model_type == "openai":
            self.client = OpenAI(api_key=api_key)
            self.deployment = deployment
            
        elif model_type == "llama3":
            try:
                logging.info(f"Loading Llama3 model from {model_path}")
                self.model, self.tokenizer = self.load_local_model(
                    model_path=model_path,
                    device_map=device_map
                )
                logging.info("Llama3 model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load Llama3 model: {str(e)}")
                
        # load tool library
        try:
            self.tool_info = pd.read_excel(tool_library)
            self.tool_info = self.tool_info[['cal_id','name','description','formula','more_info','input_schema','url']].drop_duplicates()
        except Exception as e:
            raise RuntimeError(f"Failed to load tool library: {str(e)}")

    @retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True
    )
    @sleep_and_retry
    @limits(calls=6000, period=60)
    def chat_generate(self, messages: List[Dict], temperature=0.6, max_tokens=4096) -> Optional[str]:
        
        try:
            if self.model_type in ["azure", "openai"]:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=60 if self.model_type == "azure" else 30
                )
                return response.choices[0].message.content
                
            elif self.model_type == "llama3":
                try:
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
                        torch.cuda.empty_cache()
                    raise e
                    
        except Exception as e:
            logging.error(f"API error ({self.model_type}): {str(e)}")
            if "timeout" in str(e).lower():
                logging.warning("Request timeout, retrying...")
            if "rate" in str(e).lower():
                logging.warning("Rate limit reached, waiting before retry...")
                time.sleep(2)
            raise e

    def load_local_model(self, model_path: str, device_map: str):
        """load local llama model"""
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.bfloat16
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # set llama2 chat template
        if self.model_type == "llama2" and tokenizer.chat_template is None:
            tokenizer.chat_template = """<s>[INST] {% if messages[0]['role'] == 'system' %}<<SYS>>{{ messages[0]['content'] }}<</SYS>>

{% endif %}{{ messages[1]['content'] }} [/INST]"""
            
        return model, tokenizer
    
    def get_tool_name(self, tool_id: str) -> str:
        """get tool name from tool_id"""
        try:
            return self.tool_info[self.tool_info['cal_id'].astype(str) == str(tool_id)]['name'].iloc[0]
        except Exception as e:
            logging.warning(f"Failed to get tool name for ID {tool_id}: {str(e)}")
            return f"Tool_{tool_id}"

    def calculate_risk_locally(self, tool_id: str, params: Dict) -> Dict:
        """Calculate risk score using local model"""
        try:
            # Get tool information
            tool_row = self.tool_info[self.tool_info['cal_id'].astype(str) == str(tool_id)]
            if tool_row.empty:
                return {'score': 'N/A', 'interpretation': f'Tool ID {tool_id} not found'}
            
            tool_name = tool_row['name'].iloc[0]
            tool_formula = tool_row['formula'].iloc[0] if 'formula' in tool_row and not pd.isna(tool_row['formula'].iloc[0]) else "Formula not provided"
            tool_desc = tool_row['description'].iloc[0] if 'description' in tool_row else ""
            
            # Prepare prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are a precise medical risk calculation assistant who can accurately calculate risk scores based on patient parameters and calculation formulas."
                },
                {
                    "role": "user", 
                    "content": f"""Calculate the risk score based on the following risk assessment tool's formula and provided parameters.
                    
Tool Name: {tool_name}
Tool Description: {tool_desc}
Calculation Formula: {tool_formula}

Parameters:
{json.dumps(params, indent=2, ensure_ascii=False)}

Please follow these steps:
1. Calculate the score based on the formula and parameters
2. Explain the calculation process and result meaning
3. Return results in JSON format, including "score" and "interpretation" fields
4. If calculation is not possible, please explain why

Result format example:
{{
    "score": "calculated score",
    "interpretation": "result explanation",
    "details": {{
        "risk_level": "risk level",
        "probability": "probability (if available)"
    }}
}}
"""
                }
            ]
            
            # Use model for calculation
            response = self.chat_generate(messages)
            # print(response)
            
            # Try to extract JSON result from response
            try:
                # Find JSON format content
                json_match = re.search(r'({[\s\S]*})', response)
                if json_match:
                    result_json = json.loads(json_match.group(1))
                    # Ensure return contains necessary fields
                    if 'score' not in result_json:
                        result_json['score'] = 'N/A'
                    if 'interpretation' not in result_json:
                        result_json['interpretation'] = 'Local calculation completed, but no interpretation provided'
                    return result_json
                else:
                    # If JSON cannot be extracted, create basic result
                    return {
                        'score': 'N/A',
                        'interpretation': response[:500] + ('...' if len(response) > 500 else '')
                    }
            except Exception as e:
                logging.error(f"Error parsing local calculation result: {str(e)}")
                return {
                    'score': 'N/A',
                    'interpretation': response[:500] + ('...' if len(response) > 500 else ''),
                    'error': str(e)
                }
                
        except Exception as e:
            logging.error(f"Local risk calculation error: {str(e)}")
            return {'score': 'N/A', 'interpretation': f'Local risk calculation failed: {str(e)}'}

    def calculate_risk(self, tool_id: str, params: Dict) -> Dict:
        """Calculate risk score based on tool_mode"""
        try:
            # Choose calculation method based on different modes
            if self.tool_mode == "API":
                # Use API calculation only
                result = calculate_from_url(str(tool_id), params)
                
            elif self.tool_mode == "local":
                # Use local calculation only
                result = self.calculate_risk_locally(tool_id, params)
                
            elif self.tool_mode == "auto":
                # Try API calculation first, fall back to local calculation if failed
                try:
                    result = calculate_from_url(str(tool_id), params)
                    # Check if API calculation succeeded
                    if result.get('score') == 'N/A' or 'error' in result:
                        logging.info(f"API calculation failed for tool {tool_id}, falling back to local calculation")
                        result = self.calculate_risk_locally(tool_id, params)
                except Exception as e:
                    logging.warning(f"API calculation error: {str(e)}, falling back to local calculation")
                    result = self.calculate_risk_locally(tool_id, params)
            
            # Check result validity
            if isinstance(result, dict) and 'error' not in result:
                return result
            return {'score': 'N/A', 'interpretation': 'Unable to calculate risk score'}
            
        except Exception as e:
            logging.error(f"Risk calculation error: {str(e)}")
            return {'score': 'N/A', 'interpretation': 'Risk calculation failed'}

    def generate_default_interpretation(self, patient_info: str, tool_id: str) -> str:
        """generate default interpretation when normal processing fails"""
        messages = [
            {
                "role": "system",
                "content": "You are a medical expert providing conservative risk assessments."
            },
            {
                "role": "user",
                "content": f"""Based on patient information, provide a conservative risk assessment.

Patient Information:
{patient_info}

Tool: {self.get_tool_name(tool_id)}

Please identify:
1. Potential high-risk conditions, key isk factors and your evidence. 
2. Think step by step and try to use the tools provided that may be useful for the risk assessment.
3. Basic recommendations
Please make sure your summary is consistent with the patient's information. Please be concise and precise. Shorten your summaries to 100-200 words.
"""
            }
        ]
        return self.chat_generate(messages)

    def generate_basic_assessment(self, patient_info: str) -> str:
        """generate basic risk assessment based only on patient information"""
        messages = [
            {
                "role": "system",
                "content": "You are a medical expert providing basic risk assessments."
            },
            {
                "role": "user",
                "content": f"""Provide a basic risk assessment based only on patient information.

Patient Information:
{patient_info}

Please identify:
1. Potential high-risk conditions, key isk factors and your evidence. 
2. Think step by step and try to recll some relevant tools that may be useful for the risk assessment.
3. Basic recommendations
Please make sure your summary is consistent with the patient's information. Please be concise and precise. Shorten your summaries to 100-200 words.
"""
            }
        ]
        return self.chat_generate(messages)

    

    def generate_final_output(self, patient_info: str, env4_output: str, env5_output: str) -> str:
        """generate final high-risk assessment output"""
        messages = [
            {
                "role": "system",
                "content": "You are a medical expert specialized in risk assessment. Provide concise but comprehensive high-risk predictions."
            },
            {
                "role": "user",
                "content": f"""Based on the comprehensive assessment and reflection, provide a focused summary of HIGH-RISK predictions only.

Patient Information:
{patient_info}

Comprehensive Assessment:
{env4_output}

Reflection and Analysis:
{env5_output}

Please provide a concise summary focusing ONLY on:
1. HIGH-RISK conditions identified (with probability if available)
2. Key evidence supporting each high-risk prediction
3. Brief, focused recommendations for immediate attention

Please make sure your summary is consistent with the patient's information. Please be concise and precise. Shorten your summaries to 100-200 words.
"""
            }
        ]
        
        return self.chat_generate(messages)
    
    

    def run_env1(self, patient_info: str, n=1) -> Tuple[str, List[str]]:
        """tool selection stage"""
        logging.info("Running tool selection...")
        common_calculators = [
        801, 324, 1239, 340, 1725, 617, 715, 89, 1736, 3900, 
        3991, 2048, 3316, 3320, 10524, 10190, 3992, 3816, 3170, 4037
    ]
        if self.model_type == "llama3":
        # use common calculators
            avail_tools = [
                f"Tool_{row['cal_id']}. {row['name']}: {row['description']}"
                for _, row in self.tool_info.iterrows()
                if int(row['cal_id']) in common_calculators
            ]
        else:
            avail_tools = [
                f"Tool_{row['cal_id']}. {row['name']}: {row['description']}"
                for _, row in self.tool_info.iterrows()
            ]
        avail_tools_str = '\n'.join(avail_tools)
        
        messages = [
            {
                "role": "system",
                "content": "You are a medical professional expert in selecting appropriate clinical risk assessment tools."
            },
            {
                "role": "user",
                "content": f"""Based on the patient's discharge summary, select the {n} most relevant risk assessment tools to evaluate potential disease risks.

Patient Information:
{patient_info}

Available Tools:
{avail_tools_str}

IMPORTANT: You must ONLY select tools from the provided list using their exact Tool_XX format.

Please list the {n} most relevant tools in order of relevance using this format:
Tool_xx. [tool name]
Justification: Brief explanation of why this tool is relevant (1-2 sentences)"""
            }
        ]
        
        response = self.chat_generate(messages)
        
        # validate tool ids
        tool_ids = extract_tool_ids(response)
        valid_ids, invalid_ids = validate_tool_ids(tool_ids, self.tool_info)
        
        if invalid_ids:
            logging.warning(f"find invalid tool ids: {invalid_ids}")
            
        if not valid_ids:
            raise ValueError("no valid tool ids found")
        
        return response, valid_ids

    def run_env2(self, patient_info: str, tool_id: str, max_retries=2) -> Tuple[str, Dict]:
        """parameter extraction stage"""
        logging.info("Extracting parameters...")
        input_schema = self.tool_info[self.tool_info['cal_id'].astype(str) == tool_id]['input_schema'].values[0]
        tool_name = self.tool_info[self.tool_info['cal_id'].astype(str) == tool_id]['name'].values[0]
        
        for attempt in range(max_retries + 1):
            messages = [
                {
                    "role": "system",
                    "content": "You are a medical professional expert in extracting clinical parameters from patient records."
                },
                {
                    "role": "user",
                    "content": f"""Extract ALL required parameters for the {tool_name} risk calculator from the patient's discharge summary.

Patient Information:
{patient_info}

CRITICAL REQUIREMENTS:
1. Output format MUST be: {{"name": value}}
2. Use EXACT "name" fields from schema as keys
3. ALL fields in schema are required - DO NOT SKIP ANY
4. Use ONLY values defined in schema options
5. If a parameter is not directly mentioned, use clinical reasoning to estimate based on available information
6. Explain your reasoning for each parameter

Schema:
{input_schema}

Previous attempt issues: {"None" if attempt == 0 else self.last_validation_errors}"""
                }
            ]
            
            response = self.chat_generate(messages)
            params = parse_params(response)
            
            if params:
                is_valid, errors = validate_parameters(params, input_schema)
                if is_valid:
                    return response, params
                else:
                    self.last_validation_errors = errors
                    if attempt == max_retries:
                        raise ValueError(f"parameter validation failed: {'; '.join(errors)}")
            else:
                if attempt == max_retries:
                    raise ValueError("cannot parse parameter format")
                
    def run_env1_with_retry(self, patient_info: str, max_retries=3) -> Tuple[str, List[str]]:
        """tool selection stage with retry mechanism"""
        for attempt in range(max_retries):
            try:
                # logging.info(f"\nattempt ENV1 (attempt {attempt + 1}/{max_retries})...")
                env1_output, valid_tool_ids = self.run_env1(patient_info)
                
                if valid_tool_ids:  # validate if valid tool ids are obtained
                    return env1_output, valid_tool_ids
                else:
                    logging.warning(f"ENV1 attempt {attempt + 1}: No valid tool IDs found")
                    
            except Exception as e:
                logging.warning(f"ENV1 attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # exponential backoff
                
        raise ValueError("Failed to get valid tool IDs after all retries")

    def run_env2_with_retry(self, patient_info: str, tool_id: str, max_retries=3) -> Tuple[str, Dict]:
        """parameter extraction stage with retry mechanism"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # logging.info(f"\nattempt ENV2 for tool {tool_id} (attempt {attempt + 1}/{max_retries})...")
                env2_output, params = self.run_env2(patient_info, tool_id)
                
                # validate parameters
                if params and isinstance(params, dict):
                    # get tool's parameter schema
                    tool_schema = self.tool_info[self.tool_info['cal_id'].astype(str) == tool_id]['input_schema'].iloc[0]
                    is_valid, errors = validate_parameters(params, tool_schema)
                    
                    if is_valid:
                        return env2_output, params
                    else:
                        last_error = f"Parameter validation failed: {'; '.join(errors)}"
                        logging.warning(f"ENV2 attempt {attempt + 1}: {last_error}")
                else:
                    last_error = "Invalid parameter format"
                    logging.warning(f"ENV2 attempt {attempt + 1}: {last_error}")
                    
            except Exception as e:
                last_error = str(e)
                logging.warning(f"ENV2 attempt {attempt + 1} failed: {last_error}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
                
        raise ValueError(f"Failed to extract valid parameters after {max_retries} attempts. Last error: {last_error}")


    def run_env3(self, patient_info: str, calc_result: Dict, tool_id: str) -> str:
        """risk interpretation stage"""
        logging.info("Generating final assessment...")
        tool_name = self.tool_info[self.tool_info['cal_id'].astype(str) == tool_id]['name'].values[0]
        tool_desc = self.tool_info[self.tool_info['cal_id'].astype(str) == tool_id]['description'].values[0]
        
        messages = [
            {
                "role": "system",
                "content": "You are a medical expert specialized in clinical risk assessment and interpretation."
            },
            {
                "role": "user",
                "content": f"""Interpret the risk calculator results in the context of the patient's condition.

Patient Information:
{patient_info}

Tool: {tool_name}
Description: {tool_desc}

Calculator Results:
{json.dumps(calc_result, indent=2)}

Please provide:
1. Risk level assessment
2. Clinical significance
3. Key factors contributing to the risk
4. Potential implications for patient care"""
            }
        ]
        
        return self.chat_generate(messages)

    def run_env4(self, patient_info: str, tool_outputs: str) -> str:
        """final conclusion stage"""
        messages = [
            {
                "role": "system",
                "content": "You are a medical expert specialized in comprehensive risk assessment and patient care planning."
            },
            {
                "role": "user",
                "content": f"""Based on all risk assessments performed, provide a precise analysis of the patient's disease risks.

Patient Information:
{patient_info}

Risk Assessment Results:
{tool_outputs}

Please provide:
1. Summary of significant disease risks identified
2. Overall risk assessment
3. Key areas of concern
4. Recommended monitoring or preventive measures
5. Suggestions for risk mitigation
Please make sure your summary is consistent with the patient's information. Please be concise and precise. Shorten your summaries to 100-200 words."""
            }
        ]
        
        return self.chat_generate(messages)

    def run_env5(self, patient_info: str, env1_output: str, env2_output: str, 
                 env3_output: str, env4_output: str) -> str:
        """reflection stage"""
        messages = [
            {
                "role": "system",
                "content": "You are an expert in clinical decision support and quality assurance."
            },
            {
                "role": "user",
                "content": f"""Review the risk assessment process and results for completeness and accuracy.

Patient Information:
{patient_info}

Process Stages:
1. Tool Selection:
{env1_output}

2. Parameter Extraction:
{env2_output}

3. Risk Calculation Interpretation:
{env3_output}

4. Final Assessment:
{env4_output}

Please evaluate:
1. Appropriateness of selected tools
2. Accuracy of parameter extraction
3. Validity of risk interpretations
4. Comprehensiveness of final assessment
5. Any potential gaps or areas for improvement

Output Format:
RESULT: Reflect[COMPLETE/INCOMPLETE]
ANALYSIS: If INCOMPLETE, anything to add. Keep your response brief and focused."""
            }
        ]
        
        return self.chat_generate(messages)
    
    

    def process_case(self, patient_info: str) -> Dict:
        """Process a single case with comprehensive error handling"""
        results = {
            'tool_results': [],
            'errors': [],
            'status': 'pending',
            'final_output': None
        }
        
        try:
            # ENV1: Tool Selection
            # logging.info("Running ENV1: Tool Selection...")
            env1_output, valid_tool_ids = self.run_env1_with_retry(patient_info)
            results['env1_output'] = env1_output
            
            # process each tool
            successful_tools = []
            for tool_id in valid_tool_ids:
                tool_result = {
                    'tool_id': tool_id,
                    'status': 'pending'
                }
                
                try:
                    # ENV2: Parameter Extraction
                    # logging.info(f"Processing tool {tool_id}")
                    env2_output, params = self.run_env2_with_retry(patient_info, tool_id)
                    tool_result['env2_output'] = env2_output
                    tool_result['parameters'] = params
                    
                    # Risk Calculation
                    calc_result = self.calculate_risk(tool_id, params)
                    tool_result['calc_result'] = calc_result

                    # print(calc_result)
                    
                    # ENV3: Risk Interpretation
                    env3_output = self.run_env3(patient_info, calc_result, tool_id)
                    tool_result['env3_output'] = env3_output
                    tool_result['status'] = 'success'
                    successful_tools.append(tool_result)
                    
                except Exception as e:
                    tool_result['status'] = 'error'
                    tool_result['error'] = str(e)
                    logging.error(f"Tool {tool_id} processing error: {str(e)}")
                
                results['tool_results'].append(tool_result)
            
            # ENV4 & ENV5: Final Assessment and Reflection
            if successful_tools:
                # ENV4: Comprehensive Assessment
                env4_output = self.run_env4(patient_info, successful_tools)
                results['env4_output'] = env4_output
                
                # prepare input for ENV5
                env2_outputs = [tool.get('env2_output', '') for tool in successful_tools]
                env3_outputs = [tool.get('env3_output', '') for tool in successful_tools]
                
                # ENV5: Reflection
                env5_output = self.run_env5(
                    patient_info=patient_info,
                    env1_output=env1_output,
                    env2_output=json.dumps(env2_outputs, ensure_ascii=False),
                    env3_output=json.dumps(env3_outputs, ensure_ascii=False),
                    env4_output=env4_output
                )
                results['env5_output'] = env5_output
                
                # Generate Final Output
                final_output = self.generate_final_output(patient_info, env4_output, env5_output)
                results['final_output'] = final_output
                results['status'] = 'success'
            else:
                # if no successful tool results, generate basic assessment
                results['final_output'] = self.generate_basic_assessment(patient_info)
                results['status'] = 'partial_success'
                
        except Exception as e:
            logging.error(f"Pipeline error: {str(e)}")
            results['errors'].append(f"Pipeline error: {str(e)}")
            results['final_output'] = self.generate_basic_assessment(patient_info)
            results['status'] = 'minimal_success'
        
        logging.info("Case analysis completed")
        return results 
