# RiskAgent

## Quick Start

We provide a demo for the risk agent pipeline, which can be found at `evaluate/riskagent_demo.ipynb`.

This supports report summary of risk prediction by a given patient information using RiskAgent model with just a simple setup.

```
from riskagent_pipeline import RiskAgentPipeline

pipeline = RiskAgentPipeline(
    model_type="llama3",
    model_path="jinge13288/RiskAgent-8B", 
    device_map="cuda:0",
    verbose=True
)

test_case = """
A 54-year-old female patient with a history of hypertension and diabetes presents to the clinic complaining of palpitations and occasional light-headedness. Her medical record shows a previous stroke but no history of congestive heart failure or vascular diseases like myocardial infarction or peripheral artery disease.
"""

results = pipeline.process_case(test_case)


print("\n=== Final Assessment ===")
print(results['final_output'])

```

## MedRisk Dataset

MedRisk is made up with two version (also available on [huggingface](https://huggingface.co/datasets/jinge13288/MedRisk-Bench)): 

- MedRisk-Quantity: `data/MedRisk-Quantity.xlsx`
- MedRisk-Qualitative: `data/MedRisk-Qualitative.xlsx`

Each Instance in the dataset contains the following information:

- `input_id`: unique id for each instance.	
- `cal_id`: The tool id for this question.
- `question`: the question stem
- `option_a`, `option_b`, `option_c`, `option_d`: the options for the question
- `correct_answer`: the correct answer for the question
- `split`: the split of the dataset, either `train`, `test`, or `val`
- `relevant_tools`: the full available tool list ordered with the relevance to the question.
- `inputs`: the input parameters for the tool calculation (human readable format)
- `inputs_raw`: the input parameters for the tool calculation (raw format)

## Train

We also provide the training data with the format of instruction-following data, this can be found at `data/fine_tune/ft_data.zip`. 

The trained model can be found at:

| Model                                                             | Model size                       | Base Model         |
| ----------------------------------------------------------------- | -------------------------------- | ---------------- |
| [RiskAgent-1B](https://huggingface.co/jinge13288/RiskAgent-1B)                 | 1B                          | [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)         |
| [RiskAgent-3B](https://huggingface.co/jinge13288/RiskAgent-3B)                 | 3B                           | [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)         |
| [RiskAgent-8B](https://huggingface.co/jinge13288/RiskAgent-8B)                 | 8B                           | [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)         |
| [RiskAgent-70B] Comming soon!                 | 70B                           | [Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)         |


Prior to utilizing our model, please ensure you have obtained the Llama licensing and access rights to the Llama model.

## Installation
Install necessary packages:

```
conda create -n riskagent python=3.9
pip install -r requirements.txt
```

## Evaluation:


### Baseline
The `evaluate_baseline.py` provides evaluation functions on OpenAI models and LLaMA-based models.

Run evluation with LLaMA based models:

```
python evaluate_baseline.py \
        --model_type llama3 \
        --model_path meta-llama/Meta-Llama-3-8B \
        --split test \
        --output_file llama3_pred_quantity.xlsx \
        --device_map "cuda:0" \
        --data_path data/MedRisk-Quantity.xlsx
```

`model_path` can be model card from huggingface or your local model path. <br>
`device_map`: ["auto", "cuda:0", "cuda:1", etc. ] note: please try to run on single GPU to avoid parallel erros, i.e. device_map="cuda:0" <br>
`model_type:` ["llama2", "llama3", "gpt"]
`data_path`: either `MedRisk-Quantity.xlsx` or `MedRisk-Qualitative.xlsx`

Run evluation with OpenAI models:
```
python evaluate_baseline.py 
        --model_type gpt \
        --api_key YOUR_API_KEY \
        --model_card gpt-4o \
        --split test \
        --output_file gpt4o_pred_quantity.xlsx \
        --data_path data/MedRisk-Quantity.xlsx
```

### Risk Agent

We provide inference on both OpenAI models and open source models (eg. LLaMA) for our risk agent reasoning framework.

Run evluation with LLaMA based models on MedRisk benchmark:

```
python evaluate_riskagent.py \
    --model_type llama3 \
    --model_path meta-llama/Meta-Llama-3-8B \
    --data_path data/MedRisk-Quantity.xlsx\
    --output_dir ./riskagent_llama3_quantity \
    --split test \
    --tool_lib_path data/tool_library.xlsx \
    --device_map "cuda:0"
```

Run evluation with OpenAI models:
```
python evaluate_riskagent.py \
    --model_type openai \
    --deployment gpt-4o \
    --api_key YOUR_API_KEY \
    --data_path data/MedRisk-Quantity.xlsx \
    --output_dir ./riskagent_gpt4o_quantity \
    --split test
```

Run evluation with OpenAI models via Azure:
```
python evaluate_riskagent.py \
    --model_type azure \
    --deployment gpt-4o \
    --api_key YOUR_API_KEY \
    --data_path data/MedRisk-Quantity.xlsx \
    --api_base YOUR_AZURE_ENDPOINT \
    --output_dir ./riskagent_gpt4o_quantity \
    --split test
```




### Acknowledgement

The Llama Family Models: [Open and Efficient Foundation Language Models](https://ai.meta.com/llama/)

LLaMA-Factory: [Unified Efficient Fine-Tuning of 100+ Language Models](https://github.com/hiyouga/LLaMA-Factory/tree/main)
