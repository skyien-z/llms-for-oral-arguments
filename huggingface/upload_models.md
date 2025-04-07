## Uploading finetuned models to HF

1. Login to della node with internet access and navigate to this repo
1. Activate conda env
    ```
    module load anaconda3/2024.6
    conda activate llama_finetuning_env
    ```
1. Login from the terminal by running: `huggingface-cli login`
1. Enter your HF access token when prompted. You will need to be added to the HF Space for your personal HF access token to work.
1. Create repo for model if it doesn't already exist:
    ```
    huggingface-cli repo create oral-args-llama8b-finetuned \
        --type model \
        --organization ai-law-society-lab
    ```
1. Rename your model's latest checkpoint folder to 'final_checkpoint'
1. Change the following:
    * In `adapter_config.json` file within the checkpoint: change the `base_model_name_or_path` to `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`. Make sure no `/` at the end!
    * In the `README.md`, set `base_model: unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
1. Confirm `HF_REPO_ID` and `MODEL_NAME` arguments on top of the `huggingface/upload_model_to_hf.py` script.
1. Run the script: `python huggingface/upload_model_to_hf.py`
