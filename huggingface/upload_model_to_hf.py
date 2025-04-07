from huggingface_hub import HfApi, HfFolder

HF_REPO_ID = 'ai-law-society-lab/oral-args-llama8b-finetuned'
MODEL_NAME = 'finetuned_Meta-Llama-3.1-8B-Instruct-bnb-4bit_dialogue_style_e_4_lora_r_32_lr_2e-5_gas_4'
## MAKE SURE TO RENAME FINAL CHECKPOINT FOLDER TO 'final_checkpoint'
MODEL_PATH = f'finetune/models/{MODEL_NAME}/final_checkpoint'


api = HfApi()
api.upload_folder(
    folder_path=MODEL_PATH,
    repo_id=HF_REPO_ID,
    repo_type="model",
)

