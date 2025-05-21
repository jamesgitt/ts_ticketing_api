from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv  # For loading environment variables

# Load environment variables from .env file
load_dotenv()

# HuggingFace model repository path
hf_model_path = "kmcs-casulit/ts_ticket_v1.0.0.3"
# HuggingFace access token (if required)
token = os.getenv("HF_TOKEN")

def custom_model(hf_model=hf_model_path):
    """
    Loads and returns the HuggingFace AutoModelForCausalLM model.
    Useful for direct model access (not wrapped in a pipeline).
    """
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        device_map='auto',
        token=token,
        low_cpu_mem_usage=False,
        # load in 8bit
        load_in_8bit=True,
    )
    return model

def get_tokenizer(hf_model=hf_model_path):
    """
    Loads and returns the HuggingFace AutoTokenizer for the specified model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model,
        token=token
    )
    return tokenizer
