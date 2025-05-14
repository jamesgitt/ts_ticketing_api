# Imports for HuggingFacePipeline (LangChain) and HuggingFace Transformers
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from dotenv import load_dotenv  # For loading environment variables

# Load environment variables from .env file
load_dotenv()

# HuggingFace model repository path
hf_model_path = "kmcs-casulit/ts_ticket_v1.0.0.5"
# HuggingFace access token (if required)
token = os.getenv("HF_TOKEN")

def custom_model_pipeline(hf_model=hf_model_path):
    """
    Returns a LangChain HuggingFacePipeline LLM object for text generation.
    Loads the tokenizer and model from HuggingFace, sets up a text-generation pipeline,
    and wraps it for LangChain compatibility.
    """
    # Load the tokenizer from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model, 
        token=token
    )
    
    # Load the causal language model from HuggingFace
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        device_map='auto',
        token=token,
        low_cpu_mem_usage=False
    )

    # Create a HuggingFace text-generation pipeline
    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,  # Maximum tokens to generate
        do_sample=True,       # Enable sampling
        temperature=0.1,      # Low temperature for more deterministic output
    )
    # Wrap the pipeline in a LangChain HuggingFacePipeline object
    return HuggingFacePipeline(pipeline=hf_pipe)

def custom_model(hf_model=hf_model_path):
    """
    Loads and returns the HuggingFace AutoModelForCausalLM model.
    Useful for direct model access (not wrapped in a pipeline).
    """
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        device_map='auto',
        token=token,
        low_cpu_mem_usage=False
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
