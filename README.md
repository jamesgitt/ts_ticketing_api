Certainly! Here’s a professional and comprehensive `README.md` for your ticket tagging project, based on the code and data you’ve provided:

---

# OSP AUTO TICKET TAGGING MODEL

This project leverages a Hugging Face transformer model to automatically assign ticket properties (department, techgroup, category, subcategory, priority) to Technology Services support tickets. It is designed for robust, row-by-row evaluation of model performance on real-world ticket data.

## Features

- **Automated Ticket Tagging:** Uses a fine-tuned language model to predict ticket properties from subject, description, and email.
- **Evaluation Metrics:** Calculates F1, precision, recall, and accuracy for each property.
- **Robust Handling:** Ensures all required fields are present in the output, even if the model response is incomplete.
- **CSV Integration:** Reads and writes ticket data and results in CSV format for easy analysis.

## Project Structure

```
ts_ticketing_api/
├── tickets_log.csv                # Example ticket data (input)
├── ts_ticketing_test_results_1000.csv  # Test set for evaluation (input)
├── ts_ticketing_test_results_TEST_1000.csv # Output with model predictions and metrics
├── ... (your scripts and notebooks)
```

## Requirements

- Python 3.8+
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- CUDA-enabled GPU (for efficient inference)
- Google Colab (optional, for easy cloud execution)

## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies:**
   ```bash
   pip install pandas scikit-learn transformers
   ```

3. **Prepare your Hugging Face token:**
   - Store your token securely (e.g., in Google Colab: `from google.colab import userdata`).

4. **Download or prepare your test CSV:**
   - Place your test data in `ts_ticketing_test_results_1000.csv`.

## Usage

### Inference and Evaluation

The main script processes each ticket row, generates model predictions, and evaluates them:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import json
from sklearn.metrics import f1_score, precision_score, recall_score

# ... (see process_row function in previous messages)

# Load the CSV file
test_df = pd.read_csv("ts_ticketing_test_results_1000.csv")

# Apply the process_row function to each row
test_df = test_df.apply(process_row, axis=1)

# Save the updated CSV file
test_df.to_csv("ts_ticketing_test_results_TEST_1000.csv", index=False)
```

### Model Prompt

The model is prompted with a detailed instruction and examples to ensure consistent, JSON-only output. If the model output is incomplete, missing fields are filled with `null`.

### Handling Empty or Invalid Model Output

The script ensures that all required ticket property fields are present in the output, defaulting to `null` if the model does not provide them. This prevents errors during evaluation.

## Example Input/Output

**Input Ticket:**
```json
{
  "subject": "UPT20 (SO-NickScali) : PC assistance Chanel Tolentino",
  "description": "UPT20 (SO-NickScali) : PC assistance Chanel Tolentino",
  "email": "chanel.tolentino@company.com"
}
```

**Model Output:**
```json
{
  "department": "Technology Services",
  "techgroup": "On-Site Support",
  "category": "Hardware",
  "subcategory": "Desktop/Laptop Problem",
  "priority": "P2 - General"
}
```

## Customization

- **Model:** Change the `hf_model` variable to use a different Hugging Face model.
- **Prompt:** Edit the `ticket_prompt` string to adjust instructions or examples.
- **Metrics:** Extend or modify the `calculate_metric_for_keys` function for additional evaluation.
