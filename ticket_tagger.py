# =========================
# Imports
# =========================
from model import custom_model, get_tokenizer
from fastapi import HTTPException  # For API error handling
import re  # For regex-based JSON extraction
import json  # For JSON serialization/deserialization

# =========================
# Prompt Template
# =========================
# This template instructs the LLM to tag tickets with the correct properties.
# It provides context, strict output instructions, and a couple of worked examples.
template = """
<Task_Context>
You are an expert at tagging tickets with their correct properties.

You will be given with:
Ticket information in JSON format (fields: subject, description, email).
A list of the possible ticket property values in JSON format (fields: department, techgroup, category, subcategory, priority).

Assign the most appropriate value for each property, using only the provided ticket information and the possible property values.
If you are unsure or the information is insufficient, set the property to null.
Do NOT invent or guess values outside the provided options.

<VERY IMPORTANT>
Return ONLY the JSON object for the ticket properties. Do NOT include any explanations, extra text, or formatting such as markdown or code blocks.
</VERY IMPORTANT>
</Task_Context>

<Examples>
Example 1:
<Ticket_Information>
{{"subject": "UPT20 (SO-NickScali) : PC assistance Chanel Tolentino", "description": "UPT20 (SO-NickScali) : PC assistance Chanel Tolentino", "email": "chanel.tolentino@company.com"}}
</Ticket_Information>
<Possible_Output>
{{"department": "Technology Services", "techgroup": "On-Site Support", "category": "Hardware", "subcategory": "Desktop/Laptop Problem", "priority": "P2 - General"}}
</Possible_Output>

Example 2:
<Ticket_Information>
{{"subject": "PODIUM | VITRO LINK | Alert OSPF Neighbor is Down - VITRO MAKATI PODIUM-S2", "description": "PODIUM | VITRO LINK | Alert OSPF Neighbor is Down - VITRO MAKATI PODIUM-S2", "email": "noc.alerts@company.com"}}
</Ticket_Information>
<Possible_Output>
{{"department": "Technology Services", "techgroup": "NOC", "category": "Outage", "subcategory": "ISP Outage", "priority": "P1 - Critical"}}
</Possible_Output>
</Examples>

<Ticket_Information>
{ticket_information}
</Ticket_Information>
<Output>
"""
tokenizer = get_tokenizer()
model = custom_model()

# =========================
# Utility Functions
# =========================
def extract_json(text):
    """
    Extract the first JSON object from the LLM output.
    Returns a dict if valid JSON, otherwise returns the string or None.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return match.group()  # Return the string if not valid JSON
    return None

# =========================
# Main Tagging Function
# =========================
def get_ticket_tags(subject, description, email):
    """
    Given ticket details, render the prompt,
    tokenize it, pass the tokenized input to the model, and return the predicted tags as a dict.
    """
    # Prepare ticket information as JSON string
    ticket_information = json.dumps({
        "subject": subject,
        "description": description,
        "email": email
    })

    # Render the prompt to a string (no vector DB, no RAG)
    prompt_str = template.format(ticket_information=ticket_information)

    # Tokenize the prompt string
    inputs = tokenizer(prompt_str, return_tensors="pt").to("cuda")

    # Generate output from the model
    import torch
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
        )

    # Decode the output tokens to string
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Only keep the part of the output that comes after the prompt
    if output_text.startswith(prompt_str):
        result = output_text[len(prompt_str):].strip()
    else:
        result = output_text.strip()

    # Print the raw LLM output for inspection
    print("LLM Output:", result)

    # Extract tags from the LLM output
    tags = extract_json(result)

    # Validate and map LLM output keys to API output keys
    if tags is None or not isinstance(tags, dict):
        raise HTTPException(status_code=500, detail="Failed to extract tags from LLM output")
        
    return {
        "department": tags.get("department"),
        "techgroup": tags.get("techgroup"),
        "category": tags.get("category"),
        "subcategory": tags.get("subcategory"),
        "priority": tags.get("priority")
    }
