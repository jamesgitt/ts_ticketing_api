# =========================
# Imports
# =========================
from model import custom_model, get_tokenizer
from fastapi import HTTPException  # For raising API errors
import re  # For extracting JSON using regular expressions
import json  # For working with JSON data

# =========================
# Prompt Template
# =========================
# Template for instructing the LLM to tag tickets with the correct properties.
# Includes context, strict output instructions, and example inputs/outputs.
template = """
<Task_Context>
You are an expert at tagging tickets with their correct properties.

You will be given:
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
<Output_Properties>
"""
tokenizer = get_tokenizer()
model = custom_model()

# =========================
# Utility Functions
# =========================
def extract_json(llm_output):
    # Try to find JSON inside <Output_Properties>...</Output_Properties>
    match = re.search(r"<Output_Properties>\s*({.*?})\s*</Output_Properties>", llm_output, re.DOTALL)
    if not match:
        # Fallback: Try to find JSON inside <Output>...</Output>
        match = re.search(r"<Output>\s*({.*?})\s*</Output>", llm_output, re.DOTALL)
    if not match:
        # Fallback: Try to find any JSON object in the output
        match = re.search(r"({.*?})", llm_output, re.DOTALL)
    if match:
        tags_json = match.group(1)
        try:
            tags = json.loads(tags_json)
            return tags
        except Exception as e:
            print("Failed to parse tags JSON:", e)
            print("Raw JSON string:", tags_json)
            return None
    else:
        print("No JSON found in LLM output.")
        return None

# =========================
# Main Tagging Function
# =========================
def get_ticket_tags(subject, description, email):
    """
    Given ticket details, render the prompt,
    tokenize it, pass the tokenized input to the model, and return the predicted tags as a dict.
    """
    # Prepare ticket information as a JSON string
    ticket_information = json.dumps({
        "subject": subject,
        "description": description,
        "email": email
    })

    # Render the prompt string (no vector DB, no RAG)
    prompt_str = template.format(ticket_information=ticket_information)

    # Tokenize the prompt string
    inputs = tokenizer(prompt_str, return_tensors="pt").to("cuda")

    # Generate output from the model
    import torch
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=2048,
        )

    # Decode the output tokens to a string
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("Raw Output: ", output_text)

    # Extract only the part of the output that comes after the prompt
    if output_text.startswith(prompt_str):
        result = output_text[len(prompt_str):].strip()
    else:
        result = output_text.strip()

    # Print the raw LLM output for inspection
    print("LLM Output:", result)

    # Extract tags from the LLM output
    tags = extract_json(result)

    # Validate and map LLM output keys to API output keys
    print("LLM tags output:", tags)
    if tags is None or not isinstance(tags, dict):
        raise HTTPException(status_code=500, detail="Failed to extract tags from LLM output")
        
    return {
        "department": tags.get("department"),
        "techgroup": tags.get("techgroup"),
        "category": tags.get("category"),
        "subcategory": tags.get("subcategory"),
        "priority": tags.get("priority")
    }
