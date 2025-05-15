# =========================
# Imports
# =========================
from model import custom_model, get_tokenizer
from fastapi import HTTPException  # For raising API errors
import re  # For extracting JSON using regular expressions
import json  # For working with JSON data
import torch 
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

</Output_Properties>
"""

torch.cuda.empty_cache()

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
    if match:
        tags_json = match.group(1)
        try:
            tags = json.loads(tags_json)
            return tags
        except Exception as e:
            print("Failed to parse tags JSON:", e)
            return None
    else:
        print("No <Output_Properties> or <Output> JSON found in LLM output.")
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

    torch.cuda.empty_cache()

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
        )

    # Decode the output tokens to a string
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

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

# Example usage for testing the prompt and printing the result
if __name__ == "__main__":
    # Example ticket details
    subject = "ONE AYALA MALL (AVEPOINT): Access Badge Concern - April 22  2025"
    description = "ONE AYALA MALL (AVEPOINT): Access Badge Concern - April 22  2025 From: Joshua Tolentino Sent: Wednesday  April 23  2025 1:28 PMTo: KMCS One Ayala Mall Cc: Neil Cadigoy ; KMC Service Desk Subject: Re: [Access Badge Concern] - April 22  2025Hi @KMCS One Ayala Mall Kindly requesting your assistance regarding Neil'saccess &amp; . Upon trying thebadgedoesn't work on our office doors. Most of us doesn't have an access in our boardroom(H and I- Corfu and Elba).Neilrose Cadigoy2096202 - 0002499171Thank you and kind regards Romeo Joshua Tolentino[He/Him]Operations CoordinatorP: +63 949-755-6673 / +63 917-125-1569 | Joshua.Tolentino@avepoint.comFollow AvePoint on Xand LinkedIn| Subscribe to our blogIs your data ready for AI? Prepare your data for AI success with our AI &amp; Information Management Report.Please note that I respect and understand that our schedules may not always align due to different time zones. There's no need to respond during your off-business hours  over the weekends  or when you're on your personal time off. Have a safe and fantastic day!From: Joshua TolentinoSent: Tuesday  22 April 2025 4:58 pmTo: KMCS One Ayala Mall Cc: Neil Cadigoy Subject: [Access Badge Concern] - April 22  2025Hi @KMCS One Ayala Mall Kindly requesting your assistance regarding Neil's access. Upon trying the badge doesn't work on our office doors.Neilrose Cadigoy2096202 - 0002499171Thank you and kind regards Romeo Joshua Tolentino[He/Him]Operations CoordinatorP: +63 949-755-6673 / +63 917-125-1569 | Joshua.Tolentino@avepoint.comFollow AvePoint on Xand LinkedIn| Subscribe to our blogIs your data ready for AI? Prepare your data for AI success with our AI &amp; Information Management Report.Please note that I respect and understand that our schedules may not always align due to different time zones. There's no need to respond during your off-business hours  over the weekends  or when you're on your personal time off. Have a safe and fantastic day!"
    email = "polvoron.james@kmc.solutions"
    # Call the function and print the result
    result = get_ticket_tags(subject, description, email)
    print("Parsed Tags:", result)
