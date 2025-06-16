
---

# TS Ticketing API — Process Overview

The `ts_ticketing_api` is a RESTful API service that uses a fine-tuned Hugging Face transformer model to automatically tag IT support tickets with the correct properties. Below is a step-by-step outline of how the API works:

---

## 1. **API Input**

- **Endpoint:** The API exposes an endpoint (e.g., `/tag_ticket`) that accepts HTTP POST requests.
- **Request Body:** The client sends a JSON payload containing ticket information:
  ```json
  {
    "subject": "Ticket subject here",
    "description": "Detailed ticket description here",
    "email": "user@example.com"
  }
  ```

---

## 2. **Model Inference**

- **Prompt Construction:** The API constructs a prompt for the model, embedding the ticket information in a predefined template with clear instructions and examples.
- **Tokenization:** The prompt is tokenized using the model’s tokenizer.
- **Model Call:** The tokenized prompt is passed to the fine-tuned transformer model (e.g., `kmcs-casulit/ts_ticket_v1.0.0.9`), which generates a response containing the predicted ticket properties.

---

## 3. **Post-processing**

- **Parsing:** The model’s output is parsed as JSON.
- **Validation:** The API ensures all required ticket property fields are present in the response:
  - `department`
  - `techgroup`
  - `category`
  - `subcategory`
  - `priority`
- **Fallback:** If the model output is missing any fields or is invalid, those fields are set to `null` to guarantee a complete response.

---

## 4. **API Output**

- **Response:** The API returns a JSON object with the predicted ticket properties:
  ```json
  {
    "department": "Technology Services",
    "techgroup": "Service Desk",
    "category": "General Assistance",
    "subcategory": "General Inquiry",
    "priority": "P3 - Planned"
  }
  ```
- **Error Handling:** If the input is invalid or the model fails, the API returns an appropriate error message or a response with all fields set to `null`.

---

## **Summary Table**

| Step            | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| Input           | JSON with ticket info (`subject`, `description`, `email`)                   |
| Prompting       | Construct prompt for the model                                              |
| Inference       | Model generates ticket property JSON                                        |
| Post-processing | Ensure all required fields are present; fill missing with `null`            |
| Output          | Return JSON with predicted ticket properties                                |

---

## **Example Usage**

**Request:**
```http
POST /tag_ticket
Content-Type: application/json

{
  "subject": "UPT20 (SO-NickScali) : PC assistance Chanel Tolentino",
  "description": "UPT20 (SO-NickScali) : PC assistance Chanel Tolentino",
  "email": "chanel.tolentino@company.com"
}
```

**Response:**
```json
{
  "department": "Technology Services",
  "techgroup": "On-Site Support",
  "category": "Hardware",
  "subcategory": "Desktop/Laptop Problem",
  "priority": "P2 - General"
}
```

---
```
**Current accuracies:**

------------------------------------------------------------
Accuracy Percentage: 82.00%
------------------------------------------------------------
F1 Percentage for Department: 100.00%
F1 Percentage for Techgroup: 93.00%
F1 Percentage for Category: 93.00%
F1 Percentage for Subcategory: 85.00%
F1 Percentage for Priority: 94.00%
------------------------------------------------------------
Precision Percentage for Department: 100.00%
Precision Percentage for Techgroup: 93.00%
Precision Percentage for Category: 93.00%
Precision Percentage for Subcategory: 85.00%
Precision Percentage for Priority: 94.00%
------------------------------------------------------------
Recall Percentage for Department for: 100.00%
Recall Percentage for Techgroup for: 93.00%
Recall Percentage for Category for: 93.00%
Recall Percentage for Subcategory for: 85.00%
Recall Percentage for Priority for: 94.00%
------------------------------------------------------------

```