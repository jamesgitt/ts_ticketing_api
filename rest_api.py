# =========================
# Imports
# =========================
import os  # OS operations (env vars)
import csv  # For reading/writing to CSV

from fastapi import FastAPI, Depends, HTTPException, Header, Form  # FastAPI imports for API creation and dependencies
from pydantic import BaseModel, Field  # Pydantic for data validation and serialization
from typing import List  # For type hinting lists
from dotenv import load_dotenv  # To load environment variables from .env file
from typing import Optional

from ticket_tagger import get_ticket_tags  # LLM-based function for ticket tagging using only the model (no RAG, no vector DB)

# =========================
# Environment and Globals
# =========================

# Load environment variables from .env file (e.g., API_KEY)
load_dotenv()

# No API key credits limit (unlimited credits)
API_KEY = os.getenv("API_KEY")

# Create FastAPI app instance
app = FastAPI()

# CSV file path for storing tickets
CSV_FILE = "tickets_log.csv"

# CSV headers
CSV_HEADERS = [
    "id", "subject", "description", "email",
    "department", "techgroup", "category", "subcategory", "priority"
]

# Ensure the CSV file exists and has headers
def ensure_csv_headers():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)

ensure_csv_headers()

def append_ticket_to_csv(ticket: dict):
    with open(CSV_FILE, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            ticket.get("id"),
            ticket.get("subject"),
            ticket.get("description"),
            ticket.get("email"),
            ticket.get("department"),
            ticket.get("techgroup"),
            ticket.get("category"),
            ticket.get("subcategory"),
            ticket.get("priority"),
        ])

def read_tickets_from_csv():
    tickets = []
    if not os.path.exists(CSV_FILE):
        return tickets
    with open(CSV_FILE, mode="r", newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert id to int, keep others as str or None
            ticket = {
                "id": int(row["id"]),
                "subject": row["subject"],
                "description": row["description"],
                "email": row["email"],
                "department": row["department"] if row["department"] else None,
                "techgroup": row["techgroup"] if row["techgroup"] else None,
                "category": row["category"] if row["category"] else None,
                "subcategory": row["subcategory"] if row["subcategory"] else None,
                "priority": row["priority"] if row["priority"] else None,
            }
            tickets.append(ticket)
    return tickets

def get_next_ticket_id():
    tickets = read_tickets_from_csv()
    if not tickets:
        return 1
    return max(ticket["id"] for ticket in tickets) + 1

# =========================
# Pydantic Models
# =========================

# Pydantic model for incoming ticket data (request body)
class Ticket(BaseModel):
    subject: str = Field(..., description="The subject of the ticket")
    description: str = Field(..., description="A detailed description of the issue")
    email: str = Field(..., description="The email address of the requester")

# Pydantic model for outgoing ticket data (response body, includes tags and ID)
class TicketOut(Ticket):
    id: int
    department: Optional[str]
    techgroup: Optional[str]
    category: Optional[str]
    subcategory: Optional[str]
    priority: Optional[str]

# =========================
# API Key Verification
# =========================

# API key verification dependency
# Checks if the provided API key exists (no credit check, unlimited usage)
# Raises 401 error if invalid
def verify_api_key(x_api_key: str = Header(None)):
    if API_KEY is not None and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key invalid")
    return x_api_key

# =========================
# API Endpoints
# =========================

# Endpoint to create a new ticket
# Requires API key, tags ticket using only the LLM model, and returns the ticket with tags
@app.post("/tickets", response_model=TicketOut)
def create_ticket(
    subject: str = Form(..., description="Ticket subject"),
    description: str = Form(..., description="Ticket description"),
    email: str = Form(..., description="Ticket email"),
    x_api_key: str = Depends(verify_api_key)
):
    ticket = Ticket(subject=subject, description=description, email=email)
    # No decrement of API key credits (unlimited)

    # Get next ticket ID based on CSV
    ticket_id = get_next_ticket_id()

    tags_dict = get_ticket_tags(ticket.subject, ticket.description, ticket.email)  # Tag ticket using only the LLM model
    new_ticket = {
        "id": ticket_id,
        "subject": ticket.subject,
        "description": ticket.description,
        "email": ticket.email,
        **tags_dict
    }
    append_ticket_to_csv(new_ticket)  # Store the ticket in the CSV file
    return new_ticket  # Return the created ticket

# Endpoint to list all tickets
@app.get("/tickets", response_model=List[TicketOut])
def list_tickets():
    return read_tickets_from_csv()  # Return all tickets from CSV

# Endpoint to get a specific ticket by its ID
@app.get("/tickets/{ticket_id}", response_model=TicketOut)
def get_ticket(ticket_id: int):
    tickets = read_tickets_from_csv()
    for ticket in tickets:
        if ticket["id"] == ticket_id:
            return ticket  # Return the ticket if found
    raise HTTPException(status_code=404, detail="Ticket not found")  # 404 if not found

# Endpoint to "delete" a ticket by its ID (removal only from in-memory, not from CSV)
@app.delete("/tickets/{ticket_id}", response_model=dict)
def delete_ticket(ticket_id: int):
    # Since the CSV is append-only, we cannot actually delete from it.
    # We can only acknowledge the request.
    tickets = read_tickets_from_csv()
    for ticket in tickets:
        if ticket["id"] == ticket_id:
            # Deletion is not reflected in the CSV.
            return {"message": "Ticket deleted (CSV is append-only, so ticket remains in log)"}
    raise HTTPException(status_code=404, detail="Ticket not found")  # 404 if not found 