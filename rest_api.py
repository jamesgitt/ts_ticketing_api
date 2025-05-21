# =========================
# Imports
# =========================
import os  # For environment variable operations
import csv  # For reading and writing CSV files
import hmac  # For secure API key comparison
import secrets  # For secure random operations

from fastapi import FastAPI, Depends, HTTPException, Header, Form  # FastAPI for API creation and dependencies
from pydantic import BaseModel, Field  # Pydantic for data validation and serialization
from typing import List  # For type hinting lists
from dotenv import load_dotenv  # To load environment variables from a .env file
from typing import Optional
import threading  # For thread-safe ticket ID assignment

from ticket_tagger import get_ticket_tags  # Function for tagging tickets using the LLM model

# =========================
# Environment and Globals
# =========================

# Load environment variables from .env file (e.g., API_KEY)
load_dotenv()

API_KEY = os.getenv("API_KEY")  # API key for authentication

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
            f'"{ticket.get("subject")}"' if ticket.get("subject") is not None else '""',
            f'"{ticket.get("description")}"' if ticket.get("description") is not None else '""',
            ticket.get("email"),
            ticket.get("department"),
            ticket.get("techgroup"),
            ticket.get("category"),
            ticket.get("subcategory"),
            ticket.get("priority"),
        ])

def read_tickets_from_csv():
    tickets = []
    with open("tickets_log.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Defensive: skip rows missing any required field
            if not row.get("id"):
                continue
            try:
                tickets.append({
                    "id": int(row["id"]),
                    "subject": row.get("subject", ""),
                    "description": row.get("description", ""),
                    "email": row.get("email", ""),
                    "department": row.get("department", ""),
                    "techgroup": row.get("techgroup", ""),
                    "category": row.get("category", ""),
                    "subcategory": row.get("subcategory", ""),
                    "priority": row.get("priority", ""),
                })
            except Exception as e:
                print("Skipping malformed row:", row, "Error:", e)
    return tickets

def get_next_ticket_id():
    tickets = read_tickets_from_csv()
    if not tickets:
        return 1
    return max(ticket["id"] for ticket in tickets) + 1

# =========================
# Pydantic Models
# =========================

# Model for incoming ticket data (request body)
class Ticket(BaseModel):
    subject: str = Field(..., description="The subject of the ticket")
    description: str = Field(..., description="A detailed description of the issue")
    email: str = Field(..., description="The email address of the requester")

# Model for outgoing ticket data (response body, includes tags and ID)
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

# Dependency to verify API key.
# Raises 401 error if invalid.
def verify_api_key(x_api_key: str = Header(None, convert_underscores=False)):
    if API_KEY is None:
        raise HTTPException(status_code=500, detail="API key not configured")
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key missing")
    # Use secrets.compare_digest for constant-time comparison
    if not secrets.compare_digest(str(x_api_key).strip(), API_KEY):
        raise HTTPException(status_code=401, detail="API key invalid")
    return x_api_key

# =========================
# API Endpoints
# =========================

# Endpoint to create a new ticket.
# Requires API key, tags ticket using the LLM model, and returns the ticket with tags.
@app.post("/tickets", response_model=TicketOut)
def create_ticket(
    subject: str = Form(..., description="Ticket subject"),
    description: str = Form(..., description="Ticket description"),
    email: str = Form(..., description="Ticket email"),
    api_key: str = Depends(verify_api_key)
):
    ticket = Ticket(subject=subject, description=description, email=email)

    # Get next ticket ID based on CSV
    ticket_id = get_next_ticket_id()

    tags_dict = get_ticket_tags(ticket.subject, ticket.description, ticket.email)  # Tag ticket using the LLM model
    new_ticket = {
        "id": ticket_id,
        "subject": ticket.subject,
        "description": ticket.description,
        "email": ticket.email,
        **tags_dict
    }
    append_ticket_to_csv(new_ticket)  # Store the ticket in the CSV file
    return new_ticket  # Return the created ticket