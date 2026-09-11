from datetime import datetime
import uuid


class TicketManager:

    def __init__(self):
        self.tickets = []

    def create_ticket(
        self,
        message,
        intent,
        confidence,
        reason,
        priority="High"
    ):
        ticket_id = f"HVR-{uuid.uuid4().hex[:6].upper()}"

        ticket = {
            "ticket_id": ticket_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": message,
            "intent": intent,
            "confidence": round(confidence * 100, 2),
            "priority": priority,
            "reason": reason,
            "status": "Open",
            "assigned_to": "Human Support Team",
        }

        self.tickets.append(ticket)

        return ticket

    def get_tickets(self):
        return self.tickets

    def get_ticket(self, ticket_id):
        for ticket in self.tickets:
            if ticket["ticket_id"] == ticket_id:
                return ticket

        return None

    def update_status(self, ticket_id, status):
        ticket = self.get_ticket(ticket_id)

        if ticket:
            ticket["status"] = status
            return ticket

        return None