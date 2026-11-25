import os
import random
import re
import time
from datetime import datetime
from typing import Any
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="VastraVibe AI Customer Support Agent")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = list(dict.fromkeys([PRIMARY_MODEL, "gemini-3.8-flash-lite"]))

client = genai.Client(api_key=GEMINI_API_KEY)

# --- MOCK INDIAN D2C DATABASE (IN-MEMORY FOR INSTANT ZERO-DB DEMO) ---
ORDERS_DB: dict[str, dict[str, Any]] = {
    "ORD-1042": {
        "order_id": "ORD-1042",
        "customer_name": "Ananya Verma",
        "items": "Oversized Cotton Linen Shirt (Olive, M) x 1",
        "amount_inr": 1499,
        "payment_mode": "Prepaid (UPI)",
        "status": "Out for Delivery",
        "courier": "Delhivery Express",
        "awb": "DLV99281741IN",
        "city": "Bengaluru",
        "eta": "Today by 7:00 PM",
        "delivered_days_ago": None,
    },
    "ORD-1043": {
        "order_id": "ORD-1043",
        "customer_name": "Rohan Kulkarni",
        "items": "Relaxed Fit Cargo Trousers (Charcoal, 32) x 1",
        "amount_inr": 2199,
        "payment_mode": "Prepaid (Card)",
        "status": "Delivered",
        "courier": "Blue Dart",
        "awb": "BD77412098IN",
        "city": "Mumbai",
        "eta": "Delivered 2 days ago",
        "delivered_days_ago": 2,
    },
    "ORD-1044": {
        "order_id": "ORD-1044",
        "customer_name": "Siddharth Nair",
        "items": "Everyday Tech Organizer Pouch + Bamboo Socks Pack",
        "amount_inr": 1899,
        "payment_mode": "Cash on Delivery (COD)",
        "status": "Delayed in Transit",
        "courier": "Shiprocket / XpressBees",
        "awb": "XB55102938IN",
        "city": "Jaipur",
        "eta": "Expected within 24-48 hours (hub weather delay)",
        "delivered_days_ago": None,
    },
    "ORD-1045": {
        "order_id": "ORD-1045",
        "customer_name": "Meera Iyer",
        "items": "Classic Supima Crew T-Shirt (Navy, L) x 2",
        "amount_inr": 1798,
        "payment_mode": "Prepaid (UPI)",
        "status": "Cancelled",
        "courier": "Not Dispatched",
        "awb": "N/A",
        "city": "Chennai",
        "eta": "Refund of ₹1,798 initiated to original UPI ID (3-5 business days)",
        "delivered_days_ago": None,
    },
}

TICKETS_DB: list[dict[str, Any]] = []
RETURN_REQUESTS_DB: list[dict[str, Any]] = []
CHAT_SESSIONS: dict[str, list[types.Content]] = {}
PENDING_ESCALATION: set[str] = set()

# Track tools called during the active request
_current_request_tools: list[str] = []


# --- AGENT TOOLS (CALLED AUTOMATICALLY BY GEMINI) ---
def check_order_status(order_id: str) -> dict[str, Any]:
    """Look up real-time order status, courier partner, AWB tracking number, payment mode, and ETA.

    Args:
        order_id: The order ID provided by the customer (for example, ORD-1042).
    """
    _current_request_tools.append(f"check_order_status({order_id})")
    clean_id = order_id.strip().upper()
    if not clean_id.startswith("ORD-") and clean_id.isdigit():
        clean_id = f"ORD-{clean_id}"
    order = ORDERS_DB.get(clean_id)
    if not order:
        return {
            "found": False,
            "message": f"No order found with ID '{order_id}'. Ask the customer to double-check their Order ID (e.g., ORD-1042).",
        }
    return {"found": True, **order}


def initiate_return_or_exchange(order_id: str, action: str, reason: str) -> dict[str, Any]:
    """Initiate a 7-day return or size exchange for a delivered order and schedule reverse pickup.

    Args:
        order_id: The order ID (for example, ORD-1043).
        action: Either 'Return' or 'Exchange'.
        reason: Customer's reason (e.g., size issue, wrong item, defective).
    """
    _current_request_tools.append(f"initiate_return_or_exchange({order_id}, {action})")
    clean_id = order_id.strip().upper()
    if not clean_id.startswith("ORD-") and clean_id.isdigit():
        clean_id = f"ORD-{clean_id}"
    order = ORDERS_DB.get(clean_id)
    if not order:
        return {"success": False, "message": f"Order '{order_id}' was not found."}
    if order["status"] != "Delivered":
        return {
            "success": False,
            "message": f"Order {clean_id} is currently '{order['status']}'. Returns or exchanges can only be created after delivery.",
        }
    days = order.get("delivered_days_ago") or 0
    if days > 7:
        return {
            "success": False,
            "message": f"Order {clean_id} was delivered {days} days ago, which is outside our 7-day return/exchange window.",
        }

    req_id = f"RMA-{random.randint(1000, 9999)}"
    record = {
        "request_id": req_id,
        "order_id": clean_id,
        "action": action.capitalize(),
        "reason": reason,
        "pickup_eta": "Within 24-48 hours via reverse courier",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    RETURN_REQUESTS_DB.append(record)
    return {
        "success": True,
        "request_id": req_id,
        "order_id": clean_id,
        "action": record["action"],
        "pickup_eta": record["pickup_eta"],
        "refund_note": (
            "Full refund to original payment mode within 3-5 business days after pickup QC."
            if record["action"] == "Return"
            else "Free replacement will ship immediately once pickup is scanned."
        ),
    }


def create_support_ticket(
    customer_name: str,
    phone: str,
    issue_summary: str,
    priority: str = "Normal",
) -> dict[str, Any]:
    """Create a human support escalation ticket when a customer asks for a human agent or has an unresolved issue.

    Args:
        customer_name: Full name of the customer.
        phone: 10-digit Indian mobile number or WhatsApp number.
        issue_summary: Clear summary of the issue and any relevant order ID.
        priority: Ticket priority ('Normal' or 'High').
    """
    _current_request_tools.append(f"create_support_ticket({customer_name})")
    ticket_id = f"TKT-{random.randint(1000, 9999)}"
    ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name.strip(),
        "phone": phone.strip(),
        "issue_summary": issue_summary.strip(),
        "priority": priority,
        "status": "Open",
        "sla": "Human specialist callback within 2 hours (10 AM - 7 PM IST)",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    TICKETS_DB.insert(0, ticket)
    return {"created": True, **ticket}


def backup_support_engine(session_id: str, user_msg: str) -> str:
    """Zero-downtime backup engine when Gemini cloud servers experience 503 high-demand spikes."""
    text = user_msg.strip()
    lower = text.lower()

    # 1. Human escalation flow
    phone_match = re.search(r"\b([6-9]\d{9})\b", re.sub(r"[\s\-]", "", text))
    if session_id in PENDING_ESCALATION or any(k in lower for k in ("human", "agent", "escalate", "manager", "call me")):
        if phone_match:
            phone = phone_match.group(1)
            cleaned_name = re.sub(r"[\d\+\-\(\),]", " ", text)
            cleaned_name = re.sub(r"\b(my name is|name|phone|number|mobile|call|please|hi|hello)\b", "", cleaned_name, flags=re.I).strip()
            customer_name = cleaned_name.title() if len(cleaned_name) >= 2 else "Valued Customer"
            PENDING_ESCALATION.discard(session_id)
            tkt = create_support_ticket(
                customer_name=customer_name,
                phone=phone,
                issue_summary="Customer requested human specialist callback via web support chat.",
                priority="High",
            )
            return (
                f"I've created a priority support ticket for you:\n"
                f"• Ticket ID: {tkt['ticket_id']}\n"
                f"• Name: {tkt['customer_name']}\n"
                f"• Mobile: {tkt['phone']}\n"
                f"• Callback SLA: {tkt['sla']}"
            )
        PENDING_ESCALATION.add(session_id)
        return "I'd be happy to connect you with a human support specialist. Please reply with your Full Name and 10-digit Mobile Number (e.g., Aarav Mehta, 9876543210)."

    # 2. Order lookup or return/exchange flow
    ord_match = re.search(r"\b(?:ORD-)?(10\d{2}|\d{4})\b", text, flags=re.I)
    if ord_match:
        order_id = f"ORD-{ord_match.group(1)}"
        if any(k in lower for k in ("return", "exchange", "replace", "size", "tight", "loose", "damaged", "defective")):
            action = "Return" if "return" in lower or "refund" in lower else "Exchange"
            res = initiate_return_or_exchange(order_id=order_id, action=action, reason=text)
            if not res.get("success"):
                return res["message"]
            return (
                f"Your {res['action'].lower()} request has been scheduled:\n"
                f"• Request ID: {res['request_id']}\n"
                f"• Order ID: {res['order_id']}\n"
                f"• Reverse Pickup: {res['pickup_eta']}\n"
                f"• Note: {res['refund_note']}"
            )

        order = check_order_status(order_id)
        if not order.get("found"):
            return order["message"]
        return (
            f"Here are the live details for {order['order_id']}:\n"
            f"• Item: {order['items']}\n"
            f"• Status: {order['status']}\n"
            f"• Courier & AWB: {order['courier']} ({order['awb']})\n"
            f"• Payment: ₹{order['amount_inr']} ({order['payment_mode']})\n"
            f"• Delivery / ETA: {order['eta']}"
        )

    # 3. Store Policy FAQs
    if any(k in lower for k in ("cod", "cash", "refund", "return", "exchange", "ship", "delivery", "policy")):
        return (
            "Here are VastraVibe India's store policies:\n"
            "• Shipping: Free on all Prepaid orders (2–4 business days in Metros, 4–6 days Rest of India).\n"
            "• Cash on Delivery (COD): Available up to ₹4,999 with a ₹49 COD handling fee.\n"
            "• Returns & Exchanges: 7-day hassle-free reverse pickup for unused items with tags intact; 1 size exchange is free.\n"
            "• Refunds: Prepaid refunds credit in 3–5 business days after pickup QC; COD refunds credit via instant UPI link within 48 hours."
        )

    return (
        "I can help you right away with:\n"
        "• Order Tracking (share your Order ID like ORD-1042 or ORD-1044)\n"
        "• 7-Day Returns & Size Exchanges (e.g., Exchange ORD-1043)\n"
        "• COD, Shipping & Refund Policies\n"
        "• Human Agent Callback (type Escalate to human)"
    )


SYSTEM_INSTRUCTION = """
You are the AI Customer Support Specialist for **VastraVibe India**, an Indian D2C fashion and everyday essentials brand.

TONE & STYLE:
- Warm, concise, polite, and solution-oriented. Use bullet points for multi-part details so answers read cleanly on mobile screens.
- Currency is always Indian Rupees (₹).

STORE POLICIES (STRICT SOURCE OF TRUTH):
1. Shipping: Free shipping on all Prepaid orders across India. Delivery takes 2–4 business days in Metro cities (Bengaluru, Mumbai, Delhi NCR, Hyderabad, Chennai, Pune) and 4–6 business days in Rest of India via Delhivery, Blue Dart, or XpressBees.
2. Cash on Delivery (COD): Available on orders up to ₹4,999 with a ₹49 COD handling fee. COD orders require quick WhatsApp/IVR confirmation before dispatch.
3. Returns & Exchanges: 7-day hassle-free reverse pickup from the delivery date for unused items with original tags and packaging intact. One size exchange per order is completely free.
4. Refunds:
   - Prepaid (UPI / Cards / NetBanking): Credited to original payment source within 3–5 business days after reverse pickup quality check.
   - COD Orders: Credited via instant UPI payout link within 48 hours of pickup verification.
5. Support Hours: Live human specialists are available Monday–Saturday, 10:00 AM to 7:00 PM IST.

TOOL USAGE RULES:
- Whenever a customer mentions an Order ID or asks where their order is, ALWAYS call `check_order_status`.
- Whenever a customer asks to return or exchange a delivered item and provides an Order ID + reason, call `initiate_return_or_exchange`.
- Whenever a customer is upset, has a delayed refund/shipment issue needing human intervention, or explicitly asks to talk to a human agent:
  1. Ask for their **Name** and **10-digit Mobile Number** if they haven't shared them yet.
  2. Once you have both, call `create_support_ticket` with `priority="High"` if they mention a delay or refund issue.
- Never invent order details, tracking numbers, or discounts not present in the tools or policies above.
"""


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/health")
async def health():
    return {"status": "running", "models": FALLBACK_MODELS, "tickets_count": len(TICKETS_DB)}


@app.get("/api/tickets")
async def list_tickets():
    return {"tickets": TICKETS_DB, "returns": RETURN_REQUESTS_DB}


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    global _current_request_tools

    if not GEMINI_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"reply": "Server configuration error: GEMINI_API_KEY is missing in environment variables.", "tools": []},
        )

    history = CHAT_SESSIONS.setdefault(req.session_id, [])
    user_turn = types.Content(role="user", parts=[types.Part.from_text(text=req.message)])
    conversation = history[-10:] + [user_turn]

    last_error = ""
    for model_name in FALLBACK_MODELS:
        for attempt in range(2):
            _current_request_tools = []
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=conversation,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        tools=[check_order_status, initiate_return_or_exchange, create_support_ticket],
                        temperature=0.2,
                    ),
                )
                reply_text = response.text or "I'm sorry, I couldn't process that request. Please try again."
                history.append(user_turn)
                history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply_text)]))
                if len(history) > 12:
                    CHAT_SESSIONS[req.session_id] = history[-12:]
                return {
                    "reply": reply_text,
                    "tools": list(_current_request_tools),
                    "ticket_count": len(TICKETS_DB),
                }
            except Exception as e:
                last_error = str(e)
                is_transient = any(code in last_error for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"))
                if is_transient and attempt == 0:
                    time.sleep(1.5)
                    continue
                break

    # Seamless zero-downtime fallback if Gemini cloud models are experiencing 503 high demand
    if any(code in last_error for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND")):
        _current_request_tools = []
        fallback_reply = backup_support_engine(req.session_id, req.message)
        history.append(user_turn)
        history.append(types.Content(role="model", parts=[types.Part.from_text(text=fallback_reply)]))
        return {
            "reply": fallback_reply,
            "tools": list(_current_request_tools),
            "ticket_count": len(TICKETS_DB),
        }

    return JSONResponse(
        status_code=500,
        content={"reply": f"Assistant error: {last_error}", "tools": []},
    )


@app.get("/widget.js")
async def embeddable_widget(request: Request):
    """Drop-in floating chat widget script for any client website."""
    base_url = str(request.base_url).rstrip("/")
    js_code = f"""
(function() {{
  const btn = document.createElement('button');
  btn.innerText = '💬 Support';
  Object.assign(btn.style, {{
    position: 'fixed', bottom: '20px', right: '20px', zIndex: '9999',
    background: '#111827', color: '#fff', border: 'none', borderRadius: '999px',
    padding: '12px 18px', fontSize: '14px', fontWeight: '600', cursor: 'pointer',
    boxShadow: '0 8px 24px rgba(0,0,0,0.2)'
  }});
  const frame = document.createElement('iframe');
  frame.src = '{base_url}/';
  Object.assign(frame.style, {{
    position: 'fixed', bottom: '76px', right: '20px', zIndex: '9999',
    width: '370px', maxWidth: '92vw', height: '560px', maxHeight: '78vh',
    border: '1px solid #e5e7eb', borderRadius: '16px', display: 'none',
    boxShadow: '0 16px 40px rgba(0,0,0,0.22)', background: '#fff'
  }});
  btn.onclick = () => {{
    frame.style.display = frame.style.display === 'none' ? 'block' : 'none';
  }};
  document.body.appendChild(btn);
  document.body.appendChild(frame);
}})();
"""
    return Response(content=js_code, media_type="application/javascript")


@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0" />
  <title>VastraVibe AI Customer Support</title>
  <style>
    :root {
      --bg: #f3f4f6;
      --card: #ffffff;
      --primary: #111827;
      --accent: #2563eb;
      --bot-bg: #f8fafc;
      --border: #e5e7eb;
      --text: #111827;
      --muted: #6b7280;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); display: flex; justify-content: center; height: 100dvh; }
    .app { width: 100%; max-width: 540px; background: var(--card); display: flex; flex-direction: column; height: 100dvh; box-shadow: 0 0 30px rgba(0,0,0,0.06); }
    .header { background: var(--primary); color: #fff; padding: 14px 16px; display: flex; justify-content: space-between; align-items: center; }
    .header-left { display: flex; align-items: center; gap: 10px; }
    .avatar { width: 36px; height: 36px; border-radius: 50%; background: var(--accent); display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 15px; }
    .title { font-size: 15px; font-weight: 600; }
    .subtitle { font-size: 11px; color: #9ca3af; display: flex; align-items: center; gap: 5px; margin-top: 2px; }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: #22c55e; display: inline-block; }
    .ticket-btn { background: rgba(255,255,255,0.12); color: #fff; border: 1px solid rgba(255,255,255,0.2); border-radius: 8px; padding: 6px 10px; font-size: 12px; cursor: pointer; }
    .messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
    .msg { max-width: 85%; padding: 11px 14px; border-radius: 14px; font-size: 14px; line-height: 1.48; white-space: pre-wrap; word-wrap: break-word; }
    .msg.bot { align-self: flex-start; background: var(--bot-bg); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
    .msg.user { align-self: flex-end; background: var(--accent); color: #fff; border-bottom-right-radius: 4px; }
    .tool-badge { display: inline-block; font-size: 11px; font-weight: 600; color: #1d4ed8; background: #dbeafe; padding: 3px 8px; border-radius: 6px; margin-bottom: 6px; }
    .chips { display: flex; gap: 6px; overflow-x: auto; padding: 8px 12px; border-top: 1px solid var(--border); background: #fafafa; scrollbar-width: none; }
    .chips::-webkit-scrollbar { display: none; }
    .chip { flex-shrink: 0; background: #fff; border: 1px solid #d1d5db; border-radius: 999px; padding: 6px 11px; font-size: 12px; color: #374151; cursor: pointer; }
    .chip:active { background: #e5e7eb; }
    .input-bar { display: flex; gap: 8px; padding: 10px 12px; border-top: 1px solid var(--border); background: #fff; }
    .input-bar input { flex: 1; border: 1px solid #d1d5db; border-radius: 10px; padding: 10px 12px; font-size: 14px; outline: none; }
    .input-bar input:focus { border-color: var(--accent); }
    .input-bar button { background: var(--primary); color: #fff; border: none; border-radius: 10px; padding: 0 16px; font-size: 14px; font-weight: 600; cursor: pointer; }
    .input-bar button:disabled { opacity: 0.5; }
    .modal { position: fixed; inset: 0; background: rgba(0,0,0,0.45); display: none; align-items: center; justify-content: center; padding: 16px; z-index: 50; }
    .modal-card { background: #fff; width: 100%; max-width: 460px; max-height: 80vh; border-radius: 14px; padding: 16px; overflow-y: auto; }
    .modal-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .ticket-item { border: 1px solid var(--border); border-radius: 10px; padding: 10px; margin-bottom: 8px; font-size: 13px; }
  </style>
</head>
<body>
  <div class="app">
    <div class="header">
      <div class="header-left">
        <div class="avatar">VV</div>
        <div>
          <div class="title">VastraVibe Support AI</div>
          <div class="subtitle"><span class="dot"></span> Online &bull; Order Tracking, Returns &amp; Tickets</div>
        </div>
      </div>
      <button class="ticket-btn" onclick="openTickets()">Tickets (<span id="tCount">0</span>)</button>
    </div>

    <div class="messages" id="chatBox">
      <div class="msg bot">Hi! I'm the VastraVibe AI Support Assistant. I can help you:
• Track live orders (try <b>ORD-1042</b> or <b>ORD-1044</b>)
• Start a 7-day return or size exchange (try <b>ORD-1043</b>)
• Answer COD, shipping &amp; refund questions
• Escalate urgent issues to a human specialist

How can I help you today?</div>
    </div>

    <div class="chips">
      <button class="chip" onclick="quickSend('Where is my order ORD-1042?')">Track ORD-1042</button>
      <button class="chip" onclick="quickSend('I want to exchange size for order ORD-1043 because M is tight')">Exchange ORD-1043</button>
      <button class="chip" onclick="quickSend('Check status of ORD-1044')">Delayed COD ORD-1044</button>
      <button class="chip" onclick="quickSend('What is your COD and refund policy?')">COD &amp; Refunds</button>
      <button class="chip" onclick="quickSend('Connect me to a human support agent')">Escalate to Human</button>
    </div>

    <form class="input-bar" onsubmit="sendMsg(event)">
      <input id="msgInput" type="text" placeholder="Ask about orders, returns, or policies..." autocomplete="off" required />
      <button id="sendBtn" type="submit">Send</button>
    </form>
  </div>

  <div class="modal" id="ticketModal" onclick="if(event.target===this)this.style.display='none'">
    <div class="modal-card">
      <div class="modal-head">
        <strong>Escalated Support Tickets &amp; RMAs</strong>
        <button onclick="document.getElementById('ticketModal').style.display='none'" style="border:none;background:none;font-size:18px;cursor:pointer;">✕</button>
      </div>
      <div id="ticketList">Loading...</div>
    </div>
  </div>

  <script>
    const sessionId = 'sess_' + Math.random().toString(36).slice(2, 10);
    const chatBox = document.getElementById('chatBox');
    const msgInput = document.getElementById('msgInput');
    const sendBtn = document.getElementById('sendBtn');

    function appendMsg(role, text, tools = []) {
      const div = document.createElement('div');
      div.className = 'msg ' + role;
      if (tools && tools.length > 0) {
        const badge = document.createElement('div');
        badge.className = 'tool-badge';
        badge.textContent = '⚡ Action: ' + tools.join(', ');
        div.appendChild(badge);
        div.appendChild(document.createElement('br'));
      }
      const span = document.createElement('span');
      span.textContent = text;
      div.appendChild(span);
      chatBox.appendChild(div);
      chatBox.scrollTop = chatBox.scrollHeight;
      return div;
    }

    function quickSend(text) {
      msgInput.value = text;
      sendMsg(new Event('submit'));
    }

    async function sendMsg(e) {
      e.preventDefault();
      const text = msgInput.value.trim();
      if (!text) return;
      msgInput.value = '';
      appendMsg('user', text);
      sendBtn.disabled = true;
      const typing = appendMsg('bot', 'Checking...');

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, message: text })
        });
        const data = await res.json();
        typing.remove();
        appendMsg('bot', data.reply, data.tools || []);
        if (typeof data.ticket_count === 'number') {
          document.getElementById('tCount').textContent = data.ticket_count;
        }
      } catch (err) {
        typing.textContent = 'Network error. Please try again.';
      } finally {
        sendBtn.disabled = false;
      }
    }

    async function openTickets() {
      const modal = document.getElementById('ticketModal');
      const list = document.getElementById('ticketList');
      modal.style.display = 'flex';
      const res = await fetch('/api/tickets');
      const data = await res.json();
      let html = '';
      if ((!data.tickets || data.tickets.length === 0) && (!data.returns || data.returns.length === 0)) {
        html = '<p style="color:#6b7280;font-size:13px;">No tickets or return requests created yet. Ask the bot to escalate an issue or return ORD-1043!</p>';
      } else {
        (data.tickets || []).forEach(t => {
          html += `<div class="ticket-item"><b>${t.ticket_id}</b> [${t.priority}] — ${t.status}<br><b>Customer:</b> ${t.customer_name} (${t.phone})<br><b>Issue:</b> ${t.issue_summary}<br><small style="color:#6b7280;">${t.created_at}</small></div>`;
        });
        (data.returns || []).forEach(r => {
          html += `<div class="ticket-item"><b>${r.request_id}</b> (${r.action} for ${r.order_id})<br><b>Reason:</b> ${r.reason}<br><small style="color:#6b7280;">${r.pickup_eta}</small></div>`;
        });
      }
      list.innerHTML = html;
    }
  </script>
</body>
</html>"""
