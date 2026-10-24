import os
from datetime import datetime, timezone
from uuid import uuid4

from openai import OpenAI
from uagents import Agent, Context, Protocol

from uagents.experimental.chat_agent.protocol import (
    build_llm_message_history,
)

from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    EndSessionContent,
    chat_protocol_spec,
)

from uagents_core.contrib.protocols.payment import (
    RequestPayment,
    CommitPayment,
    RejectPayment,
    CancelPayment,
    CompletePayment,
    payment_protocol_spec,
)


# ==========================================
# 1. ASI:ONE CONFIGURATION
# ==========================================

ASI_BASE_URL = os.getenv(
    "ASI1_BASE_URL",
    "https://api.asi1.ai/v1",
)

ASI_API_KEY = os.getenv("ASI1_API_KEY", "")

ASI_MODEL = os.getenv("ASI_MODEL", "asi1")

client = OpenAI(
    base_url=ASI_BASE_URL,
    api_key=ASI_API_KEY,
    timeout=60.0,
)


# ==========================================
# 2. AGENTOS AI INSTRUCTIONS
# ==========================================

SYSTEM_PROMPT = """
You are AgentOS, an all-in-one AI assistant.

Your capabilities include:

1. STUDY AGENT
Explain subjects, create notes, quizzes and revision plans.

2. CODING AGENT
Help with Python, C, C++, Java, HTML, debugging and projects.

3. TRAVEL AGENT
Help plan journeys and explain transport options.
Do not claim to have checked live routes without real data.

4. GROCERY AGENT
Create shopping lists and help organize a budget.

5. GENERAL AGENT
Help with writing, productivity, brainstorming and questions.

Instructions:
- Understand the user's request and answer helpfully.
- Identify the relevant category in your response.
- Use simple, clear language.
- Never invent live data or claim external actions were completed.
- Never claim a payment was made or verified without evidence.
- Do not expose API keys, private seeds or secrets.
- You are one AI with multiple capabilities; do not claim that
  independent specialist agents exist until they are connected.

Start your answer with the relevant category, such as:
[Study], [Coding], [Travel], [Grocery], or [General].
"""


# ==========================================
# 3. MAIN HOSTED AGENT
# ==========================================

agent = Agent()

chat_protocol = Protocol(
    spec=chat_protocol_spec,
)

payment_protocol = Protocol(
    spec=payment_protocol_spec,
    role="seller",
)


# ==========================================
# 4. CHAT MESSAGE HELPER
# ==========================================

def create_chat_message(
    text: str,
    end_session: bool = False,
) -> ChatMessage:

    content = [
        TextContent(
            type="text",
            text=text,
        )
    ]

    if end_session:
        content.append(
            EndSessionContent(type="end-session")
        )

    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=content,
    )


# ==========================================
# 5. CHAT HANDLER
# ==========================================

@chat_protocol.on_message(ChatMessage)
async def handle_chat(
    ctx: Context,
    sender: str,
    msg: ChatMessage,
):

    # Acknowledge the incoming chat message.
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id,
        ),
    )

    user_text = msg.text()

    if not user_text:
        return

    ctx.logger.info(
        f"AgentOS received a chat message from {sender}"
    )

    # Build context from the current chat session.
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *build_llm_message_history(ctx),
    ]

    try:
        response = client.chat.completions.create(
            model=ASI_MODEL,
            messages=messages,
            max_tokens=2048,
        )

        answer = (
            response.choices[0].message.content or ""
        ).strip()

        if not answer:
            answer = "I couldn't generate a response. Please try again."

    except Exception:
        ctx.logger.exception(
            "ASI:One request failed"
        )

        answer = (
            "Sorry, AgentOS couldn't process that request. "
            "Please check the ASI:One configuration and try again."
        )

    await ctx.send(
        sender,
        create_chat_message(answer),
    )


# ==========================================
# 6. CHAT ACKNOWLEDGEMENT HANDLER
# ==========================================

@chat_protocol.on_message(ChatAcknowledgement)
async def handle_chat_ack(
    ctx: Context,
    sender: str,
    msg: ChatAcknowledgement,
):
    # Acknowledgements confirm receipt of chat messages.
    pass


# ==========================================
# 7. PAYMENT PROTOCOL: SELLER HANDLERS
# ==========================================

@payment_protocol.on_message(CommitPayment)
async def handle_payment_commit(
    ctx: Context,
    sender: str,
    msg: CommitPayment,
):

    ctx.logger.info(
        f"Payment commitment received: {msg.transaction_id}"
    )

    # A commitment message does not prove funds were transferred.
    # No real payment verification is configured in this version.
    # Cancel safely rather than falsely confirming payment.

    await ctx.send(
        sender,
        CancelPayment(
            transaction_id=msg.transaction_id,
            reason=(
                "AgentOS is in demo mode. "
                "No payment verification is configured."
            ),
        ),
    )


@payment_protocol.on_message(RejectPayment)
async def handle_payment_rejection(
    ctx: Context,
    sender: str,
    msg: RejectPayment,
):

    ctx.logger.info(
        f"Payment request rejected: {msg.reason}"
    )


# ==========================================
# 8. REGISTER BOTH PROTOCOLS
# ==========================================

agent.include(
    chat_protocol,
    publish_manifest=True,
)

agent.include(
    payment_protocol,
    publish_manifest=True,
)


# ==========================================
# 9. START AGENT
# ==========================================

if __name__ == "__main__":
    agent.run()
