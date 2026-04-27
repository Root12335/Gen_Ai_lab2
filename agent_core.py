import base64
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from memory import SummaryMemoryManager, trim_messages
from tools import search_healthy_options, store_meal_record


load_dotenv()

DISCLAIMER = "This is not medical or dietary advice. Consult a qualified professional."

SYSTEM_PROMPT = f"""
You are a Nutrition AI Assistant Agent.

Your responsibilities:
- Analyze meal inputs from text and optional food images.
- Estimate calories and macro nutrients with clear uncertainty language.
- Produce structured output sections:
  1) Meal Analysis
  2) Nutrition Summary
  3) Recommendations
  4) Search Results (only when user asks search-related questions)
  5) CSV Storage (only when user asks to store/save/log data)
- Dynamically decide when tools are needed.
- Provide practical and safe guidance only.

Tool policy:
- Use search_healthy_options for nearby healthy restaurants, grocery stores, or nutrition info.
- Use store_meal_record only when user requests saving/logging data.
- Never hallucinate tool results.

Safety constraints:
- Do not provide medical diagnoses.
- Do not provide strict diet plans.
- Keep recommendations general and educational.
- Always include this disclaimer exactly:
{DISCLAIMER}
"""


def build_nutrition_agent(model_name: str = "gpt-4.1"):
    return create_agent(
        model_name,
        system_prompt=SYSTEM_PROMPT,
        tools=[search_healthy_options, store_meal_record],
        checkpointer=InMemorySaver(),
        middleware=[trim_messages],
    )


def build_user_message(user_text: str, image_bytes: Optional[bytes], mime_type: str = "image/jpeg") -> HumanMessage:
    if image_bytes:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return HumanMessage(
            content=[
                {"type": "text", "text": user_text},
                {"type": "image", "base64": encoded, "mime_type": mime_type},
            ]
        )
    return HumanMessage(content=user_text)


def invoke_with_memory(
    agent,
    memory: SummaryMemoryManager,
    conversation_messages,
    user_text: str,
    image_bytes: Optional[bytes],
    thread_id: str = "nutrition-thread-1",
    mime_type: str = "image/jpeg",
):
    user_message = build_user_message(user_text, image_bytes, mime_type)
    state_messages = [*conversation_messages, user_message]
    effective_messages = memory.inject_summary(state_messages)
    result = agent.invoke(
        {"messages": effective_messages},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result

