from __future__ import annotations

from typing import List

from langchain.agents import AgentState
from langchain.agents.middleware import before_agent
from langchain.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI


@before_agent
def trim_messages(state: AgentState, runtime: Runtime) -> AgentState:
    """Remove tool traces and empty responses to reduce token usage."""
    to_remove = [
        msg
        for msg in state["messages"]
        if isinstance(msg, ToolMessage) or (hasattr(msg, "content") and msg.content == "")
    ]
    return {"messages": [RemoveMessage(msg.id) for msg in to_remove]}


class SummaryMemoryManager:
    """Rolling summary memory used before invoking the agent."""

    def __init__(self, model_name: str = "gpt-4.1-mini", max_messages: int = 10) -> None:
        self.model = ChatOpenAI(model=model_name, temperature=0)
        self.max_messages = max_messages
        self.summary = ""

    def _format_messages(self, messages: List[BaseMessage]) -> str:
        parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "User"
            elif isinstance(msg, AIMessage):
                role = "Assistant"
            else:
                role = "Other"
            parts.append(f"{role}: {str(msg.content)}")
        return "\n".join(parts)

    def maybe_update_summary(self, messages: List[BaseMessage]) -> str:
        if len(messages) <= self.max_messages:
            return self.summary

        older_slice = messages[:-self.max_messages]
        prompt = (
            "Summarize the following conversation briefly for memory.\n"
            "Keep user preferences, goals, dietary context, and open tasks.\n"
            "Do not include sensitive assumptions.\n\n"
            f"Current summary:\n{self.summary or '(none)'}\n\n"
            f"Conversation chunk:\n{self._format_messages(older_slice)}"
        )
        self.summary = self.model.invoke(prompt).content
        return self.summary

    def inject_summary(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        summary = self.maybe_update_summary(messages)
        if not summary:
            return messages[-self.max_messages :]

        memory_message = HumanMessage(
            content=(
                "Conversation memory summary (for context only, do not repeat verbatim):\n"
                f"{summary}"
            )
        )
        return [memory_message, *messages[-self.max_messages :]]

