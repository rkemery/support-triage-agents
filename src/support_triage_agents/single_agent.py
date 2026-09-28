"""Arm A: one agent with every tool, the same model, docs and step budget as the graph.

    agent <-> tools -> END

It reads the account, searches the same snapshot, may ask the customer one
question and writes to the bank directly. No reviewer and no approval step.
Written as a two-node StateGraph (no prebuilt agent), checkpointed like the
graph arms.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from support_triage_agents.data import Task
from support_triage_agents.graph import Runtime, guarded
from support_triage_agents.llm import Usage, dump, merge_usage
from support_triage_agents.prompts import render
from support_triage_agents.schemas import AgentStep, FinalReply, ToolCall
from support_triage_agents.tools import ASK_TOOL, MAX_ASKS, ROLE_TOOLS, render_result


class AgentState(TypedDict, total=False):
    task_id: str
    customer_id: str
    ticket_text: str
    profile: dict[str, Any]
    turns: list[dict[str, str]]
    pending: dict[str, Any] | None
    asks: int
    reply: str | None
    outcome: str
    failure: str | None
    usage: Annotated[dict[str, Any], merge_usage]


def initial_agent_state(task: Task) -> AgentState:
    return {
        "task_id": task.task_id,
        "customer_id": task.customer_id,
        "ticket_text": task.ticket_text,
        "turns": [],
        "pending": None,
        "asks": 0,
        "usage": {},
    }


def build_single_agent(rt: Runtime) -> StateGraph:
    def agent(state: AgentState, usage: Usage) -> dict[str, Any]:
        profile = state.get("profile") or rt.bank.get_customer_profile()
        opening = (
            f"Customer profile: {dump(profile)}\n\n"
            f"Ticket from customer {state['customer_id']}:\n{state['ticket_text']}"
        )
        messages = [{"role": "user", "content": opening}, *state.get("turns", [])]
        step = rt.llm(state).json_call(  # type: ignore[arg-type]
            "single_agent", render("single_agent", rt.facts, rt.as_of), messages, AgentStep, usage
        )
        if isinstance(step, FinalReply):
            return {"profile": profile, "reply": step.reply_to_customer, "outcome": "replied"}
        assert isinstance(step, ToolCall)
        return {"profile": profile, "pending": step.model_dump(mode="json")}

    def tools(state: AgentState) -> dict[str, Any]:
        call = ToolCall(**state["pending"])
        asks = state.get("asks", 0)
        if call.tool not in ROLE_TOOLS["single_agent"]:
            result: dict[str, Any] = {"ok": False, "error": f"unknown tool {call.tool}"}
        elif call.tool == ASK_TOOL and asks >= MAX_ASKS:
            result = {"ok": False, "error": "you already asked the customer a question"}
        else:
            result = rt.toolbox.call("single_agent", call.tool, call.args)
            asks += int(call.tool == ASK_TOOL)
        turns = [
            *state.get("turns", []),
            {"role": "assistant", "content": dump(call)},
            {"role": "user", "content": f"Tool result: {render_result(result)}"},
        ]
        return {"turns": turns, "pending": None, "asks": asks}

    def after_agent(state: AgentState) -> str:
        return END if state.get("outcome") else "tools"

    g = StateGraph(AgentState)
    g.add_node("agent", guarded(agent))  # type: ignore[arg-type]
    g.add_node("tools", tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, ["tools", END])
    g.add_edge("tools", "agent")
    return g
