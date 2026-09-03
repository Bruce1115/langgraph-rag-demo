from langgraph.graph import MessagesState


class AgentState(MessagesState):
    current_question: str