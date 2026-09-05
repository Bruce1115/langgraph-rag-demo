from langgraph.graph import MessagesState


class AgentState(MessagesState):
    current_question: str
    summary: str
    summarized_message_count: int