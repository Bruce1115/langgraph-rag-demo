from langchain_openai import ChatOpenAI


def create_bailian_chat_model(
    *,
    api_key: str,
    base_url: str,
    model: str,
) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0,
    )