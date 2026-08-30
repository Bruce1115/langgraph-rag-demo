import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel

from src.infrastructure.llm import create_bailian_chat_model


def test_bailian_chat_model():
    load_dotenv()

    model = create_bailian_chat_model(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.environ["DASHSCOPE_BASE_URL"],
        model=os.getenv(
            "DASHSCOPE_CHAT_MODEL",
            "qwen-plus",
        ),
    )

    response = model.invoke(
        "Reply with exactly: pong"
    )

    assert response.content


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather for {city}"


def test_bailian_chat_model_tool_calling():
    load_dotenv()

    model = create_bailian_chat_model(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.environ["DASHSCOPE_BASE_URL"],
        model=os.getenv(
            "DASHSCOPE_CHAT_MODEL",
            "qwen-plus",
        ),
    )

    model_with_tools = model.bind_tools(
        [get_weather]
    )

    response = model_with_tools.invoke(
        "Use the weather tool to get the weather in Shenzhen."
    )

    assert response.tool_calls
    assert response.tool_calls[0]["name"] == "get_weather"




class TestGrade(BaseModel):
    binary_score: Literal["yes", "no"]


def test_bailian_chat_model_structured_output():
    load_dotenv()

    model = create_bailian_chat_model(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.environ["DASHSCOPE_BASE_URL"],
        model=os.getenv(
            "DASHSCOPE_CHAT_MODEL",
            "qwen-plus",
        ),
    )

    grader = model.with_structured_output(
        TestGrade
    )

    result = grader.invoke(
        "Is Paris the capital of France? "
        "Return yes if true, otherwise no."
    )
    
    assert isinstance(result, TestGrade)
    assert result.binary_score == "yes"