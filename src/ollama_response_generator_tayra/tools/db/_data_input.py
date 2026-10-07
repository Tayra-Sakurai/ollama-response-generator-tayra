# SPDX-FileCopyrightText: 2026-present Tayra Sakurai <tayra_sakurai@icloud.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Input chat data."""
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ._models import ChatData, StartRegExp
from os import PathLike
from typing import Any

__all__ = [
    'get_message_pattern',
    'load_file_content',
    'structure_data_file',
    'install_to_db',
]


def get_message_pattern(
    model: BaseChatModel,
    text: list[str]
) -> str:
    """Gets the message start pattern detected by LLM.

    Args:
        model: The LLM to be used to detect the patterns.
        text: The sample data of the chat history.

    Returns:
        The chat message group start pattern.

    Raises:
        TypeError: The type structure mismatched.
    """
    structured = create_agent(
        model=model,
        tools=[],
        response_format=StartRegExp
    )
    prompt = f"""You are the smart agent to detect the message start pattern from the following chat history data.

Since multi-line messages exist, please be sure to match ONLY the first line of message.

# Chat History

{''.join(text)}
"""
    result = structured.invoke({
        'messages': [
            HumanMessage(prompt),
        ]
    })
    if isinstance(result, StartRegExp):
        return result.exp
    elif isinstance(result, dict):
        print(result)
        return result['exp']
    else:
        raise TypeError('Structured type mismatched.')


def load_file_content(
    file_path: str | PathLike[Any],
    encoding: str = 'utf-8'
) -> list[str]:
    """Gets the first ten lines of the chat history text file.

    Args:
        file_path: The file path to the data file.
        encoding: The codec of the text file.

    Returns:
        The file contents.
    """
    with open(file_path, 'r', encoding=encoding) as file:
        return file.readlines(10)


def structure_data_file(
    model: BaseChatModel,
    file_path: str | PathLike[Any],
    pattern: str,
    encoding: str = 'utf-8'
) -> list[ChatData]:
    """Retreives the chats from the chat history data.

    Args:
        model: The LLM used to detect messages.
        file_path: The path to the history data file.
        pattern: The pattern to detect the message.
        encoding: The codec of the chat history file.

    Returns:
        The data from the chat history.

    Raises:
        TypeError: The return type from the LLM was not valid.
    """
    content: str
    with open(file_path, 'r', encoding=encoding) as file:
        content = file.read()
    structured = create_agent(
        model=model,
        tools=[],
        response_format=ChatData
    )
    splitter = RecursiveCharacterTextSplitter(
        separators=[pattern],
        is_separator_regex=True,
        chunk_size=50,
        chunk_overlap=50
    )
    chunks = splitter.split_text(content)
    data: list[ChatData] = []
    for chunk in chunks[1:]:
        prompt = f"""Please retreive the message content and the metadata from the message text.

# Message

```text
{chunk}
```
"""
        result = structured.invoke(
            {
                'messages': [
                    HumanMessage(prompt),
                ],
            }
        )
        if isinstance(result, ChatData):
            data.append(result)
        else:
            raise TypeError('Invalid return format.')
    return data


def install_to_db(
    db_provider: VectorStore,
    data: list[ChatData]
) -> None:
    """Installs the data onto the designated database.

    Args:
        db_provider: The database provider object.
        data: The data to be stored in the database.
    """
    documents: list[Document] = []
    for datumn in data:
        document = Document(
            page_content=datumn.content,
            metadata={
                'timestamp': datumn.timestamp,
                'role': datumn.user,
            }
        )
        documents.append(document)
    db_provider.add_documents(documents)
