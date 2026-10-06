# SPDX-FileCopyrightText: 2026-present Tayra Sakurai <tayra_sakurai@icloud.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Models."""
from datetime import datetime
from pydantic import BaseModel, Field

__all__ = ['ChatData', 'StartRegExp']


class ChatData(BaseModel):
    """Chat data structure class.

    Args:
        user: The user who sent the message.
        content: The user's sent content.
        timestamp: The message's timestamp.
    """
    user: str = Field(
        description="The sender's name."
    )
    content: str = Field(
        description='The message content.'
    )
    timestamp: datetime = Field(
        description='The timestamp of the message.'
    )


class StartRegExp:
    """Expresses the start pattern of a message.

    Args:
        exp: The regular expression of the message start pattern.
    """
    exp: str = Field(
        description='The expression of the chat message start pattern.'
    )
