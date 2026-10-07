# SPDX-FileCopyrightText: 2026-present Tayra Sakurai <tayra_sakurai@icloud.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from ._data_input import *
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
import argparse


def main():
    """The main thead tool."""
    parser = argparse.ArgumentParser(
        description='Adds the chat history to designated directory.'
    )
    parser.add_argument(
        'history_file',
        type=str,
        help='The chat history data file.'
    )
    parser.add_argument(
        'outdir',
        type=str,
        help='The output directory.'
    )
    parser.add_argument(
        '-b',
        '--embedding',
        type=str,
        default='embeddinggemma:latest',
        help='The Ollama embedding model.'
    )
    parser.add_argument(
        '-c',
        '--chat',
        type=str,
        default='gemma4:e2b',
        help='LLM used to interpret the document.'
    )
    parser.add_argument(
        '-e',
        '--encoding',
        default='utf_8',
        type=str,
        help='The encoding of the file.'
    )
    args = parser.parse_args()
    embedding = OllamaEmbeddings(
        model=args.embedding
    )
    chatModel = ChatOllama(
        model=args.chat
    )
    db_provider = Chroma(
        collection_name='chat_history',
        embedding_function=embedding,
        persist_directory=args.outdir
    )
    sample = load_file_content(args.history_file, args.encoding)
    pattern = get_message_pattern(chatModel, sample)
    data = structure_data_file(chatModel, args.history_file, pattern, args.encoding)
    install_to_db(db_provider, data)


if __name__ == '__main__':
    main()
