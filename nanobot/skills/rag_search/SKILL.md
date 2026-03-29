---
name: rag_search
description: Search RAG memory for relevant facts from past conversations.
always: false
---

# RAG Search Skill

Search the RAG (Retrieval-Augmented Generation) memory vector store for relevant facts from past conversations.

## Usage

Call this skill when you need to:
- Find specific facts mentioned in previous conversations
- Search for user preferences or statements
- Retrieve context that may not be in the current conversation

## How It Works

1. Embeds your query using the embedding API
2. Searches Qdrant vector store for similar content
3. Reranks results for relevance
4. Returns top-K matches with scores

## Example Queries

- What does user like to drink
- Find information about user work schedule
- Search for mentions of coffee preferences
- What did user say about their family

## Response Format

Returns matched memories with relevance scores:
- [Score: 0.85] User prefers ca phe sua da, usually drinks in the morning
- [Score: 0.72] User works at Cake Digital Bank as SRE

## Fallback

If RAG is unavailable, returns: RAG memory is currently unavailable
