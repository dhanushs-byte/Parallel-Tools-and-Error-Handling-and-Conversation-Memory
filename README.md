# Week 5 – Day 3: LLM Application Development

## Overview

This project demonstrates advanced LLM application development concepts using Python and OpenRouter API.

The implementation covers parallel tool calling, asynchronous execution, tool error handling, stateless LLM behavior, conversation history management, token counting, and JSON-based conversation persistence.

## Technologies Used

- Python 3.14
- OpenRouter API
- OpenAI Python SDK
- Asyncio
- JSON
- Tiktoken
- Python-dotenv

## Project Structure

```text
week5_day3/
│
├── .env
├── .gitignore
├── requirements.txt
├── task1_parallel_tools.py
├── task2_conversation_memory.py
├── conversations/
│   └── conversation.json
└── README.md
```

## Task 1 – Parallel Tools and Error Handling

### Objective

Execute multiple independent tools concurrently, handle tool failures, and return tool results to the LLM using the corresponding `tool_call_id`.

### Tools Implemented

1. `get_order_status(order_id)` – Retrieves order information.
2. `get_customer(customer_id)` – Retrieves customer information.

### Parallel Tool Calling

Test prompt:

```text
Status of orders 1001 and 1002 and customer 5
```

The model requests three independent tool calls:

- Order 1001
- Order 1002
- Customer 5

The tools are executed concurrently using:

```python
asyncio.gather()
```

Each result is returned with its original `tool_call_id`, ensuring correct mapping between tool requests and responses.

### Error Handling

Four error scenarios were tested:

| Test Case | Input | Expected Error |
|---|---|---|
| Unknown Order | Order ID 9999 | Order 9999 not found |
| Invalid Argument | Order ID "abc" | order_id must be an integer |
| Unknown Tool | get_payment_details | Unknown tool |
| Invalid JSON | Malformed JSON | Invalid JSON arguments |

### Error Recovery Example

Input:

```text
Order ID: 9999
```

Tool result:

```json
{
  "error": "Order 9999 not found"
}
```

Expected model behavior:

```text
Sorry, order 9999 was not found.
Could you please provide a valid order ID?
```

The error is returned to the model as a tool result rather than crashing the application.

### Key Concepts

- Parallel tool calling
- `asyncio.gather()`
- `tool_call_id`
- Tool execution loop
- Exception handling
- Error recovery using LLMs

## Task 2 – Conversation Memory

### Objective

Demonstrate that LLM API calls are stateless by default and implement conversation history to maintain context across multiple interactions.

### 1. Stateless LLM Demonstration

First API call:

```text
User: My name is Arun.
Assistant: Nice to meet you, Arun!
```

Second independent API call without previous history:

```text
User: What is my name?
Assistant: I don't know your name.
```

This demonstrates that separate API calls do not automatically share conversation history.

### 2. Conversation Class

Implemented a reusable `Conversation` class that maintains messages in the following format:

```python
messages = [
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "My name is Arun."},
    {"role": "assistant", "content": "Nice to meet you, Arun!"}
]
```

The complete message history is sent to the LLM for every new request.

### 3. Six-Turn Conversation

| Turn | User Message | Expected Memory |
|---|---|---|
| 1 | My name is Arun. | Name: Arun |
| 2 | I am learning Python programming. | Python |
| 3 | I am interested in Machine Learning. | Machine Learning |
| 4 | What is my name? | Arun |
| 5 | Which programming language am I learning? | Python |
| 6 | What field am I interested in? | Machine Learning |

### 4. Token Counting

Used the `tiktoken` library to calculate the number of tokens in the conversation history after each turn.

```python
def count_tokens(messages):
    encoding = tiktoken.get_encoding("cl100k_base")

    text = " ".join(
        str(message.get("content", ""))
        for message in messages
    )

    return len(encoding.encode(text))
```

Token counts are printed after every conversation turn.

Note: This measures content tokens approximately and does not represent the complete API billing token count.

### 5. JSON Persistence

Implemented two methods:

- `save()` – Saves conversation history into a JSON file.
- `load()` – Restores previously saved conversation history.

Output file:

```text
conversations/conversation.json
```

This allows conversations to continue after restarting the application.

## Bonus Challenge – Reset Command

Implemented `/reset` to clear the conversation history while preserving the system prompt.

Example:

```text
/reset

Conversation reset successfully!
History contains only the system prompt.
```

## Installation and Execution

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
