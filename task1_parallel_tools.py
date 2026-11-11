
import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")


# -------------------------------
# 1. Tool implementations
# -------------------------------

ORDERS = {
    1001: {"status": "Shipped", "amount": 2500},
    1002: {"status": "Processing", "amount": 1800}
}

CUSTOMERS = {
    5: {"name": "Arun", "email": "arun@example.com"}
}


def get_order_status(order_id):
    if isinstance(order_id, bool) or not isinstance(order_id, int):
        raise ValueError("order_id must be an integer")

    if order_id not in ORDERS:
        raise ValueError(f"Order {order_id} not found")

    return ORDERS[order_id]


def get_customer(customer_id):
    if isinstance(customer_id, bool) or not isinstance(customer_id, int):
        raise ValueError("customer_id must be an integer")

    if customer_id not in CUSTOMERS:
        raise ValueError(f"Customer {customer_id} not found")

    return CUSTOMERS[customer_id]


# -------------------------------
# 2. Tool schemas
# -------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get order status using order ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer"}
                },
                "required": ["order_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "Get customer information",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"}
                },
                "required": ["customer_id"],
                "additionalProperties": False
            }
        }
    }
]

tool_functions = {
    "get_order_status": get_order_status,
    "get_customer": get_customer
}


# -------------------------------
# 3. Execute a tool safely
# -------------------------------

def execute_tool(name, arguments):

    if name not in tool_functions:
        return {"error": f"Unknown tool: {name}"}

    try:
        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be a JSON object")

        result = tool_functions[name](**arguments)
        return {"result": result}

    except Exception as e:
        return {"error": str(e)}


# -------------------------------
# 4. Parallel tool execution
# -------------------------------

async def execute_parallel(tool_calls):

    async def run_one(call):
        try:
            arguments = json.loads(call.function.arguments)
            result = await asyncio.to_thread(
                execute_tool,
                call.function.name,
                arguments
            )
        except json.JSONDecodeError:
            result = {"error": "Invalid JSON arguments"}

        return {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result)
        }

    results = await asyncio.gather(
        *(run_one(call) for call in tool_calls)
    )

    return results


# -------------------------------
# 5. Main tool calling loop
# -------------------------------

async def ask_with_tools(question):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful customer support assistant. "
                "Use tools when necessary. If a tool returns an error, "
                "explain it politely and ask for corrected information. "
                "Never invent order or customer information."
            )
        },
        {"role": "user", "content": question}
    ]

    for _ in range(5):

        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if not message.tool_calls:
            print("\nFinal Assistant Message:")
            print(message.content)
            return message.content

        print(
            "\nTool calls in one round:",
            len(message.tool_calls)
        )

        messages.append(message.model_dump(exclude_none=True))

        results = await execute_parallel(message.tool_calls)

        for result in results:
            print("Tool result:", result)
            messages.append(result)

    print("Maximum tool rounds reached.")
    return None


# -------------------------------
# 6. Error recovery tests
# -------------------------------


async def test_error_case(test_name, tool_name, arguments):

    print("\n" + "=" * 55)
    print("ERROR TEST:", test_name)
    print("=" * 55)

    call_id = "test_" + test_name.replace(" ", "_")

    # Safely parse JSON arguments
    try:
        parsed_arguments = json.loads(arguments)

        if not isinstance(parsed_arguments, dict):
            raise ValueError("Arguments must be a JSON object")

        result = await asyncio.to_thread(
            execute_tool,
            tool_name,
            parsed_arguments
        )

    except json.JSONDecodeError:
        result = {"error": "Invalid JSON arguments"}

    except Exception as e:
        result = {"error": str(e)}

    # Simulate assistant tool call
    tool_call = {
        "id": call_id,
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": arguments
        }
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a customer support assistant. "
                "Explain tool errors politely and ask for "
                "corrected information. Never invent data."
            )
        },
        {
            "role": "user",
            "content": f"Execute {tool_name} with {arguments}"
        },
        {
            "role": "assistant",
            "tool_calls": [tool_call]
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result)
        }
    ]

    print("Error result:", json.dumps(result))

    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    print("\nFinal Assistant Message:")
    print(response.choices[0].message.content)


# -------------------------------
# 7. Run everything
# -------------------------------

async def main():

    print("\nTASK 1: PARALLEL TOOL CALLING")

    await ask_with_tools(
        "Status of orders 1001 and 1002 and customer 5"
    )

    print("\nTASK 1: FOUR ERROR CASES")

    await test_error_case(
        "Unknown Order",
        "get_order_status",
        '{"order_id": 9999}'
    )

    await test_error_case(
        "Invalid Argument",
        "get_order_status",
        '{"order_id": "abc"}'
    )

    await test_error_case(
        "Unknown Tool",
        "get_payment_details",
        '{"order_id": 1001}'
    )

    await test_error_case(
        "Invalid JSON",
        "get_order_status",
        '{"order_id": }'
    )


if __name__ == "__main__":
    asyncio.run(main())
