
import os
import json
import asyncio
import tiktoken

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Remember information from the conversation history. "
    "Answer accurately and do not invent personal information."
)


# -------------------------------
# 1. Token counting
# -------------------------------

def count_tokens(messages):

    encoding = tiktoken.get_encoding("cl100k_base")

    text = " ".join(
        str(message.get("content", ""))
        for message in messages
    )

    return len(encoding.encode(text))


# -------------------------------
# 2. Stateless demonstration
# -------------------------------

async def stateless_demo():

    print("\n--- STATELESS LLM DEMO ---")

    response1 = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "My name is Arun."}
        ]
    )

    print("Call 1:", response1.choices[0].message.content)

    # Completely separate call, no previous history
    response2 = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "What is my name?"}
        ]
    )

    print("Call 2:", response2.choices[0].message.content)


# -------------------------------
# 3. Conversation class
# -------------------------------

class Conversation:

    def __init__(self, system_prompt=SYSTEM_PROMPT):

        self.system_prompt = system_prompt

        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

    async def send(self, user_text):

        # Bonus: Reset command
        if user_text.strip().lower() == "/reset":

            self.messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                }
            ]

            print("\nConversation reset successfully!")
            print("History contains only the system prompt.")
            return

        # Add user message
        self.messages.append({
            "role": "user",
            "content": user_text
        })

        # Send complete conversation history
        response = await client.chat.completions.create(
            model=MODEL,
            messages=self.messages
        )

        assistant_reply = response.choices[0].message.content

        # Store assistant response
        self.messages.append({
            "role": "assistant",
            "content": assistant_reply
        })

        print("\nUser:", user_text)
        print("Assistant:", assistant_reply)

        print(
            "History token count:",
            count_tokens(self.messages)
        )

    # -------------------------------
    # 4. Save conversation
    # -------------------------------

    def save(self, filename="conversations/conversation.json"):

        os.makedirs(
            os.path.dirname(filename),
            exist_ok=True
        )

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(
                self.messages,
                file,
                indent=4,
                ensure_ascii=False
            )

        print("\nConversation saved:", filename)

    # -------------------------------
    # 5. Load conversation
    # -------------------------------

    def load(self, filename="conversations/conversation.json"):

        with open(filename, "r", encoding="utf-8") as file:
            loaded_messages = json.load(file)

        if not isinstance(loaded_messages, list):
            raise ValueError("Invalid conversation format")

        if not loaded_messages or loaded_messages[0] != {
            "role": "system",
            "content": self.system_prompt
        }:
            raise ValueError("System prompt is missing or incorrect")

        self.messages = loaded_messages

        print("\nConversation loaded successfully!")

        print(
            "Loaded history token count:",
            count_tokens(self.messages)
        )


# -------------------------------
# 6. Six-turn conversation
# -------------------------------

async def conversation_demo():

    print("\n--- CONVERSATION MEMORY DEMO ---")

    chat = Conversation()

    questions = [
        "My name is Arun.",
        "I am learning Python programming.",
        "I am interested in Machine Learning.",
        "What is my name?",
        "Which programming language am I learning?",
        "What field am I interested in?"
    ]

    for question in questions:
        await chat.send(question)

    # Save
    chat.save()

    # Load into a new conversation object
    new_chat = Conversation()
    new_chat.load()

    # Ask a question after loading
    await new_chat.send(
        "Summarize what you know about me."
    )

    # Bonus demonstration
    await new_chat.send("/reset")

    print("\nMessages after reset:")
    print(json.dumps(new_chat.messages, indent=4))


# -------------------------------
# 7. Main
# -------------------------------

async def main():

    await stateless_demo()

    await conversation_demo()


if __name__ == "__main__":
    asyncio.run(main())
