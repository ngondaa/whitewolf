import json
import re
import sys
import datetime
from pathlib import Path

from ollama import chat

# =========================
# CONFIG
# =========================

MODEL = "gemma3:1b"
CHARACTER_FILE = Path("characters/laracroft.txt")
MEMORY_FILE = Path("memory/chat.json")

OLLAMA_OPTIONS = {
    "temperature": 0.85,
    "top_p": 0.95,
    "num_predict": 250
}

SYSTEM_PROMPT = """You are Lara Croft — archaeologist, explorer, survivor.
Speak naturally. First person. No asterisks, no emotes, no stage directions.
Be direct, dry, occasionally wry. You've seen things most people can't imagine.

If the user asks for the current time, respond ONLY with valid JSON:
{"action": "get_time"}

Otherwise, reply as Lara — normal conversation.
"""

# =========================
# CHARACTER
# =========================

def load_character():
    try:
        return CHARACTER_FILE.read_text(encoding="utf-8")
    except Exception:
        return SYSTEM_PROMPT

# =========================
# MEMORY
# =========================

def load_memory(system_prompt):
    MEMORY_FILE.parent.mkdir(exist_ok=True)
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return [{"role": "system", "content": system_prompt}]

def save_memory(messages):
    try:
        MEMORY_FILE.write_text(
            json.dumps(messages, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    except Exception as e:
        print(f"[memory error] {e}")

# =========================
# ACTION ROUTER
# =========================

def extract_action(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return None

def execute_action(action_data):
    action = action_data.get("action", "").lower().strip()

    if action == "get_time":
        now = datetime.datetime.now().strftime("%I:%M %p")
        return f"The current time is {now}."

    return None

# =========================
# CHAT LOOP
# =========================

def main():
    system_prompt = load_character()
    messages = load_memory(system_prompt)

    print("WHITEWOLF — type 'quit' to exit\n")

    while True:
        try:
            user_text = input("YOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_text:
            continue
        if user_text.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_text})

        try:
            # Stream the response
            full_response = ""
            print("\nLARA: ", end="", flush=True)

            stream = chat(
                model=MODEL,
                messages=messages,
                stream=True,
                options=OLLAMA_OPTIONS
            )

            for chunk in stream:
                content = chunk["message"]["content"]
                full_response += content
                print(content, end="", flush=True)

            print("\n")

            # Check if response is an action
            action_data = extract_action(full_response)
            if action_data:
                result = execute_action(action_data)
                if result:
                    # Feed result back for a natural reply
                    messages.append({"role": "assistant", "content": full_response})
                    messages.append({"role": "user", "content": f"[SYSTEM RESULT]: {result}"})

                    follow_up = chat(
                        model=MODEL,
                        messages=messages,
                        stream=False,
                        options=OLLAMA_OPTIONS
                    )
                    reply = follow_up["message"]["content"]
                    print(f"LARA: {reply}\n")
                    messages.append({"role": "assistant", "content": reply})
                else:
                    messages.append({"role": "assistant", "content": full_response})
            else:
                messages.append({"role": "assistant", "content": full_response})

            save_memory(messages)

        except Exception as e:
            print(f"\n[error] {e}\n")

if __name__ == "__main__":
    main()
