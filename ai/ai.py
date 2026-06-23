import json
import sys
from pathlib import Path

from ollama import chat

# =========================
# CONFIG
# =========================

MODEL = "ministral-3:3b"
CHARACTER_FILE = Path("characters/laracroft.txt")
MEMORY_FILE = Path("memory/chat.json")

# =========================
# CHARACTER
# =========================

def load_character():
    try:
        with open(CHARACTER_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Character load error: {e}")
        return "You are Lara Croft."

# =========================
# MEMORY
# =========================

def load_memory(character_prompt):
    MEMORY_FILE.parent.mkdir(exist_ok=True)
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return [{"role": "system", "content": character_prompt}]

def save_memory(messages):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("Memory save error:", e)

# =========================
# CHAT LOOP
# =========================

def main():
    character_prompt = load_character()
    messages = load_memory(character_prompt)

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
            response = chat(
                model=MODEL,
                messages=messages,
                options={"temperature": 0.85, "top_p": 0.95, "num_predict": 250}
            )
            reply = response["message"]["content"]
            print(f"\nLARA: {reply}\n")
            messages.append({"role": "assistant", "content": reply})
            save_memory(messages)

        except Exception as e:
            print(f"\nSYSTEM ERROR: {e}\n")

if __name__ == "__main__":
    main()
