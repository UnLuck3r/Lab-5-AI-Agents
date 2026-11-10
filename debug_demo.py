import os
from hr_agent import graph, Command

cfg = {"configurable": {"thread_id": "T06_DEBUG"}}

print("--- 1. Пошаговый поток (stream_mode='updates') ---")
inputs = {
    "request": "Помогите",
    "clarified_text": None,
    "category": "unknown",
    "confidence": 0.0,
    "clarify_count": 0,
    "error": None,
    "response": "",
    "route_log": []
}

for update in graph.stream(inputs, cfg, stream_mode="updates"):
    print("UPDATE:", update)

# Ответ на 1-е уточнение
print("\n--- 2. Возобновление 1 ---")
for update in graph.stream(Command(resume="не знаю"), cfg, stream_mode="updates"):
    print("UPDATE:", update)

# Ответ на 2-е уточнение
print("\n--- 3. Возобновление 2 ---")
for update in graph.stream(Command(resume="просто так"), cfg, stream_mode="updates"):
    print("UPDATE:", update)

print("\n--- 4. История снимков состояния (get_state_history) ---")
for state in graph.get_state_history(cfg):
    print(f"Step: {state.metadata.get('step')}, Next: {state.next}, Log: {state.values.get('route_log')}")
