import textwrap
import socket
import os
from openai import OpenAI
import csv
import io
from trajectory import trajectory_A, trajectory_B, context
from csv_to_json import trajectory_string_to_json


host_node = socket.gethostname()
asurite_id = "apoojar4"
# print(socket.gethostbyname(host_node))
# print(ollama_base_url())

# Get the dynamic port from the environment, default to 11434 if not set
ollama_port = os.environ.get("OLLAMA_PORT", "11434")

print(f"Connecting to Ollama on {host_node}:{ollama_port}")

client = OpenAI(
    base_url=f"http://{asurite_id}@{host_node}:{ollama_port}/v1",  # Local Ollama API
    api_key="ollama",
    # extra_body={"reasoning_effort": "high"},              
)

def call_root_lm(client, messages) -> str:
    # self.llm_conversation.append({"role": role, "content": text})

    completion = client.chat.completions.create(
        model="gpt-oss:120b",
        messages=messages
    )
    response = completion.choices[0].message.content
    thinking = completion.choices[0].message.to_dict().get("reasoning", "")

    return response, thinking
    # self.add_llm_conversation(response, "assistant")
    
def call_sub_lm(prompt: str) -> str:
    print("[SubLM QUERY]\n", prompt, "\n[END SUBLM QUERY]")
    completion = client.chat.completions.create(
        model="gpt-oss:120b",
        messages=[{"role": "user", "content": prompt}]
    )
    response = completion.choices[0].message.content
    # thinking = completion.choices[0].message.to_dict().get("reasoning", "")

    return response
#     # SubLM – can be same or smaller model
#     ...
    
class PersistentREPL:
    def __init__(self, trajectory_A, trajectory_B, context, globals):
        # Create a dict as the execution globals
        self.globals = globals
        self.final_value = None

    def _final(self, *args, **kwargs):
        # store final answer in a REPL-visible way
        self.final_value = (args, kwargs)

    def execute(self, code: str) -> str:
        # execute code and capture stdout (truncate)
        import io, contextlib, sys

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, self.globals, self.globals)
        except Exception as e:
            print(f"[RUNTIME ERROR] {e}", file=buf)
        out = buf.getvalue()
        # hard truncate stdout
        max_len = 3000
        if len(out) > max_len:
            out = out[:max_len] + "\n...[truncated]..."
        return out

def extract_repl_code(i, text: str) -> str | None:
    print("*"*100, end="\n\n")
    print(f"EXECUTING REPL {i}:\n")
    print(text, end="\n\n")
    print("*"*100)
    # find ```repl ... ``` block
    start = text.find("```repl")
    if start == -1:
        return None
    start = text.find("\n", start)
    end = text.find("```", start + 1)
    if end == -1:
        return None
    return text[start+1:end]

def is_final_call(text: str) -> bool:
    return "final(" in text.lower()

def run_rlm_trajectory_judge(root_model, trajectory_A, trajectory_B, context):
    system_prompt = """
    REASONING: High

You are operating inside a persistent Python REPL environment.

You have access to:
- A variable `context` dictionary containing detailed structured information for the current task. This will include the meaning of each states and actions (Don't print it all, use the specific keys instead). 
- A variable `trajectory_A` and `trajectory_B`, each a very long trajectory (don't print them all).
- A function `llm_query(prompt: str)` that lets you call an LLM on smaller, focused inputs.
- Standard Python libraries.

You are judging which trajectory better achieves the task goal, **beyond** the raw environment reward. Consider:
- Goal attainment (does it actually succeed, and how cleanly?)
- Stability and smoothness
- Efficiency (time, path quality, obvious waste / dithering)
- Any qualitatively bad behavior a human would dislike.

Rules:
- Write only executable Python code inside ```repl``` blocks.
- Use the REPL to compute statistical metrics over the trajectories that might be useful for judging performance.
- Store intermediate results in variables (e.g., `metrics_A`, `metrics_B`, `flags_A`, `flags_B`).
- Use `print()` if you want to see the value of any variable or small summaries/statistics needed for your own reasoning, not full trajectories.
- Store the results in separate variables if you need to refer back to them.
- If you need semantic reasoning over excerpts (e.g. suspicious segments), call `llm_query(...)` on those small excerpts.
- You will be called iteratively with an updated transcript of your previous code and printed outputs. Continue reasoning from there.

When you are completely done and ready to answer which trajectory is better:
- Call the function `final("Trajectory A", "reasoning...", "confidence_score")` or `final("Trajectory B", "reasoning...", "confidence_score")` inside the ```repl``` block.
- Do **NOT** put anything else in that turn besides the `final(...)` call.
- Confidence score should be between 1-10.

TIPS:
Look for previous code execution results, it should display errors and outputs in the conversation history. Use that to debug and iteratively improve your code.

IMPORTANT:
Only use **ONE** ```repl``` block per turn, and make sure it is valid Python code that can be executed.
Also add a small description about your thought process and what you are plan to achieve in that turn as a comment at the top of the ```repl``` block.
This will help you keep track of your reasoning steps and also be helpful when you review the conversation history.

You should **not** rush to conclusion. Only after validating enough evidence, you can call the `final(...)` function to make your choice.
Analyse the response from the previous iteration to get an idea of what to compute next, and iteratively build your case.
"""

    task_prompt = """You are given two trajectories, `trajectory_A` and `trajectory_B`.

Each trajectory is a list of dictionaries, where each dictionary corresponds to a timestep and contains:
- `t`: the timestep index
- `state_ID`: a unique identifier for the state at time t
- `next_state_ID`: a unique identifier for the state at time t+1
- `state_0`, `state_1`, ..., `state_10`: the state vector at time t

Your job:
1. Analyze both trajectories.
2. Decide which trajectory better solves the task in a human-aligned sense, not just raw reward.
3. Use the REPL to compute useful metrics (e.g., progress-to-goal over time, stability, safety, smoothness).
4. If needed, you may inspect and reason about specific segments in more detail.
5. At the end, choose either "Trajectory A" or "Trajectory B" and give a brief justification.

Note: The actions would go over limit in some steps, but the environment would clip them to be within valid range.

Below are some guidelines to help you analyze the trajectories effectively:
1. Don't just summarize the trajectories; analyze the behaviors exhibited by each policy.
2. Evaluate how effectively each policy achieves the environment's goals.
3. Discuss the strengths and weaknesses of each policy based on their trajectories.
4. Consider any trade-offs made by each policy in terms of performance, stability, and efficiency.
5. Don't just rely on the length of the trajectories; consider the quality of actions taken at each step.
6. Lookout for trajectories exhibiting signs of reward hacking or gaming the environment.
7. Concentrate on the trajectory's ability of fulfilling the goal.
8. Try to identify the pivotal moments in the trajectories that lead to success or failure, and analyze those in detail.

The full trajectory data are already loaded as Python objects:
- `trajectory_A`
- `trajectory_B`
- Additional metadata lives in `context` dictionary. Be sure to check it in a batch/smart way to gather the heads-up information there (e.g., task description, goal, state/action meanings, evaluation criteria)."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]

    repl = PersistentREPL(trajectory_A, trajectory_B, context)
    i=0

    while True:
        root_out, thinking = call_root_lm(client, messages)   # assistant content string

        # If the model directly emits FINAL(...), you can just parse and return.
        if is_final_call(root_out):
            # We still execute the code to trigger FINAL() in REPL.
            code = extract_repl_code(i, root_out)
            if code:
                stdout = repl.execute(code)
            # final value should now be in repl.final_value
            return repl.final_value

        code = extract_repl_code(i, root_out)
        if code:
            stdout = repl.execute(code)
        else:
            stdout = "[NO REPL CODE IN TURN -or- NOT PROPERLY WRAPPED IN ```repl``` BLOCK]"

        # Append to history: model output and truncated stdout
        messages.append({"role": "assistant", "content": root_out})
        messages.append({"role": "user", "content": f"[REPL OUTPUT]\n{stdout}"})

        with open(f"rlm/debug_turn_{i}.txt", "w") as f:
            f.write("=== ROOT LLM OUTPUT ===\n")
            f.write(root_out + "\n\n")
            # if code:
            #     f.write("=== EXECUTED CODE ===\n")
            #     f.write(code + "\n\n")
            f.write("=== REPL STDOUT ===\n")
            f.write(stdout + "\n\n")
            f.write("=== THINKING ===\n")
            f.write(thinking + "\n")

        with open(f"rlm/final_prompt.txt", "w") as f:
            f.write(system_prompt + "\n\n" + task_prompt + "\n\n")
            f.write("=== FULL CONVERSATION HISTORY ===\n")
            for msg in messages:
                f.write(f"{msg['role'].upper()}:\n{msg['content']}\n\n")
        i+=1
    
if __name__ == "__main__":
    # Example usage
    trajectory_A = trajectory_string_to_json(trajectory_A)
    trajectory_B = trajectory_string_to_json(trajectory_B)
    context = context
    result = run_rlm_trajectory_judge(None, trajectory_A, trajectory_B, context)
    print("Final result:", result)