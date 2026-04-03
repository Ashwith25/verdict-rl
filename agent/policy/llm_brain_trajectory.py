import gymnasium as gym
import random
import numpy as np
import os
import time
from jinja2 import Template
from openai import OpenAI
import google.generativeai as genai
import anthropic
import time
import socket
from main.core.ReplEnv import PersistentREPL
from stats.inverted_double_pendulum.idp_stats import evaluate_params
import json
from configs.inverted_double_pendulum.idp_summarise_template import TEMPLATE
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError
from typing import Tuple, Any, List, Optional, Dict
from enum import Enum
import re
# from ollama_config import ollama_base_url
# from ollama import chat

class Winner(str, Enum):
    A = "new"
    B = "ref"
    TIE = "Tie"

class OutputSchema(BaseModel):
    winner: Winner = Field(description="The trajectory which is better among the two given trajectories.")
    description: str = Field(description="Detailed comparison of the two trajectories based on the aspects mentioned above and why you chose A, B, or Tie.")

class LLMBrainTrajectory:
    def __init__(
        self,
        llm_si_template: Template,
        llm_output_conversion_template: Template,
        llm_model_name: str,
    ):
        self.llm_si_template = llm_si_template
        self.llm_output_conversion_template = llm_output_conversion_template
        self.llm_conversation = []
        self.final_value = None
        # response_schema_dict = OutputSchema.model_json_schema()
        # self.response_schema_json = json.dumps(response_schema_dict, indent=2)
        self.parser = PydanticOutputParser(pydantic_object=OutputSchema)

        assert llm_model_name in [
            "o1-preview",
            "gpt-4o",
            "gemini-2.0-flash-exp",
            "gpt-4o-mini",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.5-flash-preview-04-17",
            "o3-mini-2025-01-31",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "claude-3-7-sonnet-20250219",
            "gpt-oss:120b",
        ]
        self.llm_model_name = llm_model_name
        if "gemini" in llm_model_name:
            self.model_group = "gemini"
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        elif "claude" in llm_model_name:
            self.model_group = "anthropic"
            self.client = anthropic.Client(api_key=os.environ["ANTHROPIC_API_KEY"])
        else:
            self.model_group = "openai"
            if self.llm_model_name == 'gpt-oss:120b':
                host_node = socket.gethostname()
                asurite_id = "apoojar4"
                # print(socket.gethostbyname(host_node))
                # print(ollama_base_url())
                
                # Get the dynamic port from the environment, default to 11434 if not set
                ollama_port = os.environ.get("OLLAMA_PORT", "11434")
                
                print(f"Connecting to Ollama on {host_node}:{ollama_port}")
                
                self.client = OpenAI(
                    base_url=f"http://{asurite_id}@{host_node}:{ollama_port}/v1",  # Local Ollama API
                    api_key="ollama",
                    # extra_body={"reasoning_effort": "high"},              
                )
                # print(f"http://{asurite_id}@{host_node}:11434/v1")
            # if self.llm_model_name == 'gpt-oss:120b':
            #     host_node = socket.gethostname()
            #     ollama_host = os.environ.get('OLLAMA_HOST', f'{host_node}:11434')
            #     asurite_id = "apoojar4"
            #     print(f"http://{asurite_id}@{ollama_host}/v1")
            #     # print(socket.gethostbyname(host_node))
            #     # print(ollama_base_url())
            #     self.client = OpenAI(
            #         base_url=f"http://{asurite_id}@{ollama_host}/v1",  # Local Ollama API
            #         api_key="ollama"              
            #     )
            #     # print(f"http://{asurite_id}@{host_node}:11434/v1")
            else:
                self.client = OpenAI()

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def add_llm_conversation(self, text, role, isTool = False, body = None):
        # if self.model_group == "gpt-oss":
        #     message = {"role": role, "content": [text]}
        #     if tool_name is not None:
        #         message["tool_name"] = tool_name
        #     self.llm_conversation.append(message)
        if isTool:
            self.llm_conversation.append(body)
        elif self.model_group == "openai":
            self.llm_conversation.append({"role": role, "content": text})
        elif self.model_group == "anthropic":
            self.llm_conversation.append({"role": role, "content": text})
        else:
            self.llm_conversation.append({"role": role, "parts": text})

    def call_sub_lm(self, prompt: str) -> str:
        print("[SubLM QUERY]\n", prompt, "\n[END SUBLM QUERY]")
        completion = self.client.chat.completions.create(
            model=self.llm_model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        response = completion.choices[0].message.content
        # thinking = completion.choices[0].message.to_dict().get("reasoning", "")

        return response

    def query_llm(self):
        max_iter = [0, []]
        thinking = ""
        for attempt in range(10):
            try:
                if self.model_group == "openai":
                    completion = self.client.chat.completions.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation
                    )
                    response = completion.choices[0].message.content
                    thinking = completion.choices[0].message.to_dict().get("reasoning", "")
                    # self.add_llm_conversation(response, "assistant")

                elif self.model_group == "anthropic":
                    message = self.client.messages.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                        max_tokens=1024,
                    )
                    response = message.content[0].text
                else:
                    model = genai.GenerativeModel(model_name=self.llm_model_name)
                    chat_session = model.start_chat(history=self.llm_conversation[:-1])
                    response = chat_session.send_message(
                        self.llm_conversation[-1]["parts"]
                    )
                    response = response.text
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")
                if attempt == 9:
                    raise Exception("Failed")
                else:
                    print("Waiting for 60 seconds before retrying...")
                    time.sleep(60)

            if self.model_group == "openai":
                # add the response to self.llm_conversation
                self.add_llm_conversation(response, "assistant")
            else:
                self.add_llm_conversation(response, "model")

            return response, thinking

    def extract_repl_code(self, i, text: str) -> str | None:
        # print("*"*100, end="\n\n")
        # print(f"EXECUTING REPL {i}:\n")
        # print(text, end="\n\n")
        # print("*"*100)
        # find ```repl ... ``` block
        start_keyword = "```repl" if 'repl' in text else "```python"
        start = text.find(start_keyword)
        if start == -1:
            return None
        start = text.find("\n", start)
        end = text.find("```", start + 1)
        if end == -1:
            return None
        return text[start+1:end]

    def is_final_call(self, text: str) -> bool:
        # Check if final() is called WITHIN a ```repl``` code block
        # The final() must be properly wrapped in executable code
        repl_code = self.extract_repl_code(0, text)
        if not repl_code:
            return False
        # Check if final() with valid arguments exists in the extracted code
        # Only accept literal values (letters, numbers, spaces), no variables
        pattern = r'final\s*\(\s*["\']?[A-Za-z0-9\s]+["\']?\s*(?:,\s*["\']?[^)]+["\']?)*\s*\)'
        return bool(re.search(pattern, repl_code.lower()))

    def _final(self, *args, **kwargs):
        # store final answer in a REPL-visible way
        # Extract trajectory letter from arguments
        # Handles: "Trajectory B", "trajectory B", "B", or any case variation
        if args and isinstance(args[0], str):
            # Try to match "Trajectory/trajectory B" or just "B"
            match = re.search(r'[Tt]rajectory\s+([A-Z])|^([A-Z])$', args[0])
            if match:
                # Get whichever group matched (non-None)
                self.final_value = match.group(1) or match.group(2)
            else:
                self.final_value = args[0]
        else:
            self.final_value = args[0] if args else None
        print(f"[FINAL CALLED] Final value set to: {self.final_value}")

    def llm_update_parameters_num_optim_semantics(
        self,
        context,
        trajectory_a,
        trajectory_b,
        log_dir,
        env_desc_file=None
    ):
        self.reset_llm_conversation()

        global_vars = {
            "trajectory_A": trajectory_a,
            "trajectory_B": trajectory_b,
            "context": context,
            "llm_query": self.call_sub_lm,
            "final": self._final,
        }

        self.add_llm_conversation(self.llm_si_template.render(), "system")
        self.add_llm_conversation(self.llm_output_conversion_template.render(), "user")

        repl = PersistentREPL(global_vars)
        i=0

        while i<20:
            root_out, thinking = self.query_llm()

            code = self.extract_repl_code(i, root_out)
            if code:
                stdout = repl.execute(code)
            else:
                stdout = "[NO REPL CODE IN TURN -or- NOT PROPERLY WRAPPED IN ```repl``` BLOCK]"

            # Append to history: model output and truncated stdout
            self.add_llm_conversation(root_out, "assistant")
            self.add_llm_conversation(f"[REPL OUTPUT]\n{stdout}", "user")

            with open(f"{log_dir}/debug_turn_{i}.txt", "w") as f:
                f.write("=== ROOT LLM OUTPUT ===\n")
                f.write(root_out + "\n\n")
                # if code:
                #     f.write("=== EXECUTED CODE ===\n")
                #     f.write(code + "\n\n")
                f.write("=== REPL STDOUT ===\n")
                f.write(stdout + "\n\n")
                f.write("=== THINKING ===\n")
                f.write(thinking + "\n")

            with open(f"{log_dir}/final_prompt.txt", "w") as f:
                f.write(self.llm_si_template.render() + "\n\n" + self.llm_output_conversion_template.render() + "\n\n")
                f.write("=== FULL CONVERSATION HISTORY ===\n")
                for msg in self.llm_conversation:
                    f.write(f"{msg['role'].upper()}:\n{msg['content']}\n\n")

            if self.is_final_call(root_out):
                return self.final_value
            i+=1

        return self.final_value
