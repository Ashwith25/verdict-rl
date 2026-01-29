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
from stats.inverted_double_pendulum.idp_stats import evaluate_params
import json
from configs.inverted_double_pendulum.idp_summarise_template import TEMPLATE
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError
from typing import Tuple, Any, List, Optional, Dict
from enum import Enum
# from ollama_config import ollama_base_url
# from ollama import chat

class Winner(str, Enum):
    A = "A"
    B = "B"
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
                    self.add_llm_conversation(response, "assistant")

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

    def llm_update_parameters_num_optim_semantics(
        self,
        env_desc_file,
        trajectory_a,
        trajectory_b,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "env_description": env_desc_file,
                "response_schema": self.parser.get_format_instructions(),
                "trajectory_a": trajectory_a,
                "trajectory_b": trajectory_b,
            }
        )

        self.add_llm_conversation(system_prompt, "user")

        # api_start_time = time.time()
        response, thinking = self.query_llm()
        # api_time = time.time() - api_start_time
        try:
            validated_response = OutputSchema.model_validate_json(response)
        except ValidationError as e:
            print("INCORRECT Response from LLM:", response)
            print("Validation error:", e)
            raise e

        # print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()

        # new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        # explanation = self.query_reasoning_llm(new_parameters_lis, stats) if summary else None

        return (
            validated_response.winner.value,
            validated_response.description,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + response
            + "\n\n\nThinking:\n"
            + thinking,
            # api_time,
        )
