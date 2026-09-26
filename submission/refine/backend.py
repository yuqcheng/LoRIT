import os
import random
import re
import sys
import threading
import openai
from tenacity import retry, wait_exponential, stop_after_attempt

from .io_utils import load_config

class Backend:

    def __init__(self, general_config, method_config_name=None):

        if method_config_name is not None:
            child_module_path = None
            for klass in type(self).__mro__:
                f = getattr(sys.modules.get(klass.__module__, None), "__file__", None)
                if f and os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(f)), "configs")):
                    child_module_path = os.path.dirname(os.path.abspath(f))
                    break
            if child_module_path is None:
                raise FileNotFoundError(f"no configs/ dir found for {type(self).__name__}")
            self.method_config = load_config(os.path.join(child_module_path, "configs", f"{method_config_name}.yaml"))
        
        self.model_api_config = general_config["model_api_config"]
        self.model_name = general_config["model_name"]
        self.model_temperature = general_config["model_temperature"]
        self.model_max_tokens = general_config["model_max_tokens"]
        self.model_timeout = general_config["model_timeout"]
        
        self.token_stats = {
            self.model_name: {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        }
        self.phase_token_stats = {}
        self.context_clamps = 0
        self._token_stats_lock = threading.Lock()

        self.memory = {}
        self.tools = {}
        
    def inference(self, sample):
        query = sample["query"]
        response = self.call_llm(prompt=query)
        return {"response": response}

    @retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(8))
    def call_llm(self, prompt=None, system_prompt=None, messages=None, model_name=None,
                 temperature=None, max_tokens=None, phase=None):
        
        name = model_name if model_name is not None else self.model_name
        model_dict = random.choice(self.model_api_config[name]["model_list"])
        real_model, model_url, api_key = model_dict['model_name'], model_dict['model_url'], model_dict['api_key']
        
        if messages is None:
            assert prompt is not None, "'prompt' must be provided if 'messages' is not provided."
            if system_prompt is not None:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]
        
        model_temperature = temperature if temperature is not None else self.model_temperature

        request_dict = {
            "model": real_model,
            "messages": messages,
            model_dict.get("max_tokens_param", "max_tokens"):
                max_tokens if max_tokens is not None else self.model_max_tokens,
            "timeout": self.model_timeout
        }
        if "o1" not in real_model and not model_dict.get("no_temperature"):
            request_dict["temperature"] = model_temperature
        if model_dict.get("extra_body"):
            request_dict["extra_body"] = model_dict["extra_body"]

        llm = openai.OpenAI(base_url=model_url, api_key=api_key)
        try:
            try:
                completion = llm.chat.completions.create(**request_dict)
            except openai.BadRequestError as e:
                m = re.search(r"maximum context length is (\d+) tokens and your request has (\d+) input tokens",
                              str(e))
                room = int(m.group(1)) - int(m.group(2)) - 16 if m else 0
                if room < 256:
                    raise
                request_dict["max_tokens"] = room
                with self._token_stats_lock:
                    self.context_clamps += 1
                completion = llm.chat.completions.create(**request_dict)
            response, num_prompt_tokens, num_completion_tokens = completion.choices[0].message.content, completion.usage.prompt_tokens, completion.usage.completion_tokens
        finally:
            llm.close() 

        if model_dict.get("forbid_thinking") and isinstance(response, str) and "<think>" in response:
            raise RuntimeError(f"thinking output from {name!r}; relaunch the server with the no-think chat template (scripts/serve_local.sh)")

        if isinstance(response, str):
            with self._token_stats_lock:
                if name not in self.token_stats:
                    self.token_stats[name] = {"num_llm_calls": 1, "prompt_tokens": num_prompt_tokens, "completion_tokens": num_completion_tokens}
                else:
                    self.token_stats[name]["num_llm_calls"] += 1
                    self.token_stats[name]["prompt_tokens"] += num_prompt_tokens
                    self.token_stats[name]["completion_tokens"] += num_completion_tokens
                if phase is not None:
                    ps = self.phase_token_stats.setdefault(
                        phase, {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
                    ps["num_llm_calls"] += 1
                    ps["prompt_tokens"] += num_prompt_tokens
                    ps["completion_tokens"] += num_completion_tokens
        else:
            raise ValueError(f"Invalid response from LLM: {response}")
        
        return response

    def get_token_stats(self):
        return self.token_stats

    def reset_token_stats(self):
        with self._token_stats_lock:
            self.token_stats = {self.model_name: {"num_llm_calls": 0, "prompt_tokens": 0,
                                                  "completion_tokens": 0}}
            self.phase_token_stats = {}
            self.context_clamps = 0

    def get_phase_token_stats(self):
        with self._token_stats_lock:
            return {k: dict(v) for k, v in self.phase_token_stats.items()}
    
    def optimizing(self, val_data):
        pass

    def retrieve_memory(self):
        pass

    def update_memory(self):
        pass
    
    def get_tool(self):
        pass
