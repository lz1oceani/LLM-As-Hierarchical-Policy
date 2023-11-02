import warnings, os.path as osp, requests, numpy as np, torch, re, os, contextlib, getpass, dataclasses
from typing import Union, Sequence, Optional
from copy import deepcopy


@dataclasses.dataclass
class DecodingArguments(object):
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


def get_max_length(model_name):
    assert "gpt-3.5" in model_name or "gpt-4" in model_name
    from .openai_utils import OPENAI_CONTEXT_WINDOW_MAP
    return OPENAI_CONTEXT_WINDOW_MAP[model_name]


class ChatBot:
    model_name = None
    max_seq_len = None
    
    @classmethod
    def init(cls, model_name):
        cls.model_name = model_name
        cls.max_seq_len = get_max_length(model_name)
        print("ChatBot is ready now!")

    @classmethod
    def call_model(cls, prompt, **kwargs):
        kwargs = deepcopy(kwargs)
        decoding_args = kwargs.pop("decoding_args", None)
        kwargs.setdefault("return_list", True)
        if isinstance(prompt, (list, tuple)) and len(prompt) > 1:
            return_list = kwargs.pop("return_list")
            if cls.openai_async_obj is None:
                from .openai_utils import OpenAIAsyncClient
                print("Using OpenAI Async Client!")
                cls.openai_async_obj = OpenAIAsyncClient(cls.model_name)
            assert cls.openai_async_obj.num_remaining == 0, "You need to wait for the previous request to finish!"
            for i, prompt_i in enumerate(prompt):
                decoding_args_i = decoding_args[i] if isinstance(decoding_args, (list, tuple)) else decoding_args
                cls.openai_async_obj.async_call(prompt_i, decoding_args_i, cls.model_name, job_idx=i, **kwargs)
            ret = cls.openai_async_obj.fetch(list_item=return_list)
            ret = [ret[i] for i in range(len(prompt))]
            assert len(ret) == len(prompt), f"ret: {len(ret)}, prompt: {len(prompt)}"
            return ret

        from .openai_utils import openai_completion
        
        def squeeze_prompt(prompt):
            # Call OpenAI API, the input arguments cannot be a list or a tuple!
            if isinstance(prompt, (list, tuple)):
                assert len(prompt) == 1, "#Prompts should be 1!"
                return prompt[0]
            else:
                return prompt
        
        ret = openai_completion(squeeze_prompt(prompt), squeeze_prompt(decoding_args), cls.model_name, **kwargs)
        if isinstance(prompt, (list, tuple)):
            ret = [ret]
        return ret

    @classmethod
    def get_completion_fn(cls, decoding_args):
        return lambda prompt, **kwargs: cls.call_model(prompt, decoding_args=decoding_args, **kwargs)
