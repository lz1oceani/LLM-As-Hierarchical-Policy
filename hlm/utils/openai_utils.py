import os, copy, time, logging, tiktoken, re
from IPython import embed
from typing import Union, Sequence
from .model_utils import DecodingArguments


TOTAL_TOKENS = PROMPT_TOKENS = COMPLETION_TOKENS = TOTAL_MONEY = 0


def login_openai():
    import openai
    openai.api_key = os.getenv("OPENAI_KEY")


def list_openai_model_ids():
    import openai

    engines = openai.Engine.list()
    ids = [_.id for _ in engines.data]
    return ids


def get_total_tokens():
    return TOTAL_TOKENS


def get_total_money():
    return TOTAL_MONEY


OPENAI_PRICE_MAP = {
    "gpt-4": [0.03, 0.06],
    "gpt-4-32k": [0.06, 0.12],
    "gpt-3.5-turbo": [0.0015, 0.002],
    "gpt-3.5-turbo-instruct": [0.0015, 0.002],
    "gpt-3.5-turbo-16k": [0.003, 0.004],
}


OPENAI_CONTEXT_WINDOW_MAP = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-instruct": 4096,
    "gpt-3.5-turbo-16k": 16384,
}


def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if "gpt-3.5" in model or "gpt-4" in model:
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )


def is_chat_model(model_name):
    return model_name.startswith("gpt-35") or model_name.startswith("gpt-3.5") or model_name.startswith("gpt-4")


def process_openai_args(model_name, prompt, decoding_args, decoding_kwargs):
    import openai

    chat_model = model_name.startswith("gpt-3.5") or model_name.startswith("gpt-4")
    API = openai.ChatCompletion if chat_model else openai.Completion
    if chat_model and isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]

    if decoding_args is None:
        decoding_args = DecodingArguments()

    batch_decoding_args = copy.deepcopy(decoding_args)
    num_prompt_tokens = num_tokens_from_messages(prompt, model=model_name)
    if decoding_args.max_tokens is None:
        from .model_utils import get_max_length
        batch_decoding_args.max_tokens = get_max_length(model_name)
    elif decoding_args.max_tokens <= num_prompt_tokens and decoding_args.max_tokens < get_max_length(model_name):
        print("Your prompt length is longer than max_tokens!! We reset the model's context window size!")
        batch_decoding_args.max_tokens = get_max_length(model_name)

    batch_decoding_args.max_tokens = batch_decoding_args.max_tokens - num_prompt_tokens - 1
    if batch_decoding_args.max_tokens < 0:
        from hlm.utils.model_utils import get_max_length

        if model_name == "gpt-3.5-turbo":
            model_name = "gpt-3.5-turbo-16k"
        elif model_name == "gpt-4":
            model_name = "gpt-4-32k"
        else:
            raise NotImplementedError(f"Model {model_name} is not supported.")
        print(f"Auto change to {model_name}.")
        batch_decoding_args.max_tokens = get_max_length(model_name) - num_prompt_tokens - 1

    if openai.api_type in ["azure"]:
        model_name = model_name.replace(".", "")
        shared_kwargs = dict(engine=model_name)
    else:
        shared_kwargs = dict(model=model_name)
    shared_kwargs.update(
        dict(
            **batch_decoding_args.__dict__,
            **decoding_kwargs,
        )
    )
    shared_kwargs["messages" if chat_model else "prompt"] = prompt
    return API, shared_kwargs


def process_openai_outputs(completion_batch):
    global TOTAL_TOKENS, PROMPT_TOKENS, COMPLETION_TOKENS, TOTAL_MONEY

    choices = completion_batch.choices
    usage = completion_batch.usage
    TOTAL_TOKENS += usage["total_tokens"]
    PROMPT_TOKENS += usage["prompt_tokens"]
    COMPLETION_TOKENS += usage["completion_tokens"]

    model_name = completion_batch["model"]
    price = None
    for prefix, value in OPENAI_PRICE_MAP.items():
        if "." not in model_name and "." in prefix:
            prefix = prefix.replace(".", "")
        if model_name.startswith(prefix):
            price = value
            break

    TOTAL_MONEY += (price[0] * usage["prompt_tokens"] + price[1] * usage["completion_tokens"]) / 1000
    ret = []
    for choice in choices:
        if choice.finish_reason == "content_filter":
            continue
        if is_chat_model(model_name):
            item = choice.message
            if hasattr(item, "content"):
                item = item.content
            else:
                # print("")
                print("The OpenAI output does not contain content!")
                print(item)
                embed()
                exit(0)
        else:
            item = choice.text
        if len(item) > 0:
            ret.append(item)
    return ret


def process_timeout_error(model, e, sleep_time=60):
    if "limit" in str(e).lower():
        error_messages = str(e).lower()
        groups = re.findall(r"retry after (\d*) seconds", error_messages)
        wait_time = int(groups[0]) if len(groups) >= 1 else sleep_time
        logging.warning(f"Hit request rate limit; retrying after {wait_time} seconds...")
        time.sleep(wait_time + 0.5)  # Annoying rate limit on requests.
    elif "Access denied due".lower() in str(e).lower():
        logging.error(e)
        exit(0)
    else:
        logging.warning(f"OpenAIError: {e}.")


def openai_completion(
    prompt: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: DecodingArguments = None,
    model_name="gpt-3.5-turbo",
    sleep_time=60,
    return_list=True,
    **decoding_kwargs,
):
    import openai

    API, shared_kwargs = process_openai_args(model_name, prompt, decoding_args, decoding_kwargs)
    ret = []
    for i in range(5):
        try:
            completion_batch = API.create(**shared_kwargs)
            ret += process_openai_outputs(completion_batch)
            if len(ret) < shared_kwargs["n"]:
                shared_kwargs["n"] -= len(ret)
            else:
                break
        except openai.error.OpenAIError as e:
            process_timeout_error(model_name, e, sleep_time)
    return ret if return_list or len(ret) > 1 else ret[0]

