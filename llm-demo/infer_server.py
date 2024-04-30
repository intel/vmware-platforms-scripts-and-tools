"""
Copyright (c) 2024, Intel Corporation
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
*  Neither the name of Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR APARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHTOWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOTLIMITEDTO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANYTHEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF -THE USEOF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import time
import warnings
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from contextlib import nullcontext
from typing import Any, Dict, List, Union

import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
    TextStreamer,
)
from termcolor import colored
from peft import PeftModel
from threading import Thread
import intel_extension_for_pytorch as ipex

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

import gradio as gr
import socket

print("==" * 20)
hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)
print("IP Address is:" + IPAddr)

ASSISTANT_COLOR = "light_cyan"
ASSISTANT_BACKGROUND = None
DEBUG = True


def clear_bot():
    return None, "", ""


def log_feedback(feedback, ratings):
    return "", 5


title = """
# <center> Meet Warren, your financial assistant </center>
### <center> This model is based on Llama-2-chat-7b 🦙 and was fine-tuned on a finance dataset </center>
"""

# disable
torch._C._jit_set_texpr_fuser_enabled(False)
try:
    ipex._C.disable_jit_linear_repack()
except Exception:
    pass

MODEL_DICT = {
    "llama": {
        "prefix": "<s>[INST] ",
        "system": "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n",
        "postfix": " Answer in 200 words or less [/INST] ",
        "end": " </s>",
        "history": True,
        "warmup_prompt": "Write a welcome message to the user",
    },
    "gpt_bigcode": {
        "prefix": "",
        "system": "",
        "postfix": "",
        "end": "",
        "history": False,
        "warmup_prompt": "### Function to print a welcome message to the user\ndef greet_user(name: str):",
    },
}


class CustomStreamer(TextStreamer):
    """
    Simple text streamer that prints the token(s) to stdout as soon as entire words are formed.

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

        >>> tok = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
        >>> streamer = TextStreamer(tok)

        >>> # Despite returning the usual output, the streamer will also print the generated text to stdout.
        >>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
        An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
        ```
    """

    def __init__(
        self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        # variables used in the streaming process
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

        self.first_time = True
        self.start_time = 0
        self.n_tokens = 0

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # Flush the cache, if it exists
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        if self.first_time:
            self.first_time = False
            self.start_time = time.time()
            self.n_tokens = 0

        self.n_tokens = self.n_tokens + 1
        if stream_end:
            self.start_time = time.time() - self.start_time
            time_per_token = ""
            if DEBUG:
                time_per_token = " ({:.2f} ms/token)".format(
                    1000 * (self.start_time) / (self.n_tokens - 1)
                )
            print(
                colored(text + time_per_token, ASSISTANT_COLOR, ASSISTANT_BACKGROUND),
                flush=True,
                end=None,
            )
            self.first_time = True
        else:
            print(
                colored(text, ASSISTANT_COLOR, ASSISTANT_BACKGROUND), flush=True, end=""
            )

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False


class ChatFormatter:
    """A class for formatting the chat history.

    Args:
        system: The system prompt. If None, a default ChatML-formatted prompt is used.
        user: The user prompt. If None, a default ChatML value is used.
        assistant: The assistant prompt. If None, a default ChatML value is used.

    Attributes:
        system: The system prompt.
        user: The user prompt.
        assistant: The assistant prompt.
        response_prefix: The response prefix (anything before {} in the assistant format string)
    """

    def __init__(self, model_type: str) -> None:
        print("Initiating formatter for model " + model_type)
        self.system = MODEL_DICT[model_type]["system"]
        self.prefix = MODEL_DICT[model_type]["prefix"]
        self.postfix = MODEL_DICT[model_type]["postfix"]
        self.end = MODEL_DICT[model_type]["end"]
        self.history = MODEL_DICT[model_type]["history"]
        self.warmup_prompt = MODEL_DICT[model_type]["warmup_prompt"]


class Conversation:
    """A class for interacting with a chat-tuned LLM.

    Args:
        model: The model to use for inference.
        tokenizer: The tokenizer to use for inference.
        chat_format: The chat format to use for the conversation.
        generate_kwargs: The keyword arguments to pass to `model.generate`.
        stop_tokens: The tokens to stop generation on.

    Attributes:
        model: The model to use for inference.
        tokenizer: The tokenizer to use for inference.
        chat_format: The chat format to use for the conversation.
        streamer: The streamer to use for inference.
        generate_kwargs: The keyword arguments to pass to `model.generate`.
        history: The conversation history.
        cli_instructions: The instructions to display to the user.
    """

    def __init__(
        self,
        model,
        tokenizer: Tokenizer,
        chat_format: ChatFormatter,
        generate_kwargs: Dict[str, Any],
        stop_tokens: List[str] = [],
        device: str = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.chat_format = chat_format

        stop_token_ids = self.tokenizer.convert_tokens_to_ids(stop_tokens)

        class StopOnTokens(StoppingCriteria):
            def __call__(
                self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
            ) -> bool:
                for stop_id in stop_token_ids:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        # self.streamer = CustomStreamer(
        #     tokenizer, skip_prompt=True, skip_special_tokens=True
        # )
        self.streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        self.generate_kwargs = {
            **generate_kwargs,
            # 'stopping_criteria':
            #     StoppingCriteriaList([StopOnTokens()]),
            "streamer": self.streamer,
        }
        self.history = []
        self.cli_instructions = (
            "Enter your message below.\n- Hit return twice to send input to the model\n"
            + "- Type 'clear' to restart the conversation\n- Type 'history' to see the conversation\n"
            + "- Type 'quit' to end\n- Type 'system' to change the system prompt\n"
        )
        self.device = device

    def _history_as_formatted_str(self):
        # Add system prompt
        text = self.chat_format.prefix + self.chat_format.system

        # Add history if needed
        if self.chat_format.history == True:
            for item in self.history[:-1]:
                text += (
                    item[0] + self.chat_format.postfix + item[1] + self.chat_format.end
                )
                text += self.chat_format.prefix

        # Add current prompt
        text += self.history[-1][0] + self.chat_format.postfix
        return text

    def pre_stream(self, user_inp: str, history):
        self.history = history
        self.history.append([user_inp, ""])
        conversation = self._history_as_formatted_str()
        input_ids = self.tokenizer(conversation, return_tensors="pt").input_ids
        if self.device is not None:
            input_ids = input_ids.to(self.device)

        first_token = True
        tok_len = 0
        assistant_response = ""

        # Regular decoding
        print(colored("Assistant:", ASSISTANT_COLOR, ASSISTANT_BACKGROUND))
        gkwargs = {**self.generate_kwargs, "input_ids": input_ids}
        thread = Thread(target=self.model.generate, kwargs=gkwargs)
        thread.start()
        print(colored("Start stream:", ASSISTANT_COLOR, ASSISTANT_BACKGROUND))

        return "", self.history, user_inp

    def stream(self, history):
        for text in self.streamer:
            history[-1][1] += text
            yield history

    def turn(self, user_inp: str, history):
        self.history = history
        self.history.append([user_inp, ""])
        conversation = self._history_as_formatted_str()
        input_ids = self.tokenizer(conversation, return_tensors="pt").input_ids
        if self.device is not None:
            input_ids = input_ids.to(self.device)

        first_token = True
        tok_len = 0
        assistant_response = ""

        # Regular decoding
        start = time.time()
        print(colored("Assistant:", ASSISTANT_COLOR, ASSISTANT_BACKGROUND))
        gkwargs = {**self.generate_kwargs, "input_ids": input_ids}
        output_ids = self.model.generate(**gkwargs)
        end = time.time()
        if self.device is not None:
            torch.xpu.synchronize()
            output_ids = output_ids.cpu()

        new_tokens = output_ids[0, len(input_ids[0]) :]
        response_str = "Response took {:.2f} seconds".format(end - start)
        if DEBUG:
            response_str = response_str + " seconds for {} tokens".format(
                len(input_ids[0])
            )
        print(colored(response_str, ASSISTANT_COLOR, ASSISTANT_BACKGROUND))
        assistant_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        self.history[-1][-1] = assistant_response
        return "", self.history, user_inp, ""

    def __call__(self):
        print(self.cli_instructions)
        while True:
            print("User:")
            user_inp_lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                user_inp_lines.append(line)
            user_inp = "\n".join(user_inp_lines)
            if user_inp.lower() == "quit":
                break
            elif user_inp.lower() == "clear":
                self.history = []
                continue
            elif user_inp == "history":
                print(f"history: {self.history}")
                continue
            elif user_inp == "history_fmt":
                print(f"history: {self._history_as_formatted_str()}")
                continue
            elif user_inp == "system":
                print("Enter a new system prompt:")
                new_system = input()
                self.chat_format.system = new_system
                continue
            self.turn(user_inp)


def get_dtype(dtype: str):
    if dtype == "fp32":
        return torch.float32
    elif dtype == "fp16":
        return torch.float16
    elif dtype == "bf16":
        return torch.bfloat16
    else:
        raise NotImplementedError(
            f"dtype {dtype} is not supported. "
            f"We only support fp32, fp16, and bf16 currently"
        )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def str_or_bool(v: Union[str, bool]):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        return v


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description="Load a HF CausalLM Model and use it to generate text."
    )
    parser.add_argument("-n", "--name_or_path", type=str, required=True, default=None)
    parser.add_argument("-q", "--quant_name_or_path", type=str, default=None)
    parser.add_argument("-p", "--peft_path", type=str, default=None)
    parser.add_argument("-t", "--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.05)
    parser.add_argument("--repetition_penalty", type=float, default=1.18)
    parser.add_argument("--num_beams", type=float, default=1)
    parser.add_argument(
        "--do_sample", type=str2bool, nargs="?", const=True, default=False
    )
    parser.add_argument(
        "--use_cache", type=str2bool, nargs="?", const=True, default=True
    )
    parser.add_argument("--eos_token_id", type=str, default=None)
    parser.add_argument("--pad_token_id", type=str, default=None)
    parser.add_argument(
        "--model_dtype", type=str, choices=["fp32", "fp16", "bf16"], default="bf16"
    )
    parser.add_argument(
        "--autocast_dtype", type=str, choices=["fp32", "fp16", "bf16"], default=None
    )
    parser.add_argument("--warmup", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument(
        "--trust_remote_code", type=str2bool, nargs="?", const=True, default=True
    )
    parser.add_argument(
        "--use_auth_token", type=str_or_bool, nargs="?", const=True, default=None
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--attn_impl", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--user_msg_fmt", type=str, default=None)
    parser.add_argument("--assistant_msg_fmt", type=str, default=None)
    parser.add_argument(
        "--stop_tokens",
        type=str,
        default="",
        help="A string of tokens to stop generation on; will be split on spaces.",
    )
    parser.add_argument("--port", type=int, default=65535)
    return parser.parse_args()


def maybe_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(args: Namespace) -> None:
    # Set device or device_map
    if args.device and args.device_map:
        raise ValueError("You can only set one of `device` and `device_map`.")
    if args.device is not None:
        device = args.device
        device_map = None
    else:
        device = None
        device_map = args.device_map or "auto"
    print(f"Using {device=} and {device_map=}")

    # Set model_dtype
    if args.model_dtype is not None:
        model_dtype = get_dtype(args.model_dtype)
    else:
        model_dtype = torch.float32
    print(f"Using {model_dtype=}")

    # Grab config first
    print(f"Loading HF Config...")
    from_pretrained_kwargs = {
        "use_auth_token": args.use_auth_token,
        "trust_remote_code": args.trust_remote_code,
        "revision": args.revision,
    }
    try:
        config = AutoConfig.from_pretrained(
            args.name_or_path, torchscript=True, **from_pretrained_kwargs
        )
        if args.attn_impl is not None and hasattr(config, "attn_config"):
            config.attn_config["attn_impl"] = args.attn_impl
        if hasattr(config, "init_device") and device is not None:
            config.init_device = device
        if args.max_seq_len is not None and hasattr(config, "max_seq_len"):
            config.max_seq_len = args.max_seq_len

    except Exception as e:
        raise RuntimeError(
            "If you are having auth problems, try logging in via `huggingface-cli login` "
            "or by setting the environment variable `export HUGGING_FACE_HUB_TOKEN=... "
            "using your access token from https://huggingface.co/settings/tokens."
        ) from e

    # Load HF Model
    print(f"Loading HF model with dtype={model_dtype}...")
    try:
        if args.quant_name_or_path is None:
            model = AutoModelForCausalLM.from_pretrained(
                args.name_or_path,
                config=config,
                torch_dtype=model_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                **from_pretrained_kwargs,
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                args.name_or_path,
                config=config,
                torch_dtype=torch.half,
                device_map=device_map,
                low_cpu_mem_usage=True,
                **from_pretrained_kwargs,
            )

        print("\nLoading HF tokenizer...")
        tokenizer_name = args.name_or_path
        if args.tokenizer_name_or_path is not None:
            tokenizer_name = args.tokenizer_name_or_path

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=False, **from_pretrained_kwargs
        )

        if tokenizer.pad_token_id is None:
            warnings.warn(
                "pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id."
            )
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        stop_tokens = args.stop_tokens.split()
        if config.model_type == "llama":
            tokenizer.padding_side = "right"
            model.generation_config.pad_token_id = 0
            tokenizer.pad_token_id = model.generation_config.pad_token_id
            model.generation_config.bos_token_id = 1
            tokenizer.bos_token_id = model.generation_config.bos_token_id
            model.generation_config.eos_token_id = 2
            tokenizer.eos_token_id = model.generation_config.eos_token_id
            stop_tokens.append("</s>")
        elif config.model_type == "bigcode":
            stop_tokens.append("<|endoftext|>")

        print("Stop ids: " + str(stop_tokens))

        if (args.peft_path is not None) and args.quant_name_or_path is not None:
            model = PeftModel.from_pretrained(model, args.peft_path)

        if args.quant_name_or_path is None:
            amp_enabled = None
            if device is not None:
                model = model.eval().to(device)
                model = model.to(memory_format=torch.channels_last)
                model = ipex.optimize(model, dtype=model_dtype)
                amp_enabled = torch.autocast(
                    device_type="xpu",
                    enabled=True if model_dtype != torch.float32 else False,
                    dtype=model_dtype,
                )
                model = torch.compile(model, backend="ipex")
            else:
                model = ipex.optimize_transformers(
                    model.eval(), dtype=model_dtype, inplace=True
                )
                amp_enabled = torch.cpu.amp.autocast(
                    enabled=True if model_dtype != torch.float32 else False,
                    dtype=model_dtype,
                )
                model.generate = torch.compile(model, backend="ipex")
        else:
            qconfig = ipex.quantization.default_static_qconfig_mapping
            model = ipex.optimize_transformers(
                model.eval(),
                dtype=torch.float,
                inplace=True,
                quantization_config=qconfig,
                deployment_mode=False,
            )
            if not hasattr(model, "trace_graph"):
                print("load_quantized_model")
                self_jit = torch.jit.load(args.quant_name_or_path + "/best_model.pt")
                self_jit = torch.jit.freeze(self_jit.eval())
                ipex._set_optimized_model_for_generation(
                    model, optimized_model=self_jit
                )

            print(model)
            amp_enabled = torch.cpu.amp.autocast(enabled=False)

        print(f"n_params={sum(p.numel() for p in model.parameters())}")

    except Exception as e:
        raise RuntimeError(
            "Unable to load HF model. "
            "If you are having auth problems, try logging in via `huggingface-cli login` "
            "or by setting the environment variable `export HUGGING_FACE_HUB_TOKEN=... "
            "using your access token from https://huggingface.co/settings/tokens."
        ) from e

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
        "use_cache": args.use_cache,
        "do_sample": args.do_sample,
        "eos_token_id": args.eos_token_id or tokenizer.eos_token_id,
        "pad_token_id": args.pad_token_id or tokenizer.eos_token_id,
    }
    # Autocast
    if args.autocast_dtype is not None:
        autocast_dtype = get_dtype(args.autocast_dtype)
        autocast_context = torch.autocast(model.device.type, autocast_dtype)
        print(f"Using autocast with dtype={autocast_dtype}...")
    else:
        autocast_context = nullcontext()
        print("NOT using autocast...")

    with torch.inference_mode(), torch.no_grad(), amp_enabled:
        chat_format = ChatFormatter(model_type=config.model_type)

        conversation = Conversation(
            model=model,
            tokenizer=tokenizer,
            chat_format=chat_format,
            generate_kwargs=generate_kwargs,
            stop_tokens=stop_tokens,
            device=device,
        )

        with gr.Blocks(
            css=""".gradio-container {margin: 0 !important; padding: 0 !important; max-width: 100% !important};"""
        ) as demo:
            with gr.Row():
                with gr.Column(scale=7):
                    gr.Markdown(title)
            chatbot = gr.Chatbot(elem_id="chatbot", height=600)
            with gr.Row():
                with gr.Column(scale=12):
                    textbox = gr.Textbox(
                        show_label=False,
                        placeholder="Enter your question and press ENTER",
                        container=False,
                    )
                    regenerate = gr.Textbox(
                        show_label=False,
                        placeholder="Enter your question and press ENTER",
                        container=False,
                        visible=False,
                    )
            with gr.Row(visible=True) as button_row:
                regenerate_btn = gr.Button(value="Regenerate", interactive=True)
                clear_btn = gr.Button(value="Clear history", interactive=True)

            textbox.submit(
                fn=conversation.pre_stream,
                inputs=[textbox, chatbot],
                outputs=[textbox, chatbot, regenerate],
                queue=False,
            ).then(conversation.stream, chatbot, chatbot)
            clear_btn.click(
                fn=clear_bot, outputs=[chatbot, regenerate], show_progress=False
            )
            regenerate_btn.click(
                fn=conversation.turn,
                inputs=[regenerate, chatbot],
                outputs=[textbox, chatbot, regenerate],
                show_progress=False,
            )

        demo.queue(api_open=False, max_size=5).launch(
            debug=True,
            server_name="0.0.0.0",
            server_port=args.port,
            show_api=True,
            share=False,
        )


if __name__ == "__main__":
    main(parse_args())
