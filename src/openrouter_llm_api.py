import dataclasses
import inspect
import os, getpass
import re
import time
from typing import Any, Dict, List, Tuple, Optional
from uuid import uuid1

import loguru
from langchain_openrouter import ChatOpenRouter
import tiktoken
from tenacity import retry, stop_after_attempt

from agent_import import OpenRouter_API_Endpoint

logger = loguru.logger
logger.remove()
# logger.add(level="WARNING", sink="logs/chatgpt.log")


@dataclasses.dataclass
class Message:
    ask_id: str = ""
    ask: list = dataclasses.field(default_factory=list)
    answer: list = dataclasses.field(default_factory=list)
    answer_id: str = ""
    request_start_timestamp: float = 0.0
    request_end_timestamp: float = 0.0
    time_escaped: float = 0.0


@dataclasses.dataclass
class Conversation:
    conversation_id: str = ""
    message_list: List[Message] = dataclasses.field(default_factory=list)

    def __hash__(self):
        return hash(self.conversation_id)

    def __eq__(self, other):
        if not isinstance(other, Conversation):
            return False
        return self.conversation_id == other.conversation_id


class LLMAPI:
    def __init__(self, config: OpenRouter_API_Endpoint):
        self.name = "LLMAPI_base_class"
        self.config = config
        self.client = ChatOpenRouter(
            model = config.model_name,
            reasoning = config.reasoning,
            openrouter_provider = config.openrouter_provider,
            temperature = config.temperature
        )
        self.log_dir = "logs"
        self.history_length = 5  # maintain 5 messages in the history. (5 chat memory)
        self.conversation_dict: Dict[str, Conversation] = {}

        logger.add(sink=os.path.join(self.log_dir, "chatgpt.log"), level="WARNING")

    def _count_token(self, messages) -> int:
        """
        Count the number of tokens in the messages
        Parameters
        ----------
            messages: a list of messages
        Returns
        -------
            num_tokens: int
        """
        # count the token. Use model gpt-3.5-turbo-0301, which is slightly different from gpt-4
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        model = "gpt-3.5-turbo-0301"
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = 0
        for message in messages:
            try:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            except Exception as e:  # TODO: handle other formats
                pass
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def _token_compression(self, complete_messages) -> str:
        """
        Compress the message if it is beyond the token limit.
        For GPT-4, limit is 8k. Others are set to 16k.

        Parameters
        ----------
            complete_messages: dict
        Returns
        -------
            compressed_message: str
        """
        if getattr(self.config, "model_name", "gpt-4") == "gpt-4":
            token_limit = 8000
        else:
            token_limit = 14000  # leave some budget
        if self._count_token(complete_messages) > token_limit:
            # send a separate API request to compress the message
            chat_message = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "Please reduce the word count of the given message to save tokens. Keep its original meaning so that it can be understood by a large language model.",
                },
            ]
            compressed_message = self._chat_completion(chat_message)
            return compressed_message

        # if not compressed, return the last message
        raw_message = complete_messages[-1]["content"]
        return raw_message

    def _chat_completion_fallback(self) -> str:
        """
        A fallback method for chat completion.
        This method should be overwritten by the child class to use the custom API.
        """
        return "fallback"

    def _chat_completion(self, history: List, **kwargs) -> str:
        """
        Send a chat completion request to the API
        This method should be overwritten by the child class to use the custom API.
        Given a history of messages, return the response from the API.
        Parameters
        ----------
            history: list
                A list of messages
            **kwargs: dict
                Additional arguments to be passed to the API
        Returns
        -------
            response: str
        """
        temperature = kwargs.get("temperature", getattr(self.config, "temperature", 0.3))
        # Use the ChatOpenRouter client to generate a response
        try:
            result = self.client.invoke(history)
            response_text = getattr(result, "content", str(result))
        except Exception as e:
            wait = getattr(self.config, "error_wait_time", 10)
            logger.warning(f"API Error. Waiting for {wait} seconds: {e}")
            time.sleep(wait)
            # retry once
            try:
                result = self.client.invoke(history)
                response_text = getattr(result, "content", str(result))
            except Exception as e2:
                logger.error("Second attempt failed: %s", e2)
                raise

        # if the response is a tuple, it means that the response is not valid.
        return response_text

    def send_new_message(self, prompt_text: str, image_url: Optional[str] = None):
        # create a message
        start_time = time.time()
        if image_url is not None and type(image_url) is str:
            data = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ]
        else:
            data = [{"role": "user", "content": prompt_text}]
        history = data
        message: Message = Message()
        message.ask_id = str(uuid1())
        message.ask = data
        message.request_start_timestamp = start_time
        response = self._chat_completion(history)
        message.answer = [{"role": "system", "content": response}]
        message.request_end_timestamp = time.time()
        message.time_escaped = (
            message.request_end_timestamp - message.request_start_timestamp
        )

        # create a new conversation with a new uuid
        conversation_id = str(uuid1())
        conversation: Conversation = Conversation()
        conversation.conversation_id = conversation_id
        conversation.message_list.append(message)

        self.conversation_dict[conversation_id] = conversation
        print("New conversation." + conversation_id + " is created." + "\n")
        return response, conversation_id

    # add retry handler to retry 1 more time if the API connection fails
    @retry(stop=stop_after_attempt(2))
    def send_message(
        self, prompt_text, conversation_id, image_url: Optional[str] = None, debug_mode=False
    ):
        # create message history based on the conversation id
        chat_message = [
            {
                "role": "system",
                "content": "You are a helpful assistant",
            },
        ]
        conversation = self.conversation_dict[conversation_id]

        for _message in conversation.message_list[-self.history_length :]:
            chat_message.extend(_message.ask)
            chat_message.extend(_message.answer)
        # append the new message to the history
        # form the data that contains url
        if image_url is not None and type(image_url) is str:
            data = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ]
        else:
            data = [{"role": "user", "content": prompt_text}]
        chat_message.extend(data)
        # create the message object
        message: Message = Message()
        message.ask_id = str(uuid1())
        message.ask = data
        message.request_start_timestamp = time.time()
        # count the token cost
        num_tokens = self._count_token(chat_message)
        # Get response. If the response is None, retry.
        response = self._chat_completion(chat_message)

        # update the conversation
        message.answer = [{"role": "system", "content": response}]
        message.request_end_timestamp = time.time()
        message.time_escaped = (
            message.request_end_timestamp - message.request_start_timestamp
        )
        conversation.message_list.append(message)
        self.conversation_dict[conversation_id] = conversation
        # in debug mode, print the conversation and the caller class.
        if debug_mode:
            print("Caller: ", inspect.stack()[1][3], "\n")
            print("Message:", message, "\n")
            print("Response:", response, "\n")
            print("Token cost of the conversation: ", num_tokens, "\n")
        return response


if __name__ == "__main__":
    print("OpenRouter LLMAPI module loaded.")
