import dataclasses
import hashlib
import json
import os
import pickle
from enum import Enum
from typing import List

class MessageType(Enum):
    System = 0
    Assistant = 1
    User = 2

@dataclasses.dataclass
class Message:
    message_type: MessageType
    content: str

    def format(self):
        return {
            "role": {
                MessageType.System: "system",
                MessageType.User: "user",
                MessageType.Assistant: "assistant"
            }[self.message_type],
            "content": self.content
    }
class ChatBotInterface:
    def ask(self, msgs:List[Message]):
        raise NotImplementedError()

    def ask_raw(self, message:str):
        return self.ask([
            Message(MessageType.User, message)
        ])
class RequestCache:
    def __init__(self, path=".cachex", cache_key:str=None):
        if not os.path.exists(path):
            os.mkdir(path)
        self.path = path
        self.cache_key = cache_key

    def cache_path(self, request):
        request = request if self.cache_key is None else [
            ("cache_base", self.cache_key),
            request
        ]
        if not isinstance(request, str):
            request = json.dumps(request, sort_keys=True).strip()
        hash = f"{hashlib.sha256(request.encode()).hexdigest()}"
        cache_path = os.path.join(self.path, hash + ".bin")

        return cache_path

    def get(self, request):
        cache_path = self.cache_path(request)

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as r:
                return pickle.load(r), True

        return None, False

    def save(self, result, cache_key):
        cache_path = self.cache_path(cache_key)

        with open(cache_path, "wb") as w:
            pickle.dump(result, w)

class ModelGenerator:

    def __init__(self, prompt:str, system_prompt:str):
        self.prompt = prompt
        self.system_prompt = system_prompt

    def generate(self, query, cache_tag:dict=None):
        raise NotImplementedError()

    def format_query(self, query, response:str=None):
        raise NotImplementedError()
