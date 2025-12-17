import datetime
import json
import os
import time
from threading import Semaphore
from typing import List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion
from requests import Response

from model_interface import RequestCache, ChatBotInterface, Message
from llamaapi import LlamaAPI


class ChatGPTInteractor(ChatBotInterface):
    is_setup = False
    sem = Semaphore()
    thread_local: bool = True

    llama_api_key = None

    @staticmethod
    def setup():
        with open("llama_api.cfg", "r") as r:
            api_key = r.read().strip()
            if "\n"  in api_key:
                api_key = api_key.splitlines()[0].strip()
            ChatGPTInteractor.llama_api_key = api_key

        with open("chat_gpt.cfg", "r") as r:
            api_key = r.read().strip().splitlines()[0].strip()
            os.environ["OPENAI_API_KEY"] = api_key
        ChatGPTInteractor.is_setup = True

    def __init__(self, model:str, cache:RequestCache=None, temperature:Optional[float]=None, rate_limit:float=None, reasoning_effort:str=None):
        if not ChatGPTInteractor.is_setup:
            ChatGPTInteractor.setup()

        self.server = "ChatGPT"
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

        if model.startswith("local:"):
            self.is_local = True
            model_name = model.split(":")[1]
            self.server = "Local"
        elif model.startswith("llamaapi:"):
            self.is_local = True
            self.server = "LLamaAPI"
            model_name = model.split(":")[1]
        else:
            self.is_local = False
            model_name = model

        self.model = model_name

        if self.server == "LLamaAPI":
            self.client = None
            self.llama_client = LlamaAPI(self.llama_api_key)
        else:
            self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio") if self.server == "Local" else OpenAI()

        self.cache = RequestCache() if cache is None else cache

        self.request_times: List[datetime.datetime] = []


    def ask(self, msgs: List[Message]):
        msgs_formatted = [m.format() for m in msgs]
        cache_key = [
            ("msgs", msgs_formatted),
            ("model", self.model),
            ("mtype", "ChatGPT")
        ]

        if self.reasoning_effort is not None:
            cache_key += [("reasoning_effort", self.reasoning_effort)]


        if self.temperature is not None:
            cache_key += [("temp", self.temperature)]

        cache_key = str(cache_key)
        response, is_loaded = self.cache.get(cache_key)

        if is_loaded:
            return response

        if self.is_local and not self.thread_local:
            ChatGPTInteractor.sem.acquire()

        print(f">-> {datetime.datetime.now()}")

        if self.server == "Local" or self.server == "ChatGPT":
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model,
                messages=msgs_formatted,
                temperature=self.temperature,
                reasoning_effort="none" if self.reasoning_effort is None else self.reasoning_effort
            )
            response = response.choices[0].message.content
        else:
            json_packet = {
                "model": self.model,
                "messages": msgs_formatted
            }
            for repeat in [3, 5, 10, 15, None]:
                try:
                    response: Response = self.llama_client.run(json_packet)
                    break
                except:
                    if repeat is None:
                        raise
                    print(f'Waiting for {repeat} seconds')
                    time.sleep(repeat)
                    continue

            #ChatGPTInteractor.sem.acquire()
            #self.request_times.append(datetime.datetime.now())
            #ChatGPTInteractor.sem.release()
            response = json.loads(response.content.decode("UTF8"))
            response = response["choices"][0]["message"]["content"]

        if self.is_local and not self.thread_local:
            ChatGPTInteractor.sem.release()

        self.cache.save(response, cache_key)

        return response

