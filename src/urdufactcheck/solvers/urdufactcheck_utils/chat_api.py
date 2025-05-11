from __future__ import annotations

import os
import json
import ast
import openai
import asyncio
from openai import AsyncOpenAI


class OpenAIChat:
    def __init__(
        self,
        model_name="gpt-4o",
        max_tokens=2500,
        temperature=0,
        top_p=1,
        request_timeout=120,
    ):
        if "gpt" not in model_name:
            openai.api_base = "http://localhost:8000/v1"
        else:
            # openai.api_base = "https://api.openai.com/v1"
            openai.api_key = os.environ.get("OPENAI_API_KEY", None)
            assert (
                openai.api_key is not None
            ), "Please set the OPENAI_API_KEY environment variable."
            assert (
                openai.api_key != ""
            ), "Please set the OPENAI_API_KEY environment variable."
        self.client = AsyncOpenAI()

        self.config = {
            "model_name": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "request_timeout": request_timeout,
        }

    def extract_list_from_string(self, input_string):
        start_index = input_string.find("[")
        end_index = input_string.rfind("]")

        if start_index != -1 and end_index != -1 and start_index < end_index:
            return input_string[start_index : end_index + 1]
        else:
            return None

    def extract_dict_from_string(self, input_string):
        start_index = input_string.find("{")
        end_index = input_string.rfind("}")

        if start_index != -1 and end_index != -1 and start_index < end_index:
            return input_string[start_index : end_index + 1]
        else:
            return None

    def _json_fix(self, output):
        return output.replace("```json\n", "").replace("\n```", "")

    def _boolean_fix(self, output):
        return output.replace("true", "True").replace("false", "False")

    def _type_check(self, output, expected_type):
        try:
            output_eval = ast.literal_eval(output)
            if not isinstance(output_eval, expected_type):
                return None
            return output_eval
        except:
            return None

    async def dispatch_openai_requests(
        self,
        messages_list,
    ) -> list[str]:
        """Dispatches requests to OpenAI API asynchronously.

        Args:
            messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        Returns:
            List of responses from OpenAI API.
        """

        async def _request_with_retry(messages, retry=3):
            for _ in range(retry):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.config["model_name"],
                        messages=messages,
                        max_tokens=self.config["max_tokens"],
                        temperature=self.config["temperature"],
                        top_p=self.config["top_p"],
                    )
                    return response
                except openai.RateLimitError:
                    await asyncio.sleep(1)
                except openai.Timeout:
                    await asyncio.sleep(1)
                except openai.APIError:
                    await asyncio.sleep(1)
            return None

        async_responses = [_request_with_retry(messages) for messages in messages_list]

        return await asyncio.gather(*async_responses, return_exceptions=True)

    def run(self, messages_list, expected_type):
        retry = 1
        responses = [None for _ in range(len(messages_list))]
        messages_list_cur_index = [i for i in range(len(messages_list))]

        while retry > 0 and len(messages_list_cur_index) > 0:
            messages_list_cur = [messages_list[i] for i in messages_list_cur_index]

            predictions = asyncio.run(
                self.dispatch_openai_requests(
                    messages_list=messages_list_cur,
                )
            )

            # Save the cost of the API call to a JSONL file
            if os.environ.get("SAVE_MODEL_COST", "False") == "True":
                MODEL_COST_PATH = os.environ.get("MODEL_COST_PATH", "model_cost.jsonl")
                for prediction in predictions:
                    if prediction is not None:
                        completion_tokens = prediction.usage.completion_tokens
                        prompt_tokens = prediction.usage.prompt_tokens
                        total_tokens = prediction.usage.total_tokens
                        with open(MODEL_COST_PATH, "a") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "model": self.config["model_name"],
                                        "prompt_tokens": prompt_tokens,
                                        "completion_tokens": completion_tokens,
                                        "total_tokens": total_tokens,
                                    }
                                )
                                + "\n"
                            )

            preds = [
                self._type_check(
                    self._boolean_fix(
                        self._json_fix(prediction.choices[0].message.content)
                    ),
                    expected_type,
                )
                if prediction is not None
                else None
                for prediction in predictions
            ]
            finised_index = []
            for i, pred in enumerate(preds):
                if pred is not None:
                    responses[messages_list_cur_index[i]] = pred
                    finised_index.append(messages_list_cur_index[i])

            messages_list_cur_index = [
                i for i in messages_list_cur_index if i not in finised_index
            ]

            retry -= 1

        return responses
