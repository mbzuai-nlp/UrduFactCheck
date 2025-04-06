import os
import json
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from utils.langchain_model import ChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END

# Get the relative path to the examples file
EXAMPLES_FILE = os.path.join(os.path.dirname(__file__), "examples.json")

# Load the examples from the JSONL file
with open(EXAMPLES_FILE, "r") as f:
    examples = json.load(f)

# Define the ExampleSelector
example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # This is the number of examples to produce.
    k=2,
)


class TranslationModel(BaseModel):
    question_urdu: str = Field(default=False)
    answer_urdu: str = Field(default=False)


parser = PydanticOutputParser(pydantic_object=TranslationModel)


prefix = """
You are an expert Urdu translator. Your task is to translate the following question-answer (QA) pairs from English to Urdu.

### Instructions
- Translate both the question and the answer into Urdu.
- Translate proper nouns into Urdu only if there's a widely accepted Urdu version (e.g., "India" → "بھارت", "Syria" → "شام"). But do this when necessary, avoid when these appear in the name of an organization.
- Pay careful attention to masculine and feminine grammatical forms in Urdu.
- Maintain a formal and fluent Urdu style.
- If the question or answer contains factual or technical terms (e.g., award names, organization name), retain them in transliterated form where appropriate.
- Dates should be translated into Urdu format (e.g., "January 1, 2020" → "یکم جنوری 2020").
- Important Formatting Guideline:
    When translating questions or answers from English to Urdu that include English acronyms or abbreviations (e.g., IEEE, NASA, UNESCO), follow these rules:
        1. Maintain the acronym in its original English form (do not translate or transliterate it).
        2. Place the acronym at a natural position in the sentence where it does not disrupt the right-to-left (RTL) flow of Urdu text.
        3. Prefer placing English acronyms after the Urdu date or subject, not at the beginning of the sentence.
        4. When a translated Urdu sentence includes Western numerals (e.g., 2022, 1999) or LTR text, ensure:
            a. These elements do not appear at the beginning of the Urdu sentence.
            b. Always place an Urdu phrase or word before the numeral to maintain proper RTL flow.
            c. Do not translate or change the numeral itself.
            d. This also follows for English acronyms, abbreviations, or LTR text.
        5. Ensure the final Urdu sentence is grammatically correct, visually aligned, and fluent to read.
        Incorrect (structurally broken):
            a. سال 2010 میں IEEE کس کو ایوارڈ دیا گیا؟
            b. 2 of January of 2019
            c. 2022 رگبی یورپ چیمپئن شپ کا حصہ بننے والے اسپین اور رومانیہ کے درمیان رگبی میچ میں 27 فروری 2022 کو اسپین کے لیے تمام کنورژنز کس کھلاڑی نے اسکور کیے؟
        Correct (natural Urdu structure):
            a. فرینک روزن بلیٹ ایوارڈ کس کو دیا گیا؟ IEEE سال 2010 میں
            b. سال 2019 میں 2 جنوری کو
            c. رگبی یورپ چیمپئن شپ 2022 کا حصہ بننے والے اسپین اور رومانیہ کے درمیان رگبی میچ میں، 27 فروری 2022 کو اسپین کے لیے تمام کنورژنز کس کھلاڑی نے اسکور کیے؟

Here are a few examples of QA pairs and expected translations:
"""

suffix = """
### Translation
question: {question}
answer: {answer}

### Formated Instructions:
{format_instructions}
"""

# Prompt
example_prompt = PromptTemplate.from_template(
    "question: {question}\nanswer: {answer}\nquestion_urdu: {question_urdu}\nanswer_urdu: {answer_urdu}"
)

prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    prefix=prefix,
    suffix=suffix,
)

prompt = prompt.partial(
    format_instructions=parser.get_format_instructions(),
)

# Model
model = ChatModel(
    provider="openai",
    model_name="gpt-4o",
)

# Chain
chain = prompt | model | parser


# States
class InputState(BaseModel):
    question: str = Field(description="The question to be translated.", default="")
    answer: str = Field(description="The answer to be translated.", default="")


class UrduTranslatorState(BaseModel):
    question: str = Field(description="The question to be translated.", default="")
    answer: str = Field(description="The answer to be translated.", default="")
    question_urdu: str = Field(description="The translated question.", default="")
    answer_urdu: str = Field(description="The translated answer.", default="")


class OutputState(BaseModel):
    question_urdu: str = Field(description="The translated question.", default="")
    answer_urdu: str = Field(description="The translated answer.", default="")


# Function
def urdu_translator(state: UrduTranslatorState) -> UrduTranslatorState:
    result = chain.invoke(
        {
            "question": state.question,
            "answer": state.answer,
        }
    )

    state.question_urdu = result.question_urdu
    state.answer_urdu = result.answer_urdu

    return state


# Graph
graph_builder = StateGraph(UrduTranslatorState, input=InputState, output=OutputState)

graph_builder.add_node("urdu_translator", urdu_translator)
graph_builder.add_edge(START, "urdu_translator")
graph_builder.add_edge("urdu_translator", END)

graph = graph_builder.compile()
graph.step_timeout = 20
