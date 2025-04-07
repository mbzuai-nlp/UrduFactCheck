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

formatted_examples = [
    {
        "question": ex["question"],
        "answer": ex["answer"],
        "question_urdu": ex["question_urdu"],
        "answer_urdu": ex["answer_urdu"],
        "text": f"{ex['question']} {ex['answer']}",
    }
    for ex in examples
]

# Define the ExampleSelector
example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    # This is the list of examples available to select from.
    formatted_examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # This is the number of examples to produce.
    k=3,
    input_keys=["text"],
    fetch_k=16,
)


class TranslationModel(BaseModel):
    question_urdu: str = Field(default=False)
    answer_urdu: str = Field(default=False)


parser = PydanticOutputParser(pydantic_object=TranslationModel)


prefix = """
You are an expert Urdu translator. Your task is to translate the following question-answer (QA) pairs from English to Urdu.

### Instructions
- Translate both the **question** and **answer** into **formal, fluent Urdu**.
- Use correct **masculine/feminine grammatical forms** in Urdu.
- Translate **proper nouns** only if a widely accepted Urdu version exists  
  (e.g., "India" → "بھارت", "Syria" → "شام").  
  Avoid translating proper nouns when they appear in the name of an organization.
- Retain **technical or factual terms** (e.g., award names, organization names) in **transliterated form**, where appropriate.
- Translate **dates** into proper Urdu format  
  (e.g., "January 1, 2020" → "یکم جنوری 2020").

### Important Formatting Guidelines
1. **English acronyms and abbreviations** (e.g., IEEE, NASA, UNESCO):
   - Do **not** translate or transliterate.
   - Place them at a **natural position** in the Urdu sentence (ideally after the date or subject).
   - Avoid starting Urdu sentences with acronyms or left-to-right (LTR) text.

2. **Western numerals and LTR elements** (e.g., 2022, 7.8.8, Notepad++):
   - Do **not** convert numerals to Urdu words.
   - Always place an **Urdu phrase before** such elements to maintain proper **right-to-left (RTL)** sentence flow.
   - This applies to acronyms, version numbers, software/product names, etc.

    Incorrect (structurally broken):
        a. سال 2010 میں IEEE فرینک روزن بلیٹ ایوارڈ کس کو دیا گیا؟
        b. 2 of January of 2019
        c. 2022 رگبی یورپ چیمپئن شپ کا حصہ بننے والے اسپین اور رومانیہ کے درمیان رگبی میچ میں 27 فروری 2022 کو اسپین کے لیے تمام کنورژنز کس کھلاڑی نے اسکور کیے؟
    Correct (natural Urdu structure):
        a. فرینک روزن بلیٹ ایوارڈ کس کو دیا گیا؟ IEEE سال 2010 میں
        b. سال 2019 میں 2 جنوری کو
        c. رگبی یورپ چیمپئن شپ 2022 کا حصہ بننے والے اسپین اور رومانیہ کے درمیان رگبی میچ میں، 27 فروری 2022 کو اسپین کے لیے تمام کنورژنز کس کھلاڑی نے اسکور کیے؟

3. Ensure the final Urdu sentence is:
   - Grammatically correct  
   - Visually aligned for RTL display  
   - Fluent and natural to read

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
            "text": f"{state.question} {state.answer}",
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
