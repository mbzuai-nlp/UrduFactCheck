from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from utils.langchain_model import ChatModel
from langchain_core.output_parsers import PydanticOutputParser


class TranslationModel(BaseModel):
    question_urdu: str = Field(default=False)
    answer_urdu: str = Field(default=False)


parser = PydanticOutputParser(pydantic_object=TranslationModel)


prompt_template = """
Translate the following QA pairs into Urdu. 
If a proper noun is present in the question, translate it into Urdu as well.
Keep track of the maculine and feminine forms of the words.
Question: {question}
Answer: {answer}

Formated Instructions:
{format_instructions}
"""

# Prompt
prompt = PromptTemplate.from_template(prompt_template)
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
