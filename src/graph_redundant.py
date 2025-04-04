from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from utils.langchain_model import ChatModel
from langchain_core.output_parsers import PydanticOutputParser

from langgraph.graph import StateGraph, START, END


class TranslationModel(BaseModel):
    question_urdu: str = Field(default=False)
    answer_urdu: str = Field(default=False)


parser = PydanticOutputParser(pydantic_object=TranslationModel)


prompt_template = """
You are an expert Urdu translator. Your task is to translate the following question-answer (QA) pairs from English to Urdu.

## Instructions

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

## Examples

Example 1:
question: Who was awarded the Oceanography Society's Jerlov Award in 2018?,
answer: Annick Bricaud,
question_urdu: "سال 2018 میں اوشیانوگرافی سوسائٹی کے جیرلوو ایوارڈ سے کس کو نوازا گیا,
answer_urdu: "انیک برکیوڈ"

Example 2:
question: What total number of newly discovered pieces of music by Maddalena Casulana were played for the first time in 400 years on March 8, 2022, as part of BBC Radio 3's programming for International Women's Day?
answer: 12
question_urdu: بی بی سی ریڈیو 3 کے بین الاقوامی یوم خواتین کے لیے پروگرامنگ کے حصے کے طور پر 8 مارچ 2022 کو 400 سالوں میں پہلی بار مادالینا کاسولانا کی موسیقی کے کتنے نئے دریافت ہوئے ٹکڑوں کو چلایا گیا؟
answer_urdu: 12

Example 3:
question: What is the second song on Side Two of the album Rewind by Johnny Rivers?
answer: "For Emily, Whenever I May Find Her"
question_urdu: جانی ریورز کے البم "ریوائنڈ" کی سائیڈ ٹو پر دوسرا گانا کون سا ہے؟
answer_urdu: "فور ایملی، وینیور آئی مے فائنڈ ہر"

Example 4:
question: How many miles from the Indo-Nepal border was Amit Kumar hiding from the police on February 7, 2008?
answer: 35
question_urdu: امیت کمار 7 فروری 2008 کو ہند-نیپال سرحد سے کتنے میل دور پولیس سے چھپا ہوا تھا؟
answer_urdu: 35

Example 5:
question: Who was the wife of the Indian criminal "Welding" Kumar, who was originally known as Jeyakumar?
answer: Shanti
question_urdu: بھارتی مجرم "ویلڈنگ" کمار، جس کا اصل نام جے کمار تھا، کی بیوی کون تھی؟
answer_urdu: شانتھی

Example 6:
question: According to Karl Küchler, what did Empress Elizabeth of Austria's favorite sculpture depict, which was made for her villa Achilleion at Corfu?
answer: Poet Henrich Heine.
question_urdu: کارل کوچلر کے مطابق، آسٹریا کی ملکہ الزبتھ کے لیے ان کے ولا اچیلیئن، کورفو میں بنائی گئی ان کی پسندیدہ مجسمہ کس کو ظاہر کرتا ہے؟
answer_urdu: شاعر ہائنرش ہائنے۔

Example 7:
question: What were the month and year when Obama told Christianity Today, "I am a Christian, and I am a devout Christian. I believe in the redemptive death and resurrection of Jesus Christ"?
answer: January 2008
question_urdu: اوباما نے "کرسچینیٹی ٹوڈے" کو یہ بیان کہ "میں ایک عیسائی ہوں، اور میں ایک پرہیزگار عیسائی ہوں۔ میں یسوع مسیح کی نجات بخش موت اور قیامت پر یقین رکھتا ہوں" کس مہینے اور سال میں دیا تھا؟
answer_urdu: جنوری 2008
            
Now translate the following QA pair into Urdu using the above instructions and examples as a guide:

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
