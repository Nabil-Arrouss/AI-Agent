# basic imports
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# load the env so we have the credentials 
load_dotenv()

# set up an llm (brain of our agent) (you can choose any model you want, gpt or anthropic or groq ...)
llm = ChatGroq(model="llama-3.3-70b-versatile")

### invoke to test and to get an answer with no agent functionality (testing if the brain works) ###
#response = llm.invoke("what is the meaning of life?")
#print(response)

### prompt template: here we provide the llm with more information on what we actually want it to do ###
### we specify how we want to get the output from the llm ###
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    
### we create a parser, here we create pydantic schema (sort of). we can spesify other schemas too, like json ###
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

### we create our prompt, basically providing more info to the llm on what it is supposed to be doing ###
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are research assistant that will help generate a research paper.
            Answer the user query and use use necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format())

### building the agent ###