# basic imports
from dotenv import load_dotenv
from pydantic import BaseModel
#from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from tools import search_tool, wiki_tool


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
system_prompt = f"""
    You are a research assistant that helps generate research papers.
    Answer the user's question.
    Return the output in this format:
    {parser.get_format_instructions()}
     """

### create a list of tools ###
tools = [search_tool, wiki_tool]

### building the agent ###
agent = create_agent(
    model=llm,
    system_prompt=system_prompt,
    tools=tools
)


### ask the user and generate a question ###
user_question = input("What can i help you research ?")
raw_respose = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": user_question 
            }
        ]
    }
)

print("#################################################")
try:
    structered_response = parser.parse(raw_respose["messages"][-1].content)
    print(structered_response)

except Exception as e:
    print("Error parsing response", e, "raw response - ", raw_respose)
