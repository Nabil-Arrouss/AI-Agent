from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import StructuredTool
from datetime import datetime


# create a serach tool
search = DuckDuckGoSearchRun()

def search_func(query: str) -> str:
    return search.run(query)

search_tool = StructuredTool.from_function(
    func=search_func,
    name="search",
    description="Search the web for information"
)


### wikipedia tool ###
wiki_api = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=300
)

def wiki_func(query: str) -> str:
    return wiki_api.run(query)

wiki_tool = StructuredTool.from_function(
    func=wiki_func,
    name="wikipedia",
    description="Search Wikipedia for factual encyclopedic information"
)