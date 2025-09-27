from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from agent.prompts import SYSTEM_PROMPT
from agent.tools.literature import search_literature
from agent.tools.databases import query_ensembl, query_ncbi_gene, query_gwas_catalog
from agent.tools.bioinformatics import predict_splicing, run_enrichment_analysis
from agent.tools.visualization import generate_report

TOOLS = [
    search_literature,
    query_ensembl,
    query_ncbi_gene,
    query_gwas_catalog,
    predict_splicing,
    run_enrichment_analysis,
    generate_report,
]

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


def build_agent_executor() -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
    agent = create_openai_tools_agent(llm, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=True, max_iterations=10)


def format_history(history: list[dict]) -> list:
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages
