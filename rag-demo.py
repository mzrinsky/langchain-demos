#!/usr/bin/env python3

import argparse
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader, DirectoryLoader, PyPDFLoader, PythonLoader, TextLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_core.embeddings import Embeddings
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
import os


def _add_context( args: dict ):
  """Handle the add context command"""
  text_loader_kwargs = { "autodetect_encoding": True }
  embedding_model = OllamaEmbeddings( model = args.embedding_model )
  splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, chunk_overlap = 100
  )
  # Note: It's much better to split each document in an optimal way, but this is just a demo.
  # for more information see: https://python.langchain.com/api_reference/text_splitters/index.html#
  pdf_loader = DirectoryLoader(
    args.path,
    glob = "**/*.pdf",
    loader_cls = PyPDFLoader,
    show_progress = True,
    use_multithreading = True
  )
  txt_loader = DirectoryLoader(
    args.path,
    glob = "**/*.txt",
    loader_cls = TextLoader,
    loader_kwargs = text_loader_kwargs,
    show_progress = True,
    use_multithreading = True
  )
  md_loader = DirectoryLoader(
    args.path,
    glob = "**/*.md",
    loader_cls = TextLoader,
    loader_kwargs = text_loader_kwargs,
    show_progress = True,
    use_multithreading = True
  )
  py_loader = DirectoryLoader(
    args.path,
    glob = "**/*.py",
    loader_cls = PythonLoader,
    show_progress = True,
    use_multithreading = True
  )
  csv_loader = DirectoryLoader(
    args.path,
    glob = "**/*.csv",
    loader_cls = CSVLoader,
    show_progress = True,
    use_multithreading = True
  )
  loader = MergedDataLoader(
    loaders = [ pdf_loader, txt_loader, md_loader, py_loader, csv_loader ]
  )
  vector_db = _get_vector_database(
    name = args.database_name, embedding_model = embedding_model
  )
  split_docs = loader.load_and_split( text_splitter = splitter )
  vector_db.add_documents( documents = split_docs )


def _llm_prompt( args: dict ):
  """Handle the llm command"""
  global vector_db
  embedding_model = OllamaEmbeddings( model = args.embedding_model )
  vector_db = _get_vector_database(
    name = args.database_name, embedding_model = embedding_model
  )
  llm = ChatOllama( model = args.model )
  tools = ToolNode( name = "run_tools", tools = [ query_documents ] )

  graph_builder = StateGraph( MessagesState )
  graph_builder.add_node(
    "run_llm_prompt", lambda state: run_llm_prompt( state, llm )
  )
  graph_builder.add_node( tools )
  graph_builder.add_node(
    "generate_tool_response",
    lambda state: generate_tool_response( state, llm )
  )

  graph_builder.set_entry_point( "run_llm_prompt" )
  graph_builder.add_conditional_edges(
    "run_llm_prompt",
    tools_condition,
    {
    END: END,
    "tools": "run_tools"
    },
  )
  graph_builder.add_edge( "run_tools", "generate_tool_response" )
  graph_builder.add_edge( "generate_tool_response", END )

  graph = graph_builder.compile()
  # print the lang graph chart for debugging..
  # print(graph.get_graph().draw_ascii())
  for step in graph.stream(
    { "messages": [ {
    "role": "user",
    "content": args.prompt
    } ] },
    stream_mode = "values",
  ):
    step[ "messages" ][ -1 ].pretty_print()


@tool( response_format = "content_and_artifact" )
def query_documents( query: str ):
  """Retrieve document context related to a user query."""
  global vector_db
  # this is doing the most basic type of search,
  # and probably the least useful combined with the basic text splitting above.
  retrieved_docs = vector_db.similarity_search( query, k = 2 )
  serialized = "\n\n".join(
    ( f"Source: {doc.metadata}\nContent: {doc.page_content}" )
    for doc in retrieved_docs
  )
  return serialized, retrieved_docs


def _get_vector_database(
  name: str, embedding_model: Embeddings, path: str = ""
):
  """Return a specific vector database instance"""
  if path != "":
    data_path = path
  else:
    data_path = os.path.join(
      os.path.dirname( __file__ ), "data", f"{name}-db"
    )
  return Chroma(
    collection_name = name,
    embedding_function = embedding_model,
    persist_directory = data_path,
  )


def run_llm_prompt( state: MessagesState, llm: ChatOllama ):
  """Run the user prompt on the llm"""
  llm_with_tools = llm.bind_tools( [ query_documents ] )
  response = llm_with_tools.invoke( state[ "messages" ] )
  return { "messages": [ response ] }


def generate_tool_response( state: MessagesState, llm: ChatOllama ):
  """Invoke the LLM to generate the response, using context when available."""
  recent_tool_messages = _get_tool_messages( state )
  prompt = _get_context_prompt( state, recent_tool_messages )
  response = llm.invoke( prompt )
  return { "messages": [ response ] }


def _get_context_prompt( state: MessagesState, tool_messages: list ):
  """Helper to generate a modified prompt containing context."""
  if ( tool_messages.count ):
    docs_content = "\n\n".join( doc.content for doc in tool_messages )
  else:
    docs_content = "No documents found."
  system_message_content = (
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. If you are unsure of the answer, say you are unsure."
    "\n\n"
    f"{docs_content}"
  )
  conversation_messages = [
    message for message in state[ "messages" ]
    if message.type in ( "human",
    "system" ) or ( message.type == "ai" and not message.tool_calls )
  ]
  prompt = [ SystemMessage( system_message_content ) ] + conversation_messages
  return prompt


def _get_tool_messages( state: MessagesState ):
  """Helper to aggregate the most recent tool messages and return them"""
  recent_tool_messages = []
  for message in reversed( state[ "messages" ] ):
    if message.type == "tool":
      recent_tool_messages.append( message )
    else:
      break
  return recent_tool_messages[ ::-1 ]


def main():
  parser = argparse.ArgumentParser(
    description =
    'This is a simple demo showing how to add documents to a persistent vector db, and then query it using an LLM.',
    epilog = 'See: https://github.com/mzrinsky/langchain-demos',
  )
  parser.set_defaults( embedding_model = "nomic-embed-text:latest" )
  subparsers = parser.add_subparsers(
    dest = 'command', required = True, help = 'Sub-command help'
  )

  # Context commands
  context_parser = subparsers.add_parser(
    'context', help = 'Context related operations'
  )
  context_subparsers = context_parser.add_subparsers(
    dest = 'action', help = 'Action for context'
  )

  add_context_parser = context_subparsers.add_parser(
    'add', help = 'Add a new context'
  )
  add_context_parser.set_defaults( command_function = _add_context )
  add_context_parser.add_argument(
    '-b',
    '--database-name',
    required = True,
    help = 'Name of the vector store database to store context.'
  )
  add_context_parser.add_argument(
    'path',
    help =
    'Path to the directory to add as context (all pdf, md, csv, txt, and py files will be added.)'
  )

  # LLM commands
  model_parser = subparsers.add_parser(
    'llm', help = 'LLM related operations'
  )
  model_parser.set_defaults( command_function = _llm_prompt )
  model_parser.add_argument(
    '-b',
    '--database-name',
    required = True,
    help = 'Name of the vector store database to query for context.'
  )
  model_parser.add_argument(
    '-m', '--model', required = True, help = 'Name of a LLM to prompt.'
  )
  model_parser.add_argument( 'prompt', help = 'Prompt to run on the LLM.' )

  args = parser.parse_args()
  args.command_function( args )


if __name__ == '__main__':
  main()
