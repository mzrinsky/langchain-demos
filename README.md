# langchain-demos

> Some simple demo code using LangChain


`rag-demo.py` - A demo using LangChain, Ollama, Chroma, RAG and Tool Calling with an LLM.

## rag-demo.py
```
usage: rag-demo.py [-h] {context,llm} ...

This is a simple demo showing how to add documents to a persistent vector db, and then query it using an LLM.
```
The main purpose is to setup a test demo for tool calling and providing context to an LLM for further testing and experimentation.  Feel free to use this as a base for your own testing.

It uses LangChain DocumentLoader to load multiple document types and process them into text chunks, creates embeddings from them etc.  It then allows you to load an LLM and prompt it, the LLM can then, using a tool call, query for relevant context which it will use in it's response.

### Quick Example
```bash
# This will process all pdf, md, csv, txt, and py files in my-test-folder.
./rag-demo.py context add -b 'testdb' ~/Documents/my-test-folder/

# this will then use qwen3 LLM to tool call relevant documents, and process the testdb vector database results.
./rag-demo.py llm -b 'testdb' -m 'qwen3:latest' "How many of these documents are about cats?"
```
