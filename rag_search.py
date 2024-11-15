from typing import List, Dict, Any
import requests
import os
import re
from pydantic import BaseModel, Field
import streamlit as st

import googlesearch
from trafilatura import fetch_url, extract

from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

def re_index(original):
    new = []
    miss_values = []
    flag = False
    for i in range(len(original)):
        new.append(i+1)
        if i+1 not in original:
            miss_values.append([original[i],i+1])
            flag = True
        elif flag:
            miss_values.append([original[i],i+1])
    return new, miss_values

def highlight_citations(markdown_text):
    highlighted_text = re.sub(r'\((\d+(?:,\s*\d+)*)\)', r'<span style="color:blue;">(\1)</span>', markdown_text)
    return highlighted_text

def remove_citations_and_links(text: str) -> str:
    if text is not None:
        # Remove [number] citations
        text = re.sub(r'\[\d+\]', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
    return text

def search_google(query, api_key, cse_id):
    # URL endpoint of Google Custom Search API
    url = 'https://www.googleapis.com/customsearch/v1'
    
    # Parameters for the API request
    params = {
        'q': query,            # Query topic
        'key': api_key,        # API Key
        'cx': cse_id,          # Custom Search Engine ID
        'num': 3               # The number of search results to return
    }
    # Send GET request to the API
    response = requests.get(url, params=params)
    
    # Check if the request is successful
    if response.status_code == 200:
        search_results = response.json()
        results = search_results.get('items', [])  # Get the search results
        webs = []
        for res in results:
            threshold = 0.7
            _dict = {}
            _dict['url'] = res['link']
            if 'wiki' in _dict['url']:
                threshold = 0.6
            web_text = extract(fetch_url(res['link']))
            try:
                web_text = web_text[:int(threshold*len(web_text))]
            except as Error:
                pass
            _dict['content'] = remove_citations_and_links(web_text)
            _dict['title'] = res['title']
            webs.append(_dict)
        return {"topic":query,"context":webs}
    else:
        return None
    
class CitedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user query, which is based only on the given sources.",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC sources which justify the answer.",
    )

class ReportGenerator:
    def __init__(self, embeddings, len, model_name: str = "gpt-4o"):
        """Initialize the Text Generator"""
        self.embeddings = embeddings
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
        )
        self.len = len
        self.report_chain = self._create_report_chain()

    def _create_report_chain(self):
        """Create the report generation chain."""
        report_template = """
        YOU MUST STRICTLY COMPLY WITH THE REQUIREMENTS BELOW:

        Use only the contents in all <Doc#> sections of 'Available Documents' to write a narrative story about {query}.
        
        **Must use all provided <Doc#>.**
        
        **Citation Requirements:** Cite sources directly in the text using the (number) format, referring specifically to the <Doc#> that provide each piece of information.
        
        **Required Length:** The story should be approximately **{len} WORDS** (within ±10 percents of this length).
        
        **Do not create or add 'In Conclusion', 'In Summary', or any introduction. Begin directly with the main content.**
        **Do not include any information or sections not provided or not accurate.** 
        
        Write in **MARKDOWN FORMAT**.
        
        The story should include two primary sections: **'History'** and **'Applications'**. After these, add any **topic-specific sections** based on the content and relevance to the topic. 
        - Ensure that the **length and level of detail** are balanced across sections
        - If necessary to reach the target length, additional relevant sections from the provided documents may be included.
        - Do not create content beyond the provided documents to extend the length.

        Formatting Guidelines:
        - The highest level of headings is level 2 (##).
        - Bold the most relevant terms or ideas where helpful.
        - Keep content accurate, factual, and without additions or extrapolated conclusions.

        **Available Documents:**
        {formatted_docs}

        STORY:
        """
        
        report_prompt = PromptTemplate(
            template=report_template,
            input_variables=["query", "formatted_docs", "len"]
        )
        structured_llm = self.llm.with_structured_output(CitedAnswer)
        report_chain = (
            RunnablePassthrough.assign(formatted_docs=(lambda x: self._format_documents(x["context"]))).assign(
                len=(lambda x: self.len)
            )
            | report_prompt
            | structured_llm
        )
        return  report_chain

    def _format_documents(self, docs: dict) -> str:
        """Format documents for LLM consumption with citations."""
        texts = docs["context"]
        formatted_docs = []
        texts = [doc['content'] for doc in texts]
        for i, doc in enumerate(texts, 1):
            source_info = f"[Doc{i}] "
            formatted_docs.append(
                f"<{source_info}>\n{doc}\n</{source_info}>"
            )
        return "\n\n".join(formatted_docs)

    def _format_citations(self, docs: List[dict], index: List[int], new_index: List[int]) -> str:
        """Create detailed citations for the sources section."""
        citations = []

        for i in index:
            metadata = docs[i-1]['url']
            ref_title = docs[i-1]['title']
            citation = f"[({new_index[index.index(i)]})-{ref_title}]"

            citation += f"({metadata})"
            
            citations.append(citation)
        
        return "\n\n".join(citations)

    def generate_report(self, question: str) -> Dict[str, Any]:
        """
        Generate a report answering the question using retrieved documents.
        
        Args:
            question: The research question to answer
            k: Number of documents to retrieve
            
        Returns:
            Dict containing the report and metadata
        """
        try:
            gg_api_key = st.secrets["GG_API_KEY"]
            gg_cse_id = st.secrets["CSE_ID"]
            srcs = search_google(question, gg_api_key, gg_cse_id)
            chain_input = {"query": question, "context": srcs}
            report_result = self.report_chain.invoke(chain_input)
            # Add sources section
            final_report = report_result.dict()
            index = final_report['citations']
            ans = final_report['answer']
            new_index, miss_values = re_index(index)
            for i in range(len(miss_values)):
                need_repl, repl = miss_values[i][0], miss_values[i][1]
                ans = ans.replace(f'({need_repl})', f'({repl})')
            ans = highlight_citations(ans)
            full_report = f"{ans}\n\nReferences:\n\n{self._format_citations(srcs["context"], index, new_index)}"
            full_report = f'''# {question.upper()}\n\n'''+full_report 
            return {
                "report": full_report,
                "metadata": {
                    "topic:": question,
                    "model": self.llm.model_name,
                }
            }
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            raise

