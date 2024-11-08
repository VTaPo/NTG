from typing import List, Dict, Any
# from langchain.chains import LLMChain
from openai import OpenAI
import os
from urllib.parse import quote
import streamlit as st
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
# from dotenv import load_dotenv

# load_dotenv()

# API_KEY: str = os.getenv("OPENAI_API_KEY")
# os.environ['OPENAI_API_KEY'] = API_KEY

def text_generator(org_text: str, output_len: int) -> str:
    """
    Read a raw text and generate new version using OpenAI's API.
    Args:
        org_text (str): The raw text
        max_tokens (int): The length of the output text 
    Returns:
        str: the generated text
    """

    _prompt = f'''
Generate a WIKIPEDIA-LIKE ARTICLE based on the provided source text, following these strict requirements:

TITLE AND STRUCTURE:
- THE ARTICLE IS LIMITED WITHIN {int(output_len*1.4)} TOKENS. The article must be completed before reaching the length limit.
- No need to title the output
- The level of the headings of 'History' and 'Applications' are 2.
- Generating in MARKDOWN FORMAT
- APPROPRIATE LENGTH BETWEEN 'History' and 'Applications' sections

HISTORY SECTION:
- The narrative text must be coherent and not long-winded
- Use content from the "History" section of the source text
- Maintain chronological flow and key developments
- Include significant milestones, figures, and breakthroughs
- Preserve important technical terms and concepts
- Synthesize information while maintaining accuracy

APPLICATIONS SECTION:
- The narrative text must be coherent and not long-winded
- Synthesize content from all non-history sections of the source text
- Organize information into coherent themes
- The synthesis still maintains a high level of technical accuracy
- Connect different applications logically
- Emphasize current and emerging use cases

QUALITY REQUIREMENTS:
- DO NOT USE ANY UNNECESSARY CHARACTERS, SYMBOLS, OR PUNCTUATION.
- The length of the article must be within the allowed word count
- The level of detail should be proportional between the two sections
- Ensure factual accuracy based on source material
- Maintain consistent level of technical detail
- Balance breadth and depth of coverage
- Avoid introducing unsupported information
'''

    try:
        client = OpenAI()
        
        # Call OpenAI API for summarization
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _prompt},
                {"role": "user", "content": f"Please USE the following content:\n\n{org_text}"}
            ],
            max_tokens=int(output_len*1.4),
            temperature=0.2,
            seed = 10
        )
        
        # Extract and return the summary
        output = response.choices[0].message.content
        return output
        
    except Exception as e:
        raise Exception(f"Error calling OpenAI API: {str(e)}")
