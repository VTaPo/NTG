import wikipediaapi
import re
from pprint import pprint

def get_text_from_leaf_sections(wiki_section, md_text):
    """
    Given a WikipediaPageSection object and a section title,
    this function returns text from all leaf sections.
    O(T + L): where T is the total number of subsections,
    L is the number of leaf sections.
    """
    md_text = md_text
    level = 2
    
    def dfs(_section, lev):
        nonlocal md_text
        # Check if the section has no subsections (leaf node)
        if not _section.sections:
            tt_level = '#' * lev
            md_text += f'''{tt_level} {_section.title}\n{_section.text}\n'''
            lev+=1
        else:
            tt_level = '#' * lev
            md_text += f'''{tt_level} {_section.title}\n'''
            if len(_section.text) > 0:
                md_text += f'''{_section.text}\n'''
            lev+=1
            # Recursively process each subsection
            for subsection in _section.sections:
                dfs(subsection, lev)
    
    dfs(wiki_section, level)
    
    return md_text

def find_matching_sections_regex(sections, pattern):
    """
    Function to find matching sections using regex
    O(k + n*m): where k is the number of keywords,
    n is the number of sections,
    and m is the average length of each section title.
    """
    wikipage_matching_sections = []
    matching_sections = {}
    for section in sections:
        # Use regex search to find if any keyword is in the section title
        if pattern.search(section.title):
            wikipage_matching_sections.append(section)
            matching_sections[section.title] = ''''''
    return matching_sections, wikipage_matching_sections

def get_topic(topic):
    # Initialize Wikipedia object
    wiki = wikipediaapi.Wikipedia(language='en', user_agent="Retriever (vtphongtpt@gmail.com)")
    page = wiki.page(topic)

    # List of keywords to look for in section titles
    keywords = ['history', 'applications', 'tasks']

    # Compile the regex pattern to match any of the keywords (case-insensitive)
    pattern = re.compile(r'(' + '|'.join(re.escape(keyword) for keyword in keywords) + r')',
                        re.IGNORECASE)

    # Retrieve and display matching sections at the top level
    matching_sections = {}
    wikipage_matching_sections = []
    if page.exists():
        matching_sections, wikipage_matching_sections = find_matching_sections_regex(page.sections, pattern)
    else:
        print("Page not found.")
    title = f'''# {page.title.upper()}\n[*]({page.fullurl})\n\n'''
    page_text= ''''''
    for i in range(len(wikipage_matching_sections)):
        key = wikipage_matching_sections[i].title
        matching_sections[key] = get_text_from_leaf_sections(wikipage_matching_sections[i],
                                                            matching_sections[key])
        page_text += matching_sections[key] + '\n'
    return title, page_text

# print(get_topic('Computer vision'))