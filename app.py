from langchain.schema.runnable import RunnablePassthrough,RunnableParallel
from langchain_core.runnables import ConfigurableField
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from collections import defaultdict
import pandas as pd,streamlit as st
import os
# from dotenv import load_dotenv
# load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Books Classifications"

OpenAI = ChatOpenAI(model="gpt-4",temperature=0.2).configurable_fields(
    temperature=ConfigurableField(
        id="temperature",
        name="LLM Temperature",
        description="The temperature of the LLM model",
    )
).with_fallbacks([ChatOpenAI(model="gpt-3.5-turbo")])

import tempfile

def pack_to_excel(response, excel_data):
    Chapter_Title = []

    for index, entry in excel_data[:200].iterrows():
        if entry["Type"] == "Head":
            Chapter_Title.append(entry["Titles"])

    data = {
        "Chapter_Title": Chapter_Title,
        "Summary": response["Summary"],
        "SWEBOK_Area_Category": response["SWEBOK_Area_Category"],
        "Primary_SWEBOK_Area_Percentage": response["Primary_SWEBOK_Area_Percentage"]
    }
    
    df = pd.DataFrame(data)

    # Create a temporary Excel file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
        temp_file_path = temp_file.name
        writer = pd.ExcelWriter(temp_file_path, engine='openpyxl')
        df.to_excel(writer, index=False)
        
        worksheet = writer.sheets['Sheet1']
        
        for col in worksheet.columns:
            max_length = 0
            for cell in col:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            adjusted_width = max_length + 2  # Adding a little extra space
            worksheet.column_dimensions[col[0].column_letter].width = adjusted_width

        writer.close()

        st.info("Output Data from the first some rows:")
        st.dataframe(df)

        st.download_button(
            label="Download Excel file",
            data=open(temp_file_path, "rb"),
            file_name="output_data.xlsx",
            mime="application/vnd.ms-excel")

from utils import summary_few_short_examples

SummaryPromptPrompt = ChatPromptTemplate.from_messages([("user",summary_few_short_examples[0]),("ai",summary_few_short_examples[1]),("user","""Your task is to generate a comprehensive summary for each section of the book '{Book}', specifically from chapter '{Chapter}'. These sections are delimited with ```. The sections belong to chapter '{Chapter}' of the book '{Book}'. Emphasize the key points and insights of each section. Present your output by starting with the chapter name as the title, followed by a concise introduction to the chapter, and then a summary of each section. For clarity, use the section names as titles for their respective summaries. Also, provide a justification for your assignment for each section summary. Please ensure high-quality work as this is crucial to my career.

----------------

Sections: ```{Sections}```

Also, justify your assignment for each section summary.""")])

SWEBOK_Area_CategoryChain = PromptTemplate.from_template("""Your task is to classify a particular book chapter each section into SWEBOK knowledge areas. For the book '{Book}', chapter '{Chapter}', and sections delimited with XML tag, please classify each section into the most relevant SWEBOK knowledge areas and also provide your justification of each categorization.
The SWEBOK knowledge areas are delimited with ```.
Provide your output, including each section title along with the corresponding SWEBOK knowledge area, and provide justification. For example:
1. Section Title: Data Abstraction
   - SWEBOK Area Software Design
   - Justification: This section delves into the concept of data abstraction and its significance in crafting clean code. It aligns with the principles of software design by emphasizing the encapsulation and interaction of data.

Please do your best it is very important to my career.
                                                              
----------------
                                                         
<sections>: {Sections} </sections>                                                      
                                                              
```
- Software Requirements
- Software Design
- Software Construction
- Software Testing
- Software Maintenance
- Software Configuration Management
- Software Engineering Management
- Software Engineering Process
- Software Engineering Models and Methods
- Software Quality
- Software Engineering Professional Practice
- Software Engineering Economics
- Computing Foundations
- Mathematical Foundations
- Engineering Foundations```

Your expertise is vital for accurately classifying each section into one or more SWEBOK areas, along with providing justification.""")

Primary_SWEBOK_Area_PercentagePrompt = ChatPromptTemplate.from_template("""Please identify the SWEBOK knowledge areas that the book '{Book}', particularly chapter '{Chapter}', primarily focuses on. Additionally, determine the secondary SWEBOK areas associated with this chapter. Provide the percentage prominence for each of these primary and secondary areas, along with a justification for your assessment.Please do your best it is very important to my career.""")

SummaryPromptChain = SummaryPromptPrompt | OpenAI | StrOutputParser()
SWEBOK_Area_CategoryChain = SWEBOK_Area_CategoryChain | OpenAI | StrOutputParser()
Primary_SWEBOK_Area_PercentageChain = Primary_SWEBOK_Area_PercentagePrompt | OpenAI | StrOutputParser()

def chunking(data):

    lst_data = []

    current_book = None
    current_chapter = None

    for index,row in data.iterrows():
        if row["Type"] == "Head":
            current_book = row["Book"]
            current_chapter = row["Titles"]
            
        elif row["Type"] == "Section" and current_book and current_chapter:
            section = row['Titles']
            data = {
                "Book":current_book,
                "Chapter":current_chapter,
                "Section":section
            }
            lst_data.append(data)

    return lst_data[:200]

def merge_sections(data):
    merged_data = defaultdict(lambda: defaultdict(list))

    for entry in data:
        merged_data[entry['Book']][entry['Chapter']].append(entry['Section'])

    list_of_dicts = [
        {'Book': book, 'Chapter': chapter, 'Sections': "\n".join(sections)}
        for book, chapters in merged_data.items()
        for chapter, sections in chapters.items()
    ]
    return list_of_dicts

def extract_chapter(data):
    seen = set()
    unique_data = []

    for item in data:
        identifier = (item["Book"],item["Chapter"])
        if identifier not in seen:
            seen.add(identifier)
            unique_data.append(item)

    return [{"Book":item["Book"],"Chapter":item["Chapter"]} for item in unique_data]

main_chain = RunnableParallel(
Summary = RunnablePassthrough() | merge_sections | SummaryPromptChain.map().with_config(configurable={"temperature":0.8}),
SWEBOK_Area_Category = RunnablePassthrough() | merge_sections | SWEBOK_Area_CategoryChain.map(),
Primary_SWEBOK_Area_Percentage = RunnablePassthrough() | extract_chapter | Primary_SWEBOK_Area_PercentageChain.map())

st.set_page_config(page_title="Books Classifications", page_icon=":books:", layout="wide")

st.title("Books Classifications")

example_data = {
    'Book': ['Clean Code', 'Clean Code', 'Clean Code', 'Clean Code', 'Clean Code'],
    'Titles': [
        'Chapter 1: Clean Code',
        'There Will Be Code',
        'Bad Code',
        'The Total Cost of Owning a Mess',
        'The Grand Redesign in the Sky'
    ],
    'Chapter': ['Chapter 1', 'Chapter 1', 'Chapter 1', 'Chapter 1', 'Chapter 1'],
    'Type': ['Head', 'Section', 'Section', 'Section', 'Section']
}


example_data = pd.DataFrame(example_data)

st.info("Please Make Sure Your Excel File Format Should Look Like This: ")

st.write(example_data)

file_name = st.file_uploader("Upload your excel file", type=["xlsx"])
if st.button("Submit"):
    if file_name is not None:
        excel_data = pd.read_excel(file_name)

        try:
            data = chunking(excel_data)

            try:
                with st.spinner("Please wait while we process your data..."):
                    st.session_state.response = main_chain.invoke(data)

                st.success("Data Processed Successfully")

                pack_to_excel(st.session_state.response, excel_data)

            except Exception as e:
                st.error(f"Something went wrong, please try again. {e}")

        except Exception as e:
            st.warning("Incorrect File Format! Please Make Sure Your Excel File Format Should Look Like This with 200 rows: ")
            st.dataframe(example_data)
