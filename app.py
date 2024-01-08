from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema.runnable import RunnablePassthrough
import os
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Books Classifications"

OpenAI = ChatOpenAI(model="gpt-4",temperature=0)
Google = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0)

SWEBOK_categories = """Software Requirements
Software Design
Software Construction
Software Testing
Software Maintenance
Software Configuration Management
Software Engineering Management
Software Engineering Process
Software Engineering Models and Methods
Software Quality
Software Engineering Professional Practice
Software Engineering Economics
Computing Foundations
Mathematical Foundations
Engineering Foundations"""

prompt = ChatPromptTemplate.from_messages(
    [
("user","""Your objective is to categorize chapters titles and sections into SWEBOK categories, identified by #### delimiters. Chapters titles and sections are demarcated by triple backticks.

Ensure to adhere to the following instructions:
- Thoroughly analyze each chapter title, section and SWEBOK categories before assigning a label, as accuracy is crucial for my career.
- Avoid generating any additional text, as it may have adverse effects.
- Provide your output in CSV format with only two columns; adding any extra columns may have adverse effects.
Example format:
"There Will Be Code",Software Construction
"Bad Code",Software Quality
"41. Pragmatic Teams",Software Engineering Management

Please note, ensure to enclose the section or title in double quotes before assigning a label to it.
  
> SWEBOK Categories: ####\n{SWEBOK_categories} ####

-----------

> Titles and Sections: ```\n{Titles_and_Sections} ```
""")])

prompt = prompt.partial(SWEBOK_categories=SWEBOK_categories)

chain = prompt | Google | StrOutputParser()

import pandas as pd

def data():

    Titles_and_Sections = []

    if any(file.endswith('.xlsx') for file in os.listdir('data')):

        for xlsx in os.listdir("data"):
            df = pd.read_excel(os.path.join("data",xlsx))

            if "chapter_section_titles" in df.columns:
                print(f"Column was found with the name 'Titles_and_Sections' in the {xlsx} file.")
                Titles_and_Sections.extend(df['chapter_section_titles'].tolist())
                
            else:
                print(f"No column was found with the name 'Titles_and_Sections' in the {xlsx} file.")
    else:
        print("No Excel file found in Data Directory!")

    return Titles_and_Sections
    
def chunking(items):

    chunk_size = 30

    return [{"Titles_and_Sections":"\n".join(items[i:i + chunk_size])} for i in range(0, len(items), chunk_size)]

title_and_sections = RunnablePassthrough() | chunking | chain.map()

print("Please wait, it may take some time to finish")
output = "\n".join(title_and_sections.invoke(data()))

import pandas as pd
import io
import csv

csv_reader = csv.reader(io.StringIO(output))
data = list(csv_reader)
df = pd.DataFrame(data, columns=["Titles_and_Sections", "SWEBOK_categories"]).iloc[1:]

xlsx_filename = "label_data.xlsx"

with pd.ExcelWriter(xlsx_filename, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name='Sheet1')

    workbook  = writer.book
    worksheet = writer.sheets['Sheet1']

    for i, col in enumerate(df.columns):
        max_len = max(df[col].astype(str).apply(len).max(), len(col))
        worksheet.set_column(i, i, max_len + 2)

print(f"Excel file '{xlsx_filename}' created successfully.")