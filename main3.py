GEMINI_API_KEY = "AIzaSyB3aQ9m_TI-Cch6u1ecW7oOlFPYOCqPeow"  # Replace with your actual API key


import streamlit as st 
import os
import json
import pdfplumber
import pandas as pd
import camelot
import numpy as np
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from google import genai
from google.genai.types import EmbedContentConfig

# EXCEL_INPUT_FILE = "volex_queries.xls"

prompt_template = """
You are a Financial Analyst, and you have to split a question into 2 or more questions if the following conditions are met:
1) The question contains more than one Financial KPI or metric
2) If the question involves comparison of two values followed by an arithmetic operation
If these conditions are not met, DO NOT split the question and return the original question without any changes.
Your answer should only consist of the question/split questions, not anything else.
Example 1
Question: What are the Total Current Liabilities as % of Total Liabilities as of 31st March 2024?
Here the KPIs are "Total Current Liabilities" and "Total Liabilities", and a "%" operation is being done, therefore the split questions are:
q1- What are the Total Current Liabilities as of 31st March 2024?
q2- What are the Total Liabilities as of 31st March 2024?

Example 2
Question: What is the Profit After tax as of 31st Mar 2024?
Since here, only 1 KPI/metric is present, that is "Profit After tax", hence there will be no splitting performed, original question to be returned:
q- What is the Profit After tax as of 31st Mar 2024?

Now, using this logic, answer the following question: 
Question : {query}
"""

# Constants
top_k = 3
top_k1 = 5

def clean_chunks(chunks):
    return [chunk.strip() for chunk in chunks if isinstance(chunk, str) and chunk.strip()]

# Function to process Excel to JSON (move this definition before any calls to it)
def process_excel_to_json(excel_file_path):
    try:
        excel_data = pd.ExcelFile(excel_file_path)
        json_data = {}
        for sheet_name in excel_data.sheet_names:
            df = excel_data.parse(sheet_name)
            json_data[sheet_name] = df.to_dict(orient="records")
        return json_data
    except Exception as e:
        print(f"Error processing Excel file '{excel_file_path}': {e}")
        return {}

# Initialize Streamlit App
st.title("Financial Table Extraction using Google Embeddings")

# df_queries = pd.read_excel(EXCEL_INPUT_FILE)

# File upload
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Initialize Google Generative AI Client
client = genai.Client(api_key=GEMINI_API_KEY)

if uploaded_files:
    os.makedirs("temp", exist_ok=True)

    # Save uploaded PDFs
    pdf_paths = {}
    for uploaded_file in uploaded_files:
        pdf_path = os.path.join("temp", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        pdf_paths[uploaded_file.name] = pdf_path

    selected_pdf = st.selectbox("Select a PDF:", list(pdf_paths.keys()))

    if selected_pdf:
        pdf_path = pdf_paths[selected_pdf]
        base_name = Path(selected_pdf).stem
        EXCEL_FILE_STREAM = f"temp/{base_name}_stream.xlsx"
        EXCEL_FILE_LATTICE = f"temp/{base_name}_lattice.xlsx"
        JSON_FILE_STREAM = f"temp/{base_name}_stream.json"
        JSON_FILE_LATTICE = f"temp/{base_name}_lattice.json"
        EMBEDDINGS_CACHE = f"temp/{base_name}_embeddings.json"

        # Query input
        query = st.text_input(label="Enter a query: ")

        # Add a custom submit button positioned at the top-right corner
        submit_button = st.button('Submit', key='submit_button')

        def process_question(query):
            try:
                final_prompt = prompt_template.format(query=query)
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=final_prompt
                )
                if isinstance(response.text, str):
                    split_questions = [q.strip() for q in response.text.split("\n") if q.strip()]
                else:
                    split_questions = [response.text]
                return split_questions
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                return ["ERROR"]

        import asyncio

        def generate_google_embeddings(contents):
            """
            Generate embeddings using Google Generative AI, handling batch size limits.
            """
            if not contents or not isinstance(contents, list) or all(not content.strip() for content in contents):
                raise ValueError("Contents must be a non-empty list of non-empty strings.")

            try:
                batch_size = 100  # Maximum allowed batch size by the API
                embeddings = []

                # Split contents into batches of 100
                for i in range(0, len(contents), batch_size):
                    batch = contents[i:i + batch_size]
                    response = client.models.embed_content(
                        model="text-embedding-004",
                        contents=batch,
                        config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                    )
                    embeddings.extend([embedding.values for embedding in response.embeddings])

                return embeddings
            except Exception as e:
                print(f"Error generating embeddings: {e}")
                return []

        if not os.path.exists(JSON_FILE_STREAM) or not os.path.exists(JSON_FILE_LATTICE):
            st.write("Processing PDF and extracting tables...")

            with pd.ExcelWriter(EXCEL_FILE_STREAM, engine='xlsxwriter') as writer:
                stream_tables = camelot.read_pdf(pdf_path, flavor="stream", pages='all')
                for i, table in enumerate(stream_tables):
                    table.df.to_excel(writer, sheet_name=f"Stream_Table_{i+1}", index=False, header=False)

            with pd.ExcelWriter(EXCEL_FILE_LATTICE, engine='xlsxwriter') as writer:
                lattice_tables = camelot.read_pdf(pdf_path, flavor="lattice", pages='all')
                for i, table in enumerate(lattice_tables):
                    table.df.to_excel(writer, sheet_name=f"Lattice_Table_{i+1}", index=False, header=False)

            data = process_excel_to_json(EXCEL_FILE_STREAM)
            with open(JSON_FILE_STREAM, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            data1 = process_excel_to_json(EXCEL_FILE_LATTICE)
            with open(JSON_FILE_LATTICE, 'w', encoding='utf-8') as f:
                json.dump(data1, f, indent=4, ensure_ascii=False)

        # Check if embeddings already exist
        if os.path.exists(EMBEDDINGS_CACHE):
            with open(EMBEDDINGS_CACHE, 'r', encoding='utf-8') as f:
                cached_embeddings = json.load(f)
                table_chunks1 = cached_embeddings.get("stream_tables", [])
                table_chunks2 = cached_embeddings.get("lattice_tables", [])
                docs = cached_embeddings.get("docs", [])
        else:
            with open(JSON_FILE_STREAM, 'r', encoding='utf-8') as f:
                json_data1 = json.load(f)
            table_chunks1 = [json.dumps(table) for table in json_data1.values()]

            with open(JSON_FILE_LATTICE, 'r', encoding='utf-8') as f:
                json_data2 = json.load(f)
            table_chunks2 = [json.dumps(table) for table in json_data2.values()]

            with pdfplumber.open(pdf_path) as pdf:
                document_texts = "**".join([page.extract_text() for page in pdf.pages if page.extract_text()])

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=40,
                length_function=len,
                is_separator_regex=False
            )
            docs = text_splitter.split_text(document_texts)

            # Clean all chunks
            docs = clean_chunks(docs)
            table_chunks1 = clean_chunks(table_chunks1)
            table_chunks2 = clean_chunks(table_chunks2)

            # Cache the embeddings
            with open(EMBEDDINGS_CACHE, 'w', encoding='utf-8') as f:
                json.dump({
                    "stream_tables": table_chunks1,
                    "lattice_tables": table_chunks2,
                    "docs": docs
                }, f, indent=4)

        # Add basic Streamlit checks for visibility
        if not docs:
            st.error("No valid text chunks found in the PDF.")
            st.stop()
        if not table_chunks1:
            st.warning("No valid stream tables extracted.")
        if not table_chunks2:
            st.warning("No valid lattice tables extracted.")

        def search_google_embeddings(query, chunks, top_k):
            """
            Search for the most relevant chunks using Google embeddings.
            """
            chunks = clean_chunks(chunks)  # Ensure cleaned chunks
            if not chunks:
                print(f" No valid chunks to search for query: {query}")
                return []

            try:
                # Generate query embedding
                query_embedding = generate_google_embeddings([query])
                if not query_embedding:
                    print(f" No embedding generated for query: {query}")
                    return []

                query_embedding = query_embedding[0]  # Safely access the first element

                # Generate chunk embeddings
                chunk_embeddings = generate_google_embeddings(chunks)
                if not chunk_embeddings:
                    print(f" No embeddings generated for chunks. Skipping search for: {query}")
                    return []

                # Compute cosine similarity
                similarities = np.dot(chunk_embeddings, query_embedding)
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                return [chunks[i] for i in top_indices]

            except Exception as e:
                print(f"Error during embedding search for query '{query}': {e}")
                return []


        # Initialize an empty DataFrame to store all query results
        all_responses = pd.DataFrame()

        if query:
            st.write("User Question: ", query)
            query1 = process_question(query)
            for q in query1:
                try:
                    # Clear previous results for each query
                    retrieved_table_rows_stream = []
                    retrieved_table_rows_lattice = []
                    retrieved_text_chunks = []

                    # Perform searches
                    retrieved_table_rows_stream.extend(search_google_embeddings(q, table_chunks1, top_k1))
                    retrieved_table_rows_lattice.extend(search_google_embeddings(q, table_chunks2, top_k1))
                    retrieved_text_chunks.extend(search_google_embeddings(q, docs, top_k))

                    # Combine results
                    combined_results = retrieved_table_rows_stream + retrieved_table_rows_lattice + retrieved_text_chunks

                    if not combined_results:
                        st.warning(f"No relevant data found for query: {q}")
                        continue
                except Exception as e:
                    st.error(f"Error processing query '{q}': {e}")

            # Generate the final prompt for the LLM
            final_prompt = f'''
            You are a Financial analyst. The input given to you is multiple chunks of data based on one or more financial statements.
            The chunks are retrieved from vector store and Given to you.
            Understand the query (user question), co-relate with the chunk's content and provide your answer
            Rules for answering:
            1. The answer should be based on the chunk's content only. Understand the chunk's content, relate them semantically and answer your question. Do not provide answer from your knowledge. 
            2. If the chunk is a list of strings of dict, it may be table values, rest all are text string values. Co-relate them and answer. 
            3. Do not provide the steps to arrive the answer or your understanding on the question or the context (chunks)
            4. If asked for comparison, use basic arithmetic operations. Use your reasoning to answer the query.
            5. Output should be only the answer to the question.

            Context: {combined_results}\nQuestion: {query}\n
            '''

            # Get the response from the LLM
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=final_prompt
            )
            st.write(response.text)
            # Prepare the data to save
            response_data = {
                "Query": [query],
                #"Retrieved Chunks": [json.dumps(combined_results)],  # Convert list to JSON string for safe storage
                "Response": [response.text]
            }
            response_df = pd.DataFrame(response_data)  # Convert to a DataFrame

            # Append the current response to the main DataFrame
            all_responses = pd.concat([all_responses, response_df], ignore_index=True)

        # Save the final DataFrame to the CSV file after processing all queries
        output_file = "volex.csv"
        all_responses.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        st.success(f"All results saved to {output_file}")

