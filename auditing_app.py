"""
Automated Auditing with OpenAI:
This module provides functionalities to automate the auditing process using OpenAI.
It allows the user to upload an annual report, processes the report to extract information,
and facilitates a conversational interface for querying the extracted data.

Author: Group 20
Version: 1.0
"""

import os
import time
import shutil
import tempfile

# Importing necessary modules for environment variable loading
from dotenv import load_dotenv

# Importing libraries for handling document loading, embeddings, and storage
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

# Streamlit for creating the web application interface
import streamlit as st

# Load environment variables
load_dotenv()
persist_directory = os.environ.get('PERSIST_DIRECTORY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def calculate_chunk_size(total_pages, rate_limit=150000, tokens_per_page_estimate=2000):
    """
    Calculate the optimal chunk size for processing.

    Args:
    - total_pages (int): Total number of pages in the PDF.
    - rate_limit (int): API rate limit per minute. Defaults to 150000.
    - tokens_per_page_estimate (int): Estimated tokens per page. Defaults to 2000.

    Returns:
    - int: Calculated chunk size.
    """
    total_tokens_estimate = total_pages * tokens_per_page_estimate
    chunk_size = rate_limit // tokens_per_page_estimate
    return chunk_size


def process_documents(pdf_file_name, vectordb_directory):
    """
    Process and load documents into a vector database.

    Args:
    - pdf_file_name (str): Path to the PDF file to process.
    - vectordb_directory (str): Directory for storing vector data.

    Returns:
    - Chroma: Vector database with the processed documents.
    """

    # Check and clean up any previous database instances
    if persist_directory and os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    # Initialize embeddings and document loader
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    loader = PyPDFLoader(pdf_file_name)
    pages = loader.load_and_split()

    if not pages:
        return "not_parsed"
    else:
        total_pages = len(pages)
        chunk_size = calculate_chunk_size(total_pages)

    # Initialize Chroma for vector storage and set up embedding function
    vectordb = Chroma(persist_directory=vectordb_directory, embedding_function=embeddings)
    
    # Split pages into manageable chunks
    chunks = [pages[i:i + chunk_size] for i in range(0, len(pages), chunk_size)]
    
    # Delay time to respect API rate limits
    delay_time = 30  # Increase this if you're still hitting rate limit

    # Process chunks and handle rate limit exceptions
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1} of {len(chunks)}")
        while True:
            try:
                vectordb.add_documents(chunk)
                vectordb.persist()
                break
            except Exception as e:
                if 'Rate limit reached' in str(e):
                    print(f"Rate limit reached. Error: {e}. Sleeping for {delay_time} seconds.")
                    time.sleep(delay_time)
                else:
                    raise # If it's a different error, raise it
        # Sleep for 30 seconds after processing each chunk
        print(f"Sleeping for 30 seconds after processing chunk {i+1}")
        time.sleep(30)

    return vectordb


def get_conversation_chain(vectorstore):
    """
    Create a retrieval conversation chain.

    Args:
    - vectorstore (Chroma): Vector database with processed documents.

    Returns:
    - ConversationalRetrievalChain: Initialized conversational retrieval chain.
    """
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), 
                                                            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                                                            memory=memory)
    return conversation_chain


def question_answer(query, vectordb, chat_history):
    """
    Retrieve the answer for a given query from the vector database.

    Args:
    - query (str): User query.
    - vectordb (Chroma): Vector database.
    - chat_history (list): History of previous conversations.

    Returns:
    - tuple: Time taken and the AI-generated answer.
    """
    qa = get_conversation_chain(vectordb)
    time.sleep(20)  # Sleep for 20 seconds before processing the query
    start = time.time()
    
    try:
        res = qa(query)
        answer = res['answer']
    except Exception as e:
        if "maximum context length" in str(e):
            raise ValueError("Your query is too long for the model to process. Please shorten your query and try again.")
        else:
            raise e
    
    end = time.time()   
                            
    chat_history.append((query, answer))  # Append the user query and AI response to the chat history
    return (round(end - start, 2),answer)


def main():
    """
    Main function for the Streamlit web application. It provides the interface for 
    uploading annual reports, processing them, and querying the processed data.
    """

    # Session state initialization for keeping track of data across interactions
    if "db" not in st.session_state:
        st.session_state.db = None

    if "file_parsed" not in st.session_state:  
        st.session_state.file_parsed = False  

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.set_page_config(page_title="Auditing with OpenAI", layout="wide")
    st.title("Automated Auditing with OpenAI")
    st.write("Welcome to the automated auditing application. Please follow the steps below:")

    #Step 1:
    # Interface for file upload
    st.header("Step 1: Upload Annual Report")
    uploaded_file = st.file_uploader("Upload your Annual Report for auditing:", type=["pdf"],accept_multiple_files=False)
    if uploaded_file:
        st.success("File uploaded successfully.")

    # Step 2
    # Interface for report processing
    st.header("Step 2: Process Annual Report")
    if st.button("Process Report"):
        if uploaded_file is None:
            st.warning("Please upload an Annual Report first.")
        else:
            # Save uploaded file to temporary directory
            with open(os.path.join(tempfile.gettempdir(), uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process the uploaded file
            with st.spinner("Processing"):
                try:
                    if st.session_state.file_parsed:
                        st.write("File was processed")  # Notify if the file was processed before
                    else:
                        st.session_state.db = process_documents(os.path.join(tempfile.gettempdir(), uploaded_file.name), persist_directory)
                        if st.session_state.db == "not_parsed":
                            st.error("Not able to parse the pdf. Please provide a valid pdf file.")
                            st.session_state.file_parsed = False 
                        else:
                            st.write("File Processed")
                            st.session_state.file_parsed = True 
                except Exception as e:
                    st.error(f"Error occurred during processing: {e}")
                    st.session_state.file_parsed = False 
                finally:
                    # Cleanup: Remove temporary file after processing
                    os.remove(os.path.join(tempfile.gettempdir(), uploaded_file.name))

    # Step 3
    # Interface for asking questions and getting answers
    st.header("Step 3: Ask Questions")
    question_option = st.radio("Choose an option:", ("Standard Questions", "Custom Question"))
    if not st.session_state.file_parsed: 
         st.warning("Please upload an Annual Report and process it to ask questions.")
    else:
        if question_option == "Standard Questions":
            standard_questions = [
                    "16A: Is the entity applying IFRS16 for the first time?",
                    "16B: Has the entity elected to apply the optional recognition exemption related to short-term leases?",
                    "16C: Has the entity elected to apply the optional recognition exemption related to leases of low value?",
                    "16D: Does the entity have a contract that may be a lease or may contain a lease?",
                    "16E: Have any of the terms and conditions of the contract changed?",
                    "16F: Does the contract include lease and non-lease components?",
                    "16G: Is the entity a lessee?",
                    "16H: Has the lease liability changed because of changes in circumstances?",
                    "16I: Has there been a lease modification?",
                    "16J: Has the entity obtained legal title to an underlying asset before that legal title was transferred to the lessor and the asset is leased to the lessee?",
                    "16K: Is the entity a lessor?",
                    "16L: Has the entity entered into a sale and leaseback transaction?",
                    "16M: Has the entity elected to apply the practical expedient to apply this Standard to a portfolio of leases with similar characteristics ?",
                    "16N: Has the entity combined two or more contracts entered into at or near the same time with the same counterparty (or related parties of the counterparty)?",

            ]
            selected_question = st.selectbox("Select a question:", standard_questions)
            if st.button("Submit Standard Question"):
                selected_question = selected_question.split(": ", 1)[-1]
                try:
                    time,answer=question_answer(selected_question,st.session_state.db,st.session_state.chat_history)
                    st.write("Answer:", answer)
                    st.write("Time taken:", time)
                except ValueError as e:  # Catch the value error we raised in the question_answer function
                    st.error(str(e))
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        elif question_option == "Custom Question":
            custom_question = st.text_input("Type your question:")
            if st.button("Submit Custom Question"): 
                if custom_question.strip():
                    try:
                        time,answer=question_answer(custom_question,st.session_state.db,st.session_state.chat_history)
                        st.write("Answer:", answer)
                        st.write("Time taken:", time)
                    except ValueError as e:  # Catch the value error we raised in the question_answer function
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"An error occurred: {e}") 
                else:
                    st.warning("Please enter a question.")
        
        # Display chat history 
        if st.session_state.chat_history:
            st.header("Chat History")
            for user_query, ai_response in st.session_state.chat_history:
                st.write("User:", user_query)
                st.write("AI:", ai_response)
                st.write("---")

            # Button to clear the chat history
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []

    # Display prompting styles for user reference            
    st.write("Checkout the below mentioned prompting styles!")
    st.write("""Prompt Style 1:\n
    Give the name of the company in place of 'entity' in the question followed by - 
    Answer as Yes or No or NA if they have not provided sufficient information. Frame your answer and justify as per their annual financial report of {year}""")
    st.write("""Prompt Style 2:\n
    Please answer the following question using the below context and Annual Report:
    1) {Write your question here}?
    Context:
    A) IFRS 16 Standard: There are four key steps related to accounting standards related to leases. Here is the summary of these steps:
        1. Data Collection and Assessment: The first step is to gather all relevant lease data across the organization. This includes identifying all lease contracts, whether they are operating leases or finance leases. The company must assess the terms and conditions of each lease agreement to determine how they should be accounted for under IFRS 16.
        2. Software and Technology: Implementing IFRS 16 can be complex, especially for companies with a large number of leases. Investing in lease accounting software or technology solutions can greatly streamline the process. These tools help in organizing and managing lease data, performing the necessary calculations, and generating accurate financial reports that comply with the new standard.
        3. Transition Adjustments: With the adoption of IFRS 16, there will be adjustments needed for the company's financial statements. Operating leases that were previously treated as off-balance sheet items will now be recognized on the balance sheet as 'right-of-use' assets and corresponding lease liabilities. Companies must make these transition adjustments to ensure their financial statements are compliant with the new standard.
        4. Training and Communication: Proper training of employees involved in the lease accounting process is essential to ensure they understand the requirements of IFRS 16 and how to implement it correctly. Additionally, clear communication with stakeholders, such as investors and auditors, is crucial to keep them informed about the changes in lease accounting. it impacts the 
    B) Role: Based on the above concepts answer the questions by considering yourself as a Financial Accounting Analyst.
    C) Give your answer as 
        1. Yes, when there is context related to it.
        2. No, when contradicting information is present.
        3. N/A, when no information is present about it in the annual report
    Also, present the explanation for the answered questions.""")


if __name__ == '__main__':
    # main()' function is called, initiating the user interface and starting the application
    main()
