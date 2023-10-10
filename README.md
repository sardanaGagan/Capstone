# Capstone - Automated Auditing with OpenAI

## Version: 1.0

## Introduction
This application seeks to revolutionize the auditing process by utilizing OpenAI's advanced language models. Traditional auditing is often time-consuming and can be prone to human error. This tool automates that process, making it swift and accurate. By allowing users to upload annual reports, it extracts crucial information and facilitates a conversational interface for intuitive data querying.

## Features
1. File Upload and Processing: An intuitive interface allows users to upload annual reports in PDF format. The processing ensures data extraction and preparation for auditing analysis.
2. Conversational Interface: Leveraging the power of OpenAI, this tool offers an interactive querying system. Users can engage in natural language conversations to extract financial insights.
3. Standard and Custom Questions: Apart from a set of standard auditing questions, users can also pose custom questions, offering a wider breadth of report analysis.
4. Chat History: With the chat history feature, every query and its response are recorded, providing users a clear trail of their auditing process.

## How It Works:

1. File Upload and Processing:
   - Users start by uploading an annual financial report in PDF format through the web interface.
   - The module uses PyPDFLoader to load and split the document into individual pages for further processing.
   - It calculates the optimal chunk size to ensure efficient processing while respecting OpenAI's API rate limits.

2. Data Extraction and Vector Storage:
   - The module leverages OpenAIEmbeddings to convert the textual data from the report into vector embeddings.
   - These embeddings represent the essential information present in the document.
   - The embeddings are stored in a Chroma vector database for efficient retrieval and analysis.

3. Conversational Retrieval Chain:
   - The vector database, Chroma, is used to create a ConversationalRetrievalChain using OpenAI's language model (LLM).
   - The chain allows the application to have a conversational interface for querying the extracted data.
   - The chain is designed to retrieve relevant information from the vector database based on user queries.

4. Question and Answer:
   - Users can ask questions related to the financial report using either standard or custom questions.
   - Standard questions are predefined and cover common auditing aspects, while custom questions allow users to ask specific queries.
   - The module takes the user's question, processes it through the conversational retrieval chain, and returns an AI-generated answer based on the extracted data.

5. Chat History and Interaction:
   - The application maintains a chat history that records user queries and AI-generated responses for reference.
   - Users can interact with the AI repeatedly, asking multiple questions and receiving answers in real-time.
   - The chat history provides transparency and allows users to track their interactions with the AI during the auditing process.

## Dependencies and Installation
To use this app, you need to install the required dependencies and set up environment variables. Make sure you have Python 3.x installed on your system.
1. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```
2. Set up environment variables:
    
    PERSIST_DIRECTORY=<Directory for storing vector data during processing>
    OPENAI_API_KEY=<Your OpenAI API key>

## Usage
-----
To use the app, please follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file.

2. Run the `auditing_app.py` file using the Streamlit CLI. Execute the following command:
   ```
   python -m streamlit run auditing_app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Follow the on-screen instructions to interact with the application. First, upload an annual report   for processing, then process the report, and finally, ask questions related to the financial data.

## Troubleshooting
-----
1. If you encounter API rate limits during processing, the application is designed to respect these limits and will automatically delay processing. Please be patient during this period.
2. Ensure the .env file is correctly set up in the root directory.

## Acknowledgments
-----
Special thanks to OpenAI for providing the powerful language models and Streamlit for their intuitive web app framework.

