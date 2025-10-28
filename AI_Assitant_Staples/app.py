# import streamlit as st
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# import os

# # --- Streamlit page setup ---
# st.set_page_config(page_title="🖇️ Staples AI Assistant", page_icon="🖇️")
# st.title("🖇️ Staples AI Assistant")
# st.write("Ask me anything about Staples' product catalog, vendor policy, or customer service FAQs!")

# # --- Sidebar: upload PDFs ---
# st.sidebar.header("📂 Upload Documents")
# uploaded_files = st.sidebar.file_uploader(
#     "Upload PDF documents", type=["pdf"], accept_multiple_files=True
# )

# if uploaded_files:
#     with st.spinner("📄 Processing uploaded documents..."):
#         docs = []
#         for f in uploaded_files:
#             # Save uploaded file temporarily
#             temp_path = f"temp_{f.name}"
#             with open(temp_path, "wb") as temp_file:
#                 temp_file.write(f.read())
            
#             loader = PyPDFLoader(temp_path)
#             docs.extend(loader.load())
#             os.remove(temp_path)  # Clean up temp file

#         # --- Split text into chunks ---
#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         chunks = splitter.split_documents(docs)

#         # --- Create embeddings & store in Chroma ---
#         embeddings = OpenAIEmbeddings()
#         vectorstore = Chroma.from_documents(chunks, embedding=embeddings)

#         # --- Build RAG chain using LCEL ---
#         retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
#         llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        
#         # Create prompt template
#         prompt = ChatPromptTemplate.from_template("""
#         Answer the question based only on the following context:
#         {context}
        
#         Question: {question}
        
#         Answer:
#         """)
        
#         # Format documents function
#         def format_docs(docs):
#             return "\n\n".join(doc.page_content for doc in docs)
        
#         # Create RAG chain using LCEL
#         rag_chain = (
#             {"context": retriever | format_docs, "question": RunnablePassthrough()}
#             | prompt
#             | llm
#             | StrOutputParser()
#         )

#         st.success("✅ Documents processed successfully!")

#         # --- Chat interface ---
#         user_q = st.text_input("💬 Ask a question:")
#         if st.button("Get Answer"):
#             if user_q.strip():
#                 with st.spinner("🤖 Thinking..."):
#                     answer = rag_chain.invoke(user_q)
#                     st.markdown(f"**Answer:** {answer}")
#             else:
#                 st.warning("Please enter a question first.")
# else:
#     st.info("Upload one or more PDFs from the sidebar to begin.")

# # --- Notes for users ---
# st.sidebar.markdown("---")
# st.sidebar.markdown("🔑 **Set your API key first:**")
# st.sidebar.code("setx OPENAI_API_KEY your_key_here", language="bash")

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import chromadb
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Get secrets ---
def get_secret(key, required=True):
    """Get secret from .env or st.secrets"""
    value = os.getenv(key)
    if value:
        return value
    
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except:
        pass
    
    if required:
        st.error(f"⚠️ {key} not found!")
        st.info(f"Add {key} to your .env file")
        st.stop()
    
    return None

# Get API keys
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
CHROMA_API_KEY = get_secret("CHROMA_API_KEY", required=False)
CHROMA_TENANT = get_secret("CHROMA_TENANT", required=False)
CHROMA_DATABASE = get_secret("CHROMA_DATABASE", required=False)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Check if ChromaDB Cloud is configured
USE_CHROMA_CLOUD = CHROMA_API_KEY and CHROMA_TENANT and CHROMA_DATABASE

# --- Streamlit page setup ---
st.set_page_config(page_title="🖇️ Staples AI Assistant", page_icon="🖇️")
st.title("🖇️ Staples AI Assistant")
st.write("Ask me anything about Staples' product catalog, vendor policy, or customer service FAQs!")

# Show configuration
col1, col2 = st.columns(2)
with col1:
    st.success("✓ OpenAI configured")
with col2:
    if USE_CHROMA_CLOUD:
        st.success("✓ ChromaDB Cloud configured")
    else:
        st.warning("⚠ Using local storage")

# --- Initialize ChromaDB Cloud client ---
@st.cache_resource
def get_chroma_client():
    """Initialize ChromaDB Cloud client using CloudClient"""
    if not USE_CHROMA_CLOUD:
        return None
    
    try:
        client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE
        )
        
        # Test connection
        client.heartbeat()
        st.sidebar.success(f"✓ Connected to ChromaDB Cloud")
        st.sidebar.info(f"📊 Tenant: {CHROMA_TENANT}")
        st.sidebar.info(f"🗄️ Database: {CHROMA_DATABASE}")
        return client
        
    except Exception as e:
        st.sidebar.error(f"❌ ChromaDB connection failed: {str(e)}")
        st.sidebar.info("💡 Falling back to local storage")
        return None

# --- Initialize session state ---
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = "staples_docs"

# --- Sidebar ---
st.sidebar.header("📂 Upload Documents")

# Collection name (only for ChromaDB Cloud)
if USE_CHROMA_CLOUD:
    collection_name = st.sidebar.text_input(
        "Collection Name", 
        value=st.session_state.collection_name,
        help="Name for your document collection"
    )
    st.session_state.collection_name = collection_name

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents", 
    type=["pdf"], 
    accept_multiple_files=True
)

# --- Process Documents ---
# --- Process Documents (IMPROVED) ---
if st.sidebar.button("🔄 Process Documents") and uploaded_files:
    with st.spinner("📄 Processing documents..."):
        try:
            # Extract text from PDFs
            docs = []
            for f in uploaded_files:
                temp_path = f"temp_{f.name}"
                with open(temp_path, "wb") as temp_file:
                    temp_file.write(f.read())
                
                loader = PyPDFLoader(temp_path)
                docs.extend(loader.load())
                os.remove(temp_path)

            # IMPROVED: Better chunking for product catalogs
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # Increased from 1000 to capture more context
                chunk_overlap=300,  # Increased overlap to avoid losing info
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                length_function=len,
            )
            chunks = splitter.split_documents(docs)

            # Add metadata for better retrieval
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i
                # Add source filename
                if "source" not in chunk.metadata:
                    chunk.metadata["source"] = "uploaded_document"

            # Create embeddings
            embeddings = OpenAIEmbeddings()
            
            # Choose storage method
            if USE_CHROMA_CLOUD:
                chroma_client = get_chroma_client()
                
                if chroma_client:
                    vectorstore = Chroma(
                        client=chroma_client,
                        collection_name=st.session_state.collection_name,
                        embedding_function=embeddings
                    )
                    vectorstore.add_documents(chunks)
                    storage_msg = f"✅ Stored {len(chunks)} chunks in ChromaDB Cloud"
                else:
                    vectorstore = Chroma.from_documents(
                        chunks, 
                        embedding=embeddings
                    )
                    st.session_state.vectorstore = vectorstore
                    storage_msg = f"✅ Stored {len(chunks)} chunks locally"
            else:
                vectorstore = Chroma.from_documents(
                    chunks, 
                    embedding=embeddings
                )
                st.session_state.vectorstore = vectorstore
                storage_msg = f"✅ Stored {len(chunks)} chunks locally"
            
            # IMPROVED: Better retrieval settings
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5  # Increased from 3 to get more relevant chunks
                }
            )
            
            llm = ChatOpenAI(
                model="gpt-3.5-turbo", 
                temperature=0.2  # Lower temperature for more factual responses
            )
            
            # IMPROVED: Better prompt template
            # IMPROVED: Professional prompt template
            prompt = ChatPromptTemplate.from_template("""
            You are the Staples AI Assistant, an intelligent system designed to help Staples employees quickly access internal knowledge from company documents including product catalogs, vendor policies, customer service FAQs, and operational procedures.

            ROLE & RESPONSIBILITIES:
            - Provide accurate, actionable information from internal Staples documentation
            - Help employees make informed decisions quickly
            - Cite specific document sections, SKUs, policy numbers, and procedures when available
            - Maintain professional, clear communication suitable for business operations

            RESPONSE GUIDELINES:

            1. **ACCURACY FIRST**
            - Use ONLY information explicitly stated in the provided context
            - Never make assumptions or add information not in the documents
            - If information is incomplete, clearly state what's missing

            2. **BE SPECIFIC & ACTIONABLE**
            - Include: Product names, SKU numbers, prices, specifications
            - Include: Policy section numbers, effective dates, contact information
            - Include: Procedure steps, approval requirements, timelines
            - Use exact terminology from the source documents

            3. **STRUCTURE YOUR RESPONSE**
            - Start with a direct answer to the question
            - Use bullet points for lists and multiple items
            - Use numbered steps for procedures
            - Bold key information (product names, prices, important terms)
            - Keep paragraphs concise (2-3 sentences max)

            4. **CONTEXT AWARENESS**
            - Recognize document types (catalog, policy, FAQ, contract)
            - Distinguish between customer-facing and internal information
            - Note when information applies to specific departments or roles

            5. **HANDLE UNCERTAINTY**
            - If the context doesn't contain enough information, say: "Based on the available documents, I cannot find [specific information]. You may need to check [suggest relevant department/resource]."
            - If multiple interpretations exist, present all options clearly
            - If information seems outdated, mention the document date if available

            6. **BUSINESS VALUE**
            - For product queries: Include pricing, availability, specifications
            - For policy queries: Include requirements, exceptions, approval processes
            - For vendor queries: Include contact info, SLAs, compliance requirements
            - For customer service: Include return windows, coverage, escalation paths

            CONTEXT FROM STAPLES INTERNAL DOCUMENTS:
            {context}

            EMPLOYEE QUESTION:
            {question}

            RESPONSE:""")
            
            def format_docs(docs):
                # Better formatting with separators
                formatted = []
                for i, doc in enumerate(docs, 1):
                    formatted.append(f"--- Document {i} ---\n{doc.page_content}")
                return "\n\n".join(formatted)
            
            st.session_state.rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            st.success(storage_msg)
            
            # Show sample chunks for debugging
            with st.expander("🔍 Preview processed chunks"):
                st.write(f"Total chunks: {len(chunks)}")
                st.write("**First chunk preview:**")
                st.code(chunks[0].page_content[:500] + "...")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())

# --- Load existing collection (ChromaDB Cloud only) ---
if USE_CHROMA_CLOUD and st.sidebar.button("📥 Load Existing Collection"):
    with st.spinner("Loading from ChromaDB Cloud..."):
        try:
            chroma_client = get_chroma_client()
            
            if chroma_client:
                embeddings = OpenAIEmbeddings()
                
                vectorstore = Chroma(
                    client=chroma_client,
                    collection_name=st.session_state.collection_name,
                    embedding_function=embeddings
                )
                
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
                
                prompt = ChatPromptTemplate.from_template("""
                Answer the question based only on the following context:
                {context}
                
                Question: {question}
                
                Answer:
                """)
                
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                st.session_state.rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                st.sidebar.success(f"✓ Loaded collection: {st.session_state.collection_name}")
            else:
                st.sidebar.error("Cannot load: ChromaDB Cloud not connected")
                
        except Exception as e:
            st.sidebar.error(f"Error loading collection: {e}")
            with st.sidebar.expander("Error details"):
                st.code(str(e))

# --- List collections (ChromaDB Cloud only) ---
if USE_CHROMA_CLOUD:
    with st.sidebar.expander("📋 View Collections"):
        try:
            chroma_client = get_chroma_client()
            if chroma_client:
                collections = chroma_client.list_collections()
                if collections:
                    st.write("**Available collections:**")
                    for col in collections:
                        st.write(f"- {col.name}")
                else:
                    st.info("No collections found")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Delete collection (ChromaDB Cloud only) ---
if USE_CHROMA_CLOUD and st.sidebar.button("🗑️ Delete Collection"):
    try:
        chroma_client = get_chroma_client()
        if chroma_client:
            chroma_client.delete_collection(name=st.session_state.collection_name)
            st.session_state.rag_chain = None
            st.sidebar.success(f"✓ Deleted collection: {st.session_state.collection_name}")
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# --- Clear local data (local storage only) ---
if not USE_CHROMA_CLOUD and st.session_state.vectorstore and st.sidebar.button("🗑️ Clear Documents"):
    st.session_state.vectorstore = None
    st.session_state.rag_chain = None
    st.rerun()

# --- Status ---
if st.session_state.rag_chain:
    st.sidebar.success("✓ Ready to answer questions!")
else:
    if USE_CHROMA_CLOUD:
        st.sidebar.info("👆 Upload PDFs or load an existing collection")
    else:
        st.sidebar.info("👆 Upload PDFs to get started")

# --- Chat Interface ---
if st.session_state.rag_chain:
    user_q = st.text_input("💬 Ask a question:")
    if st.button("Get Answer"):
        if user_q.strip():
            with st.spinner("🤖 Thinking..."):
                try:
                    answer = st.session_state.rag_chain.invoke(user_q)
                    st.markdown(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question.")
else:
    st.info("📁 Upload and process PDFs to begin.")

# --- Setup instructions ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Setup")

if not USE_CHROMA_CLOUD:
    with st.sidebar.expander("🔧 Enable ChromaDB Cloud"):
        st.markdown("""
        Add to your `.env` file:
```
        CHROMA_API_KEY="your-api-key"
        CHROMA_TENANT="your-tenant-id"
        CHROMA_DATABASE="your-database-name"
```
        """)


st.sidebar.markdown("ℹ️ **About:** AI-powered document search using RAG")