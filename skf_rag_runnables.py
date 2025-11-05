import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import warnings


# Suppress LangChain deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*deprecated.*')


# Updated LangChain v0.1+ imports with langchain_core.runnables
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Tuple, Dict, Any


# ============================================================================
# CONFIGURATION
# ============================================================================
# CSV_FILES = ['SKR-data-0.csv', 'SKF-data-1.csv']
CSV_FILES = ['cleaned_data.csv']
MISTRAL_API_KEY = '9Lwb0Sbr6L5bqEZxl7PJxdfpTNIzsm2R'
MISTRAL_MODEL = 'mistral-small-latest'


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
@st.cache_data
def load_bearing_data(files):
    """Load and combine CSV files"""
    dfs = [pd.read_csv(f) for f in files]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.dropna(subset=['Designation'])
    return combined_df


def create_bearing_documents(df: pd.DataFrame) -> List[Document]:
    """Convert bearing data to LangChain Documents"""
    documents = []
    for idx, row in df.iterrows():
        # Create rich content for each bearing
        content = f"""
Designation: {row['Designation']}
Category: {row['Category']}
Type: {row.get('Short_Description', '')}


Specifications:
- Bore Diameter: {row['Bore diameter (mm)']} mm
- Outside Diameter: {row['Outside diameter (mm)']} mm
- Width: {row.get('Width (mm)', 'N/A')} mm
- Dynamic Load Rating: {row.get('Basic dynamic load rating (kN)', 'N/A')} kN
- Static Load Rating: {row.get('Basic static load rating (kN)', 'N/A')} kN
- Speed Limit: {row.get('Limiting speed (r/min)', 'N/A')} r/min
- Weight: {row.get('Product net weight (kg)', 'N/A')} kg
- Material: {row.get('Material, bearing', 'N/A')}
- Sealing: {row.get('Sealing', 'N/A')}
- Number of Rows: {row.get('Number of rows', 'N/A')}


Benefits:
{row.get('Benefits', 'N/A')}
        """.strip()
        
        metadata = {
            'designation': row['Designation'],
            'category': row['Category'],
            'bore_diameter': float(row['Bore diameter (mm)']),
            'outside_diameter': float(row['Outside diameter (mm)']),
            'dynamic_load': float(row.get('Basic dynamic load rating (kN)', 0)),
            'static_load': float(row.get('Basic static load rating (kN)', 0)),
            'speed_limit': float(row.get('Limiting speed (r/min)', 0)),
            'photo_url': row.get('Photo_URL', ''),
            'long_description': row.get('Long_Description', ''),
            'index': idx
        }
        
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    return documents


# ============================================================================
# EMBEDDING AND VECTOR STORE SETUP
# ============================================================================
@st.cache_resource
def setup_vector_store(documents: List[Document]):
    """Create LangChain vector store using FAISS"""
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vector_store = LangChainFAISS.from_documents(documents, embeddings)
    return vector_store, embeddings


# ============================================================================
# LANGCHAIN RAG SETUP WITH RUNNABLES
# ============================================================================
@st.cache_resource
def setup_langchain_rag(api_key: str):
    """Initialize LangChain components with updated imports"""
    
    # Initialize Mistral LLM using new import
    llm = ChatMistralAI(
        api_key=api_key,
        model=MISTRAL_MODEL,
        temperature=0.3,
        max_tokens=800
    )
    
    return llm


def create_bearing_expert_chain(llm: ChatMistralAI):
    """Create LangChain chain using langchain_core.runnables (modern approach)"""
    
    # System prompt template
    system_template = """You are an expert SKF bearing specialist with deep knowledge of industrial bearings.


Your role is to:
1. Understand what the user is looking for (their application, constraints, preferences)
2. Analyze the retrieved bearing products from the database
3. Translate the user's natural language query into technical bearing requirements
4. Recommend the most suitable bearings and explain WHY they match the user's needs
5. Provide professional, catalog-style recommendations


When analyzing:
- Look for speed requirements (high-speed, low-speed, etc.)
- Consider load capacity (radial vs axial vs combined)
- Note the application type implied by the user
- Evaluate bore diameter, space constraints
- Consider operating environment (temperature, moisture, etc.)


Format your response professionally with clear sections for each recommended bearing.
Always explain the technical reasoning behind your recommendations."""


    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    
    # Human prompt template
    human_template = """User Query: {user_query}


Here are the top bearing products from our database that match the search:


{context}


Please:
1. Briefly identify what the user is looking for in technical terms
2. Analyze each bearing's suitability
3. Recommend the top 3-5 most suitable bearings with detailed explanations
4. For each recommendation, explain how it addresses the user's requirements
5. Highlight any tradeoffs or considerations


Format with clear sections and bold the bearing designations."""


    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    # Combine into chat prompt
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    
    # Create runnable chain using | pipe operator (modern LangChain approach)
    chain = chat_prompt | llm
    
    return chain


def create_bearing_expert_runnable(llm: ChatMistralAI):
    """
    Alternative: Create a more complex runnable with custom processing.
    Shows advanced Runnable patterns.
    """
    
    system_template = """You are an expert SKF bearing specialist with deep knowledge of industrial bearings.

Your role is to:
1. Understand what the user is looking for (their application, constraints, preferences)
2. Analyze the retrieved bearing products from the database
3. Translate the user's natural language query into technical bearing requirements
4. Recommend the most suitable bearings and explain WHY they match the user's needs
5. Provide professional, catalog-style recommendations

When analyzing:
- Look for speed requirements (high-speed, low-speed, etc.)
- Consider load capacity (radial vs axial vs combined)
- Note the application type implied by the user
- Evaluate bore diameter, space constraints
- Consider operating environment (temperature, moisture, etc.)

Format your response professionally with clear sections for each recommended bearing.
Always explain the technical reasoning behind your recommendations."""

    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    
    human_template = """User Query: {user_query}

Here are the top bearing products from our database that match the search:

{context}

Please:
1. Briefly identify what the user is looking for in technical terms
2. Analyze each bearing's suitability
3. Recommend the top 3-5 most suitable bearings with detailed explanations
4. For each recommendation, explain how it addresses the user's requirements
5. Highlight any tradeoffs or considerations

Format with clear sections and bold the bearing designations."""

    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    
    # Create runnable chain with pipe operator
    # This is equivalent to: chain = chat_prompt | llm
    runnable_chain = chat_prompt | llm
    
    return runnable_chain


# ============================================================================
# SEARCH AND FILTERING
# ============================================================================
def search_bearings(query, index, df, model, k=10, filters=None):
    """Search for bearings matching the query with optional filters"""
    
    # Create query embedding
    query_emb = model.embed_query(query)
    
    # Search in FAISS index
    D, I = index.search(np.array([query_emb]), min(k * 3, len(df)))
    
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1 or idx >= len(df):
            continue
        
        row = df.iloc[idx]
        
        # Apply filters if provided
        if filters:
            if filters.get('category') and row['Category'] != filters['category']:
                continue
            if filters.get('min_bore') and row['Bore diameter (mm)'] < filters['min_bore']:
                continue
            if filters.get('max_bore') and row['Bore diameter (mm)'] > filters['max_bore']:
                continue
            if filters.get('min_load') and row.get('Basic dynamic load rating (kN)', 0) < filters['min_load']:
                continue
        
        results.append((idx, row, float(dist)))
        
        if len(results) >= k:
            break
    
    return results


# ============================================================================
# STREAMLIT UI
# ============================================================================
def display_bearing_card(row, rank):
    """Display a single bearing in a nice card format"""
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Display image if available
            if pd.notna(row.get('Photo_URL')):
                try:
                    st.image(row['Photo_URL'], width=150)
                except:
                    st.write("🔧")
            else:
                st.write("🔧")
        
        with col2:
            st.markdown(f"### {rank}. {row['Designation']}")
            st.markdown(f"**{row['Category']}**")
            st.caption(row.get('Short_Description', ''))
            
            # Key specs in columns
            spec_col1, spec_col2, spec_col3 = st.columns(3)
            with spec_col1:
                st.metric("Bore Ø", f"{row['Bore diameter (mm)']} mm")
            with spec_col2:
                st.metric("Outer Ø", f"{row['Outside diameter (mm)']} mm")
            with spec_col3:
                load = row.get('Basic dynamic load rating (kN)', 'N/A')
                st.metric("Load Rating", f"{load} kN")
            
            # Additional specs
            with st.expander("📋 More Details"):
                st.write(f"**Width:** {row.get('Width (mm)', 'N/A')} mm")
                st.write(f"**Speed Limit:** {row.get('Limiting speed (r/min)', 'N/A')} r/min")
                st.write(f"**Weight:** {row.get('Product net weight (kg)', 'N/A')} kg")
                st.write(f"**Material:** {row.get('Material, bearing', 'N/A')}")
                st.write(f"**Sealing:** {row.get('Sealing', 'N/A')}")
                if pd.notna(row.get('Benefits')):
                    st.write(f"**Benefits:** {row['Benefits']}")
        
        st.divider()


def main():
    # Page config
    st.set_page_config(
        page_title="SKF Bearing Search",
        page_icon="⚙️",
        layout="wide"
    )
    
    # Title and description
    st.title("⚙️ SKF Bearing Intelligent Search (LangChain + Runnables)")
    st.markdown("""
    **Search for bearings using natural language!** 
    
    Try queries like:
    - "I need a small bearing for high-speed applications"
    - "Show me bearings with 10mm bore diameter"
    - "Heavy duty bearings for industrial machinery"
    - "Bearings that can handle both radial and axial loads"
    """)
    
    # Load data
    with st.spinner("Loading bearing database..."):
        df = load_bearing_data(CSV_FILES)
        documents = create_bearing_documents(df)
        vector_store, embeddings = setup_vector_store(documents)
        llm = setup_langchain_rag(MISTRAL_API_KEY)
        chain = create_bearing_expert_chain(llm)
        
        # Build FAISS index from embeddings
        texts = [doc.page_content for doc in documents]
        embeddings_arr = embeddings.embed_documents(texts)
        embeddings_arr = np.array(embeddings_arr).astype('float32')
        index = faiss.IndexFlatL2(embeddings_arr.shape[1])
        index.add(embeddings_arr)
    
    st.success(f"✅ Loaded {len(df)} bearing products")
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Category filter
        categories = ['All'] + sorted(df['Category'].unique().tolist())
        selected_category = st.selectbox("Bearing Type", categories)
        
        # Bore diameter filter
        st.subheader("Bore Diameter (mm)")
        bore_range = st.slider(
            "Range",
            float(df['Bore diameter (mm)'].min()),
            float(df['Bore diameter (mm)'].max()),
            (float(df['Bore diameter (mm)'].min()), float(df['Bore diameter (mm)'].max()))
        )
        
        # Load rating filter
        st.subheader("Min Load Rating (kN)")
        min_load = st.number_input(
            "Minimum",
            min_value=0.0,
            max_value=float(df['Basic dynamic load rating (kN)'].max()),
            value=0.0,
            step=1.0
        )
        
        # Number of results
        num_results = st.slider("Number of Results", 5, 20, 10)
    
    # Main search interface
    query = st.text_input(
        "🔎 What kind of bearing are you looking for?",
        placeholder="e.g., 'small bearing for electric motor shaft'",
        key="search_query"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🚀 Search", type="primary", use_container_width=True)
    with col2:
        use_ai = st.checkbox("Use AI-powered recommendations", value=True)
    
    if search_button and query:
        # Prepare filters
        filters = {
            'category': selected_category if selected_category != 'All' else None,
            'min_bore': bore_range[0],
            'max_bore': bore_range[1],
            'min_load': min_load
        }
        
        with st.spinner("Searching bearing catalog..."):
            # Search for bearings
            results = search_bearings(query, index, df, embeddings, k=num_results, filters=filters)
        
        if not results:
            st.warning("No bearings found matching your criteria. Try adjusting the filters.")
            return
        
        st.success(f"Found {len(results)} matching bearings")
        
        # AI-powered recommendations
        if use_ai and MISTRAL_API_KEY != 'your_mistral_api_key_here':
            with st.spinner("🤖 AI is analyzing your requirements..."):
                try:
                    # Prepare context
                    context_text = "\n\n".join([
                        f"{i}. {result[1]['Designation']} ({result[1]['Category']})\n"
                        f"   Bore: {result[1]['Bore diameter (mm)']}mm, Outer: {result[1]['Outside diameter (mm)']}mm\n"
                        f"   Load: {result[1].get('Basic dynamic load rating (kN)', 'N/A')} kN"
                        for i, result in enumerate(results[:10], 1)
                    ])
                    
                    # Use Runnable chain with invoke (modern approach)
                    # Old way: chain.run(user_query=query, context=context_text)
                    # New way: chain.invoke({"user_query": query, "context": context_text})
                    
                    ai_response = chain.invoke({
                        "user_query": query, 
                        "context": context_text
                    })
                    
                    # Extract content from AIMessage
                    if hasattr(ai_response, 'content'):
                        response_text = ai_response.content
                    else:
                        response_text = str(ai_response)
                    
                    with st.expander("💡 AI Recommendations", expanded=True):
                        st.markdown(response_text)
                    st.divider()
                except Exception as e:
                    st.error(f"AI error: {str(e)}")
                    st.divider()
        
        # Display results in catalog format
        st.header("📦 Product Catalog Results")
        
        for rank, (_, row, dist) in enumerate(results, 1):
            display_bearing_card(row, rank)


if __name__ == "__main__":
    main()
