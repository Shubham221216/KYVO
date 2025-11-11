# import streamlit as st
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import warnings


# # Suppress LangChain deprecation warnings
# warnings.filterwarnings('ignore', category=DeprecationWarning)
# warnings.filterwarnings('ignore', message='.*deprecated.*')


# # Updated LangChain v0.1+ imports with langchain_core.runnables
# from langchain_mistralai import ChatMistralAI
# from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_core.documents import Document
# from langchain_community.vectorstores import FAISS as LangChainFAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from typing import List, Tuple, Dict, Any


# # ============================================================================
# # CONFIGURATION
# # ============================================================================
# # CSV_FILES = ['SKR-data-0.csv', 'SKF-data-1.csv']
# CSV_FILES = ['cleaned_data.csv']
# MISTRAL_API_KEY = '9Lwb0Sbr6L5bqEZxl7PJxdfpTNIzsm2R'
# MISTRAL_MODEL = 'mistral-small-latest'


# # ============================================================================
# # DATA LOADING AND PREPROCESSING
# # ============================================================================
# @st.cache_data
# def load_bearing_data(files):
#     """Load and combine CSV files"""
#     dfs = [pd.read_csv(f) for f in files]
#     combined_df = pd.concat(dfs, ignore_index=True)
#     combined_df = combined_df.dropna(subset=['Designation'])
#     return combined_df


# def create_bearing_documents(df: pd.DataFrame) -> List[Document]:
#     """Convert bearing data to LangChain Documents"""
#     documents = []
#     for idx, row in df.iterrows():
#         # Create rich content for each bearing
#         content = f"""
# Designation: {row['Designation']}
# Category: {row['Category']}
# Type: {row.get('Short_Description', '')}


# Specifications:
# - Bore Diameter: {row['Bore diameter (mm)']} mm
# - Outside Diameter: {row['Outside diameter (mm)']} mm
# - Width: {row.get('Width (mm)', 'N/A')} mm
# - Dynamic Load Rating: {row.get('Basic dynamic load rating (kN)', 'N/A')} kN
# - Static Load Rating: {row.get('Basic static load rating (kN)', 'N/A')} kN
# - Speed Limit: {row.get('Limiting speed (r/min)', 'N/A')} r/min
# - Weight: {row.get('Product net weight (kg)', 'N/A')} kg
# - Material: {row.get('Material, bearing', 'N/A')}
# - Sealing: {row.get('Sealing', 'N/A')}
# - Number of Rows: {row.get('Number of rows', 'N/A')}


# Benefits:
# {row.get('Benefits', 'N/A')}
#         """.strip()
        
#         metadata = {
#             'designation': row['Designation'],
#             'category': row['Category'],
#             'bore_diameter': float(row['Bore diameter (mm)']),
#             'outside_diameter': float(row['Outside diameter (mm)']),
#             'dynamic_load': float(row.get('Basic dynamic load rating (kN)', 0)),
#             'static_load': float(row.get('Basic static load rating (kN)', 0)),
#             'speed_limit': float(row.get('Limiting speed (r/min)', 0)),
#             'photo_url': row.get('Photo_URL', ''),
#             'long_description': row.get('Long_Description', ''),
#             'index': idx
#         }
        
#         doc = Document(page_content=content, metadata=metadata)
#         documents.append(doc)
    
#     return documents


# # ============================================================================
# # EMBEDDING AND VECTOR STORE SETUP
# # ============================================================================
# @st.cache_resource
# def setup_vector_store(documents: List[Document]):
#     """Create LangChain vector store using FAISS"""
#     embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
#     vector_store = LangChainFAISS.from_documents(documents, embeddings)
#     return vector_store, embeddings


# # ============================================================================
# # LANGCHAIN RAG SETUP WITH RUNNABLES
# # ============================================================================
# @st.cache_resource
# def setup_langchain_rag(api_key: str):
#     """Initialize LangChain components with updated imports"""
    
#     # Initialize Mistral LLM using new import
#     llm = ChatMistralAI(
#         api_key=api_key,
#         model=MISTRAL_MODEL,
#         temperature=0.3,
#         max_tokens=800
#     )
    
#     return llm


# def create_bearing_expert_chain(llm: ChatMistralAI):
#     """Create LangChain chain using langchain_core.runnables (modern approach)"""
    
#     # System prompt template
#     system_template = """You are an expert SKF bearing specialist with deep knowledge of industrial bearings.


# Your role is to:
# 1. Understand what the user is looking for (their application, constraints, preferences)
# 2. Analyze the retrieved bearing products from the database
# 3. Translate the user's natural language query into technical bearing requirements
# 4. Recommend the most suitable bearings and explain WHY they match the user's needs
# 5. Provide professional, catalog-style recommendations


# When analyzing:
# - Look for speed requirements (high-speed, low-speed, etc.)
# - Consider load capacity (radial vs axial vs combined)
# - Note the application type implied by the user
# - Evaluate bore diameter, space constraints
# - Consider operating environment (temperature, moisture, etc.)


# Format your response professionally with clear sections for each recommended bearing.
# Always explain the technical reasoning behind your recommendations."""


#     system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    
#     # Human prompt template
#     human_template = """User Query: {user_query}


# Here are the top bearing products from our database that match the search:


# {context}


# Please:
# 1. Briefly identify what the user is looking for in technical terms
# 2. Analyze each bearing's suitability
# 3. Recommend the top 3-5 most suitable bearings with detailed explanations
# 4. For each recommendation, explain how it addresses the user's requirements
# 5. Highlight any tradeoffs or considerations


# Format with clear sections and bold the bearing designations."""


#     human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
#     # Combine into chat prompt
#     chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    
#     # Create runnable chain using | pipe operator (modern LangChain approach)
#     chain = chat_prompt | llm
    
#     return chain


# def create_bearing_expert_runnable(llm: ChatMistralAI):
#     """
#     Alternative: Create a more complex runnable with custom processing.
#     Shows advanced Runnable patterns.
#     """
    
#     system_template = """You are an expert SKF bearing specialist with deep knowledge of industrial bearings.

# Your role is to:
# 1. Understand what the user is looking for (their application, constraints, preferences)
# 2. Analyze the retrieved bearing products from the database
# 3. Translate the user's natural language query into technical bearing requirements
# 4. Recommend the most suitable bearings and explain WHY they match the user's needs
# 5. Provide professional, catalog-style recommendations

# When analyzing:
# - Look for speed requirements (high-speed, low-speed, etc.)
# - Consider load capacity (radial vs axial vs combined)
# - Note the application type implied by the user
# - Evaluate bore diameter, space constraints
# - Consider operating environment (temperature, moisture, etc.)

# Format your response professionally with clear sections for each recommended bearing.
# Always explain the technical reasoning behind your recommendations."""

#     system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    
#     human_template = """User Query: {user_query}

# Here are the top bearing products from our database that match the search:

# {context}

# Please:
# 1. Briefly identify what the user is looking for in technical terms
# 2. Analyze each bearing's suitability
# 3. Recommend the top 3-5 most suitable bearings with detailed explanations
# 4. For each recommendation, explain how it addresses the user's requirements
# 5. Highlight any tradeoffs or considerations

# Format with clear sections and bold the bearing designations."""

#     human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
#     chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    
#     # Create runnable chain with pipe operator
#     # This is equivalent to: chain = chat_prompt | llm
#     runnable_chain = chat_prompt | llm
    
#     return runnable_chain


# # ============================================================================
# # SEARCH AND FILTERING
# # ============================================================================
# def search_bearings(query, index, df, model, k=10, filters=None):
#     """Search for bearings matching the query with optional filters"""
    
#     # Create query embedding
#     query_emb = model.embed_query(query)
    
#     # Search in FAISS index
#     D, I = index.search(np.array([query_emb]), min(k * 3, len(df)))
    
#     results = []
#     for dist, idx in zip(D[0], I[0]):
#         if idx == -1 or idx >= len(df):
#             continue
        
#         row = df.iloc[idx]
        
#         # Apply filters if provided
#         if filters:
#             if filters.get('category') and row['Category'] != filters['category']:
#                 continue
#             if filters.get('min_bore') and row['Bore diameter (mm)'] < filters['min_bore']:
#                 continue
#             if filters.get('max_bore') and row['Bore diameter (mm)'] > filters['max_bore']:
#                 continue
#             if filters.get('min_load') and row.get('Basic dynamic load rating (kN)', 0) < filters['min_load']:
#                 continue
        
#         results.append((idx, row, float(dist)))
        
#         if len(results) >= k:
#             break
    
#     return results


# # ============================================================================
# # STREAMLIT UI
# # ============================================================================
# def display_bearing_card(row, rank):
#     """Display a single bearing in a nice card format"""
#     with st.container():
#         col1, col2 = st.columns([1, 3])
        
#         with col1:
#             # Display image if available
#             if pd.notna(row.get('Photo_URL')):
#                 try:
#                     st.image(row['Photo_URL'], width=150)
#                 except:
#                     st.write("🔧")
#             else:
#                 st.write("🔧")
        
#         with col2:
#             st.markdown(f"### {rank}. {row['Designation']}")
#             st.markdown(f"**{row['Category']}**")
#             st.caption(row.get('Short_Description', ''))
            
#             # Key specs in columns
#             spec_col1, spec_col2, spec_col3 = st.columns(3)
#             with spec_col1:
#                 st.metric("Bore Ø", f"{row['Bore diameter (mm)']} mm")
#             with spec_col2:
#                 st.metric("Outer Ø", f"{row['Outside diameter (mm)']} mm")
#             with spec_col3:
#                 load = row.get('Basic dynamic load rating (kN)', 'N/A')
#                 st.metric("Load Rating", f"{load} kN")
            
#             # Additional specs
#             with st.expander("📋 More Details"):
#                 st.write(f"**Width:** {row.get('Width (mm)', 'N/A')} mm")
#                 st.write(f"**Speed Limit:** {row.get('Limiting speed (r/min)', 'N/A')} r/min")
#                 st.write(f"**Weight:** {row.get('Product net weight (kg)', 'N/A')} kg")
#                 st.write(f"**Material:** {row.get('Material, bearing', 'N/A')}")
#                 st.write(f"**Sealing:** {row.get('Sealing', 'N/A')}")
#                 if pd.notna(row.get('Benefits')):
#                     st.write(f"**Benefits:** {row['Benefits']}")
        
#         st.divider()


# def main():
#     # Page config
#     st.set_page_config(
#         page_title="SKF Bearing Search",
#         page_icon="⚙️",
#         layout="wide"
#     )
    
#     # Title and description
#     st.title("⚙️ SKF Bearing Intelligent Search (LangChain + Runnables)")
#     st.markdown("""
#     **Search for bearings using natural language!** 
    
#     Try queries like:
#     - "I need a small bearing for high-speed applications"
#     - "Show me bearings with 10mm bore diameter"
#     - "Heavy duty bearings for industrial machinery"
#     - "Bearings that can handle both radial and axial loads"
#     """)
    
#     # Load data
#     with st.spinner("Loading bearing database..."):
#         df = load_bearing_data(CSV_FILES)
#         documents = create_bearing_documents(df)
#         vector_store, embeddings = setup_vector_store(documents)
#         llm = setup_langchain_rag(MISTRAL_API_KEY)
#         chain = create_bearing_expert_chain(llm)
        
#         # Build FAISS index from embeddings
#         texts = [doc.page_content for doc in documents]
#         embeddings_arr = embeddings.embed_documents(texts)
#         embeddings_arr = np.array(embeddings_arr).astype('float32')
#         index = faiss.IndexFlatL2(embeddings_arr.shape[1])
#         index.add(embeddings_arr)
    
#     st.success(f"✅ Loaded {len(df)} bearing products")
    
#     # Sidebar filters
#     with st.sidebar:
#         st.header("🔍 Filters")
        
#         # Category filter
#         categories = ['All'] + sorted(df['Category'].unique().tolist())
#         selected_category = st.selectbox("Bearing Type", categories)
        
#         # Bore diameter filter
#         st.subheader("Bore Diameter (mm)")
#         bore_range = st.slider(
#             "Range",
#             float(df['Bore diameter (mm)'].min()),
#             float(df['Bore diameter (mm)'].max()),
#             (float(df['Bore diameter (mm)'].min()), float(df['Bore diameter (mm)'].max()))
#         )
        
#         # Load rating filter
#         st.subheader("Min Load Rating (kN)")
#         min_load = st.number_input(
#             "Minimum",
#             min_value=0.0,
#             max_value=float(df['Basic dynamic load rating (kN)'].max()),
#             value=0.0,
#             step=1.0
#         )
        
#         # Number of results
#         num_results = st.slider("Number of Results", 5, 20, 10)
    
#     # Main search interface
#     query = st.text_input(
#         "🔎 What kind of bearing are you looking for?",
#         placeholder="e.g., 'small bearing for electric motor shaft'",
#         key="search_query"
#     )
    
#     col1, col2 = st.columns([1, 5])
#     with col1:
#         search_button = st.button("🚀 Search", type="primary", use_container_width=True)
#     with col2:
#         use_ai = st.checkbox("Use AI-powered recommendations", value=True)
    
#     if search_button and query:
#         # Prepare filters
#         filters = {
#             'category': selected_category if selected_category != 'All' else None,
#             'min_bore': bore_range[0],
#             'max_bore': bore_range[1],
#             'min_load': min_load
#         }
        
#         with st.spinner("Searching bearing catalog..."):
#             # Search for bearings
#             results = search_bearings(query, index, df, embeddings, k=num_results, filters=filters)
        
#         if not results:
#             st.warning("No bearings found matching your criteria. Try adjusting the filters.")
#             return
        
#         st.success(f"Found {len(results)} matching bearings")
        
#         # AI-powered recommendations
#         if use_ai and MISTRAL_API_KEY != 'your_mistral_api_key_here':
#             with st.spinner("🤖 AI is analyzing your requirements..."):
#                 try:
#                     # Prepare context
#                     context_text = "\n\n".join([
#                         f"{i}. {result[1]['Designation']} ({result[1]['Category']})\n"
#                         f"   Bore: {result[1]['Bore diameter (mm)']}mm, Outer: {result[1]['Outside diameter (mm)']}mm\n"
#                         f"   Load: {result[1].get('Basic dynamic load rating (kN)', 'N/A')} kN"
#                         for i, result in enumerate(results[:10], 1)
#                     ])
                    
#                     # Use Runnable chain with invoke (modern approach)
#                     # Old way: chain.run(user_query=query, context=context_text)
#                     # New way: chain.invoke({"user_query": query, "context": context_text})
                    
#                     ai_response = chain.invoke({
#                         "user_query": query, 
#                         "context": context_text
#                     })
                    
#                     # Extract content from AIMessage
#                     if hasattr(ai_response, 'content'):
#                         response_text = ai_response.content
#                     else:
#                         response_text = str(ai_response)
                    
#                     with st.expander("💡 AI Recommendations", expanded=True):
#                         st.markdown(response_text)
#                     st.divider()
#                 except Exception as e:
#                     st.error(f"AI error: {str(e)}")
#                     st.divider()
        
#         # Display results in catalog format
#         st.header("📦 Product Catalog Results")
        
#         for rank, (_, row, dist) in enumerate(results, 1):
#             display_bearing_card(row, rank)


# if __name__ == "__main__":
#     main()



# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # SKF BEARING RAG SYSTEM WITH LANGCHAIN v0.1+ USING RUNNABLES
# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# import streamlit as st
# import pandas as pd
# import numpy as np
# import warnings
# from typing import List, Tuple, Dict, Any

# warnings.filterwarnings('ignore', category=DeprecationWarning)
# warnings.filterwarnings('ignore', message='.*deprecated.*')

# # LangChain imports
# from langchain_mistralai import ChatMistralAI
# from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain_core.documents import Document
# from langchain_community.vectorstores import FAISS as LangChainFAISS
# from langchain_huggingface import HuggingFaceEmbeddings

# # ============================================================================
# # CONFIGURATION
# # ============================================================================
# CSV_FILES = ['cleaned_data.csv']
# MISTRAL_API_KEY = '9Lwb0Sbr6L5bqEZxl7PJxdfpTNIzsm2R'
# MISTRAL_MODEL = 'mistral-small-latest'

# # ============================================================================
# # DATA LOADING & PREPROCESSING
# # ============================================================================
# @st.cache_data
# def load_bearing_data(files):
#     """Load and combine CSV files with deduplication"""
#     try:
#         dfs = []
#         for f in files:
#             try:
#                 df = pd.read_csv(f)
#                 dfs.append(df)
#             except Exception as e:
#                 st.warning(f"Could not load {f}: {str(e)}")
#                 continue
        
#         if not dfs:
#             st.error("No CSV files could be loaded")
#             return pd.DataFrame()
        
#         combined_df = pd.concat(dfs, ignore_index=True)
#         combined_df = combined_df.dropna(subset=['Designation'])
#         combined_df = combined_df.drop_duplicates(subset=['Designation'], keep='first')
#         combined_df = combined_df.reset_index(drop=True)
#         return combined_df
#     except Exception as e:
#         st.error(f"Error loading data: {str(e)}")
#         return pd.DataFrame()


# def create_bearing_documents(df: pd.DataFrame) -> List[Document]:
#     """Convert bearing data to LangChain Documents"""
#     documents = []
    
#     for idx, row in df.iterrows():
#         try:
#             content = f"""
# Bearing Designation: {row.get('Designation', 'N/A')}
# Category: {row.get('Category', 'N/A')}
# Type: {row.get('Short_Description', 'N/A')}

# Technical Specifications:
# - Bore Diameter: {row.get('Bore diameter (mm)', 'N/A')} mm
# - Outside Diameter: {row.get('Outside diameter (mm)', 'N/A')} mm
# - Width: {row.get('Width (mm)', 'N/A')} mm
# - Dynamic Load Rating: {row.get('Basic dynamic load rating (kN)', 'N/A')} kN
# - Static Load Rating: {row.get('Basic static load rating (kN)', 'N/A')} kN
# - Speed Limit: {row.get('Limiting speed (r/min)', 'N/A')} r/min
# - Weight: {row.get('Product net weight (kg)', 'N/A')} kg
# - Material: {row.get('Material, bearing', 'N/A')}
# - Sealing: {row.get('Sealing', 'N/A')}

# Key Benefits:
# {row.get('Benefits', 'N/A')}
#             """.strip()
            
#             metadata = {
#                 'designation': str(row.get('Designation', 'N/A')),
#                 'category': str(row.get('Category', 'N/A')),
#                 'bore_diameter': float(row.get('Bore diameter (mm)', 0)) if pd.notna(row.get('Bore diameter (mm)')) else 0,
#                 'outside_diameter': float(row.get('Outside diameter (mm)', 0)) if pd.notna(row.get('Outside diameter (mm)')) else 0,
#                 'dynamic_load': float(row.get('Basic dynamic load rating (kN)', 0)) if pd.notna(row.get('Basic dynamic load rating (kN)')) else 0,
#                 'static_load': float(row.get('Basic static load rating (kN)', 0)) if pd.notna(row.get('Basic static load rating (kN)')) else 0,
#                 'speed_limit': float(row.get('Limiting speed (r/min)', 0)) if pd.notna(row.get('Limiting speed (r/min)')) else 0,
#                 'max_temp': float(row.get('Maximum operating temperature (°C)', 0)) if pd.notna(row.get('Maximum operating temperature (°C)')) else 0,
#                 'min_temp': float(row.get('Minimum operating temperature (°C)', 0)) if pd.notna(row.get('Minimum operating temperature (°C)')) else 0,
#                 'photo_url': str(row.get('Photo_URL', '')) if pd.notna(row.get('Photo_URL')) else '',
#                 'sealing': str(row.get('Sealing', 'N/A')) if pd.notna(row.get('Sealing')) else 'N/A',
#                 'index': idx
#             }
            
#             doc = Document(page_content=content, metadata=metadata)
#             documents.append(doc)
#         except Exception as e:
#             st.warning(f"Error processing row {idx}: {str(e)}")
#             continue
    
#     return documents


# # ============================================================================
# # SAFE UNIQUE VALUES EXTRACTION
# # ============================================================================
# def get_safe_unique_values(df: pd.DataFrame, column_name: str) -> List[str]:
#     """Safely extract unique values from a column, handling mixed types"""
#     try:
#         if column_name not in df.columns:
#             return []
        
#         # Convert all values to string, filtering out NaN
#         unique_vals = []
#         for val in df[column_name].unique():
#             if pd.notna(val):  # Skip NaN values
#                 str_val = str(val).strip()
#                 if str_val and str_val.lower() != 'nan':  # Skip empty strings and 'nan'
#                     unique_vals.append(str_val)
        
#         # Remove duplicates and sort
#         unique_vals = list(set(unique_vals))
#         unique_vals.sort()
#         return unique_vals
#     except Exception as e:
#         st.warning(f"Error extracting unique values from {column_name}: {str(e)}")
#         return []


# # ============================================================================
# # VECTOR STORE SETUP
# # ============================================================================
# @st.cache_resource
# def setup_vector_store(documents: List[Document]):
#     """Create and cache vector store"""
#     try:
#         embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
#         vector_store = LangChainFAISS.from_documents(documents, embeddings)
#         return vector_store, embeddings
#     except Exception as e:
#         st.error(f"Error setting up vector store: {str(e)}")
#         return None, None


# # ============================================================================
# # LLM SETUP
# # ============================================================================
# @st.cache_resource
# def setup_llm(api_key: str):
#     """Initialize Mistral LLM"""
#     try:
#         return ChatMistralAI(
#             api_key=api_key,
#             model=MISTRAL_MODEL,
#             temperature=0.3,
#             max_tokens=500
#         )
#     except Exception as e:
#         st.error(f"Error setting up LLM: {str(e)}")
#         return None


# # ============================================================================
# # QUERY ANALYSIS
# # ============================================================================
# def analyze_user_query(user_query: str, llm) -> Dict[str, Any]:
#     """Analyze user input to extract requirements"""
    
#     if not llm:
#         return {
#             "bearing_type": "All",
#             "speed_requirement": "not specified",
#             "load_requirement": "not specified",
#             "confidence": 0.0
#         }
    
#     analysis_prompt = ChatPromptTemplate.from_template("""
# You are a bearing specification analyzer. Analyze this query and extract key requirements.

# Query: {query}

# Return JSON with: bearing_type, speed_requirement, load_requirement, confidence (0-1).
# Return ONLY valid JSON, no other text.
# """)
    
#     chain = analysis_prompt | llm
    
#     try:
#         response = chain.invoke({"query": user_query})
#         content = response.content if hasattr(response, 'content') else str(response)
        
#         import json
#         import re
        
#         json_match = re.search(r'\{.*\}', content, re.DOTALL)
#         if json_match:
#             return json.loads(json_match.group())
#         else:
#             return {}
#     except Exception as e:
#         st.warning(f"Query analysis error: {str(e)}")
#         return {}


# # ============================================================================
# # FILTERING FUNCTION
# # ============================================================================
# def apply_filters(results: List[Tuple], filters: Dict, df: pd.DataFrame) -> List[Tuple]:
#     """Apply user-selected filters to results"""
    
#     if not filters or not any(filters.values()):
#         return results
    
#     filtered = []
    
#     for idx, row, score in results:
#         keep = True
        
#         # Category filter
#         if keep and filters.get('category') and filters['category'] != 'All':
#             try:
#                 if str(row.get('Category', 'N/A')).strip() != filters['category']:
#                     keep = False
#             except:
#                 pass
        
#         # Bore diameter filter
#         if keep and filters.get('bore_min') is not None:
#             try:
#                 bore = float(row.get('Bore diameter (mm)', 0))
#                 if bore < filters['bore_min']:
#                     keep = False
#             except:
#                 pass
        
#         if keep and filters.get('bore_max') is not None:
#             try:
#                 bore = float(row.get('Bore diameter (mm)', 0))
#                 if bore > filters['bore_max']:
#                     keep = False
#             except:
#                 pass
        
#         # Speed filter
#         if keep and filters.get('speed_min') is not None and filters['speed_min'] > 0:
#             try:
#                 speed = float(row.get('Limiting speed (r/min)', 0))
#                 if speed < filters['speed_min']:
#                     keep = False
#             except:
#                 pass
        
#         # Load filter
#         if keep and filters.get('load_min') is not None and filters['load_min'] > 0:
#             try:
#                 load = float(row.get('Basic dynamic load rating (kN)', 0))
#                 if load < filters['load_min']:
#                     keep = False
#             except:
#                 pass
        
#         # Sealing filter
#         if keep and filters.get('sealing') and filters['sealing'] != 'All':
#             try:
#                 bearing_sealing = str(row.get('Sealing', 'N/A')).lower().strip()
#                 filter_sealing = str(filters['sealing']).lower().strip()
#                 if bearing_sealing != filter_sealing:
#                     keep = False
#             except:
#                 pass
        
#         if keep:
#             filtered.append((idx, row, score))
    
#     return filtered


# # ============================================================================
# # RANKING FUNCTION
# # ============================================================================
# def rank_results(results: List[Tuple], df: pd.DataFrame) -> List[Tuple]:
#     """Simple ranking by vector similarity"""
    
#     scored = []
#     for idx, row, score in results:
#         # Normalize score (lower L2 distance = higher similarity)
#         normalized = 1.0 / (1.0 + score)
#         scored.append((idx, row, float(normalized * 100)))
    
#     # Sort by score
#     scored.sort(key=lambda x: x[2], reverse=True)
    
#     return scored


# # ============================================================================
# # DISPLAY BEARING RESULT
# # ============================================================================
# def display_bearing_result(rank: int, row: pd.Series):
#     """Display a bearing result with image and specs"""
    
#     with st.container():
#         # Header
#         col1, col2 = st.columns([1, 3])
#         with col1:
#             st.markdown(f"### #{rank}. {row.get('Designation', 'N/A')}")
#         with col2:
#             st.markdown(f"**Category:** `{row.get('Category', 'N/A')}`")
        
#         # Image and specs
#         img_col, spec_col = st.columns([2, 3])
        
#         with img_col:
#             st.markdown("#### Image")
#             photo_url = row.get('Photo_URL', '')
            
#             if pd.notna(photo_url) and photo_url and photo_url != '':
#                 try:
#                     st.image(photo_url, use_column_width=True, caption=f"SKF {row.get('Designation', 'N/A')}")
#                 except:
#                     st.info("📷 Image not available")
#             else:
#                 st.info("📷 Image not available")
        
#         with spec_col:
#             st.markdown("#### 📊 Specifications")
            
#             st.write(f"**Bore:** {row.get('Bore diameter (mm)', 'N/A')} mm")
#             st.write(f"**Outer:** {row.get('Outside diameter (mm)', 'N/A')} mm")
#             st.write(f"**Width:** {row.get('Width (mm)', 'N/A')} mm")
            
#             st.divider()
            
#             st.write(f"**Dynamic Load:** {row.get('Basic dynamic load rating (kN)', 'N/A')} kN")
#             st.write(f"**Speed Limit:** {row.get('Limiting speed (r/min)', 'N/A')} r/min")
#             st.write(f"**Material:** {row.get('Material, bearing', 'N/A')}")
#             st.write(f"**Sealing:** {row.get('Sealing', 'N/A')}")
        
#         # Description
#         with st.expander("📝 Full Details"):
#             st.write(f"**Short Description:** {row.get('Short_Description', 'N/A')}")
#             st.write(f"**Benefits:** {row.get('Benefits', 'N/A')}")
#             st.write(f"**Description:** {row.get('Long_Description', 'N/A')}")
        
#         st.divider()


# # ============================================================================
# # MAIN APPLICATION
# # ============================================================================
# def main():
#     st.set_page_config(
#         page_title="KYVO AI Product Finder",
#         page_icon="⚙️",
#         layout="wide"
#     )
    
#     st.title(" KYVO AI Product Finder")
#     # st.markdown("""
#     # **Smart bearing search** with natural language understanding and advanced filtering.
    
#     # 1.  **Describe** what you need
#     # 2.  **Refine** with filters (optional)
#     # 3.  **Get** best matches with images and specs
#     # """)
    
#     # Load data
#     with st.spinner("Loading bearing database..."):
#         df = load_bearing_data(CSV_FILES)
        
#         if df.empty:
#             st.error("Failed to load bearing data. Please check your CSV files.")
#             return
        
#         documents = create_bearing_documents(df)
        
#         if not documents:
#             st.error("No documents created from data.")
#             return
        
#         vector_store, embeddings = setup_vector_store(documents)
        
#         if vector_store is None or embeddings is None:
#             st.error("Failed to set up vector store.")
#             return
        
#         llm = setup_llm(MISTRAL_API_KEY)
    
#     st.success(f"✅ Ready! Database: {len(df)} bearings")

    
#     # Sidebar filters
#     with st.sidebar:
#         st.header("🔎 Filters (Optional)")
        
#         # Category filter - SAFE extraction
#         categories = ['All'] + get_safe_unique_values(df, 'Category')
#         selected_category = st.selectbox("Bearing Type", categories, key="cat_filter")
        
#         # Bore diameter filter
#         st.subheader("Bore Diameter (mm)")
#         bore_min = st.number_input("Min", value=0.0, key="bore_min")
#         bore_max = st.number_input("Max", value=1000.0, key="bore_max")
        
#         # Speed filter
#         st.subheader("Speed (r/min)")
#         speed_min = st.number_input("Min Speed", value=0, step=1000, key="speed")
        
#         # Load filter
#         st.subheader("Load (kN)")
#         load_min = st.number_input("Min Load", value=0.0, key="load")
        
#         # Sealing filter - SAFE extraction
#         sealings = ['All'] + get_safe_unique_values(df, 'Sealing')
#         selected_sealing = st.selectbox("Sealing Type", sealings, key="seal_filter")
        
#         # Results
#         num_results = st.slider("Results to Show", 3, 20, 10, key="num_results")
    
#     # Main search
#     st.header("Search for Bearings")
#     query = st.text_area(
#         "What bearing do you need?",
#         placeholder="e.g., 'bearing for electric motor with 15mm bore'",
#         height=80
#     )
    
#     if st.button("Search", type="primary", use_container_width=True):
#         if not query.strip():
#             st.warning("Please enter a search query")
#             return
        
#         with st.spinner("Searching..."):
#             try:
#                 # Step 1: Analyze query
#                 st.write("🔍 Analyzing requirements...")
#                 analysis = analyze_user_query(query, llm)
                
#                 # Step 2: Vector search
#                 st.write("📊 Searching database...")
#                 docs_with_scores = vector_store.similarity_search_with_score(query, k=30)
                
#                 results = []
#                 seen = set()
#                 for doc, score in docs_with_scores:
#                     desig = doc.metadata.get('designation', 'unknown')
#                     if desig not in seen:
#                         idx = doc.metadata.get('index', -1)
#                         if 0 <= idx < len(df):
#                             row = df.iloc[idx]
#                             results.append((idx, row, score))
#                             seen.add(desig)
#                             if len(results) >= 30:
#                                 break
                
#                 st.write(f"Found {len(results)} candidates")
                
#                 # Step 3: Apply filters
#                 st.write("🔎 Applying filters...")
#                 filters = {
#                     'category': selected_category,
#                     'bore_min': bore_min if bore_min > 0 else None,
#                     'bore_max': bore_max if bore_max < 1000 else None,
#                     'speed_min': speed_min if speed_min > 0 else None,
#                     'load_min': load_min if load_min > 0 else None,
#                     'sealing': selected_sealing,
#                 }
                
#                 results = apply_filters(results, filters, df)
#                 st.write(f"After filters: {len(results)} results")
                
#                 # Step 4: Rank
#                 st.write("🏆 Ranking results...")
#                 results = rank_results(results, df)
                
#                 st.divider()
                
#                 if results:
#                     st.success(f"✅ Found {len(results)} matching bearings")
                    
#                     # Display results
#                     for rank, (idx, row, score) in enumerate(results[:num_results], 1):
#                         display_bearing_result(rank, row)
#                 else:
#                     st.warning("❌ No bearings match your criteria. Try adjusting filters.")
            
#             except Exception as e:
#                 st.error(f"Error during search: {str(e)}")
#                 import traceback
#                 st.error(traceback.format_exc())


# if __name__ == "__main__":
#     main()


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SKF BEARING RAG SYSTEM WITH LANGCHAIN v0.1+ USING RUNNABLES - PROFESSIONAL UI
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import warnings
from typing import List, Tuple, Dict, Any

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*deprecated.*')

# LangChain imports
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================================
# CONFIGURATION
# ============================================================================
CSV_FILES = ['ai_studio_code.csv']
MISTRAL_API_KEY = '9Lwb0Sbr6L5bqEZxl7PJxdfpTNIzsm2R'
MISTRAL_MODEL = 'mistral-small-latest'

# Sample prompts for users
SAMPLE_PROMPTS = [
    "Ball bearings for high-speed electric motors",
    "Sealed bearings with 20mm bore diameter",
    "Heavy-duty bearings for industrial applications",
    "Bearing for 15mm shaft with high temperature resistance"
]

# ============================================================================
# CUSTOM CSS FOR PROFESSIONAL LOOK
# ============================================================================
def load_custom_css():
    st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .header-container {
        background: white;
        padding: 1rem 2rem;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 2rem;
    }
    
    .logo-text {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
    }
    
    /* Hero section */
    .hero-section {
        background: white;
        padding: 4rem 2rem;
        text-align: center;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 1rem;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    
    /* Sample prompts */
    .sample-prompts {
        margin-top: 1.5rem;
        text-align: center;
    }
    
    .sample-prompt-label {
        color: #6b7280;
        font-size: 0.875rem;
        margin-bottom: 0.75rem;
    }
    
    /* Search results header */
    .results-header {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e5e7eb;
    }
    
    /* Product card styling */
    .product-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: box-shadow 0.3s ease;
    }
    
    .product-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .product-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .product-category {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    /* Grid view styling */
    .grid-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        overflow: hidden;
        transition: box-shadow 0.3s ease;
        height: 100%;
    }
    
    .grid-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .grid-card-image {
        width: 100%;
        aspect-ratio: 1;
        background: #f3f4f6;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .grid-card-content {
        padding: 1rem;
    }
    
    /* Info box styling */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    /* Spec table */
    .spec-table {
        width: 100%;
        margin: 1rem 0;
    }
    
    .spec-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #f3f4f6;
    }
    
    .spec-label {
        color: #6b7280;
        font-weight: 500;
    }
    
    .spec-value {
        color: #1f2937;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING & PREPROCESSING (Same as before)
# ============================================================================
@st.cache_data
def load_bearing_data(files):
    """Load and combine CSV files with deduplication"""
    try:
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                st.warning(f"Could not load {f}: {str(e)}")
                continue
        
        if not dfs:
            st.error("No CSV files could be loaded")
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.dropna(subset=['Designation'])
        combined_df = combined_df.drop_duplicates(subset=['Designation'], keep='first')
        combined_df = combined_df.reset_index(drop=True)
        return combined_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()


def create_bearing_documents(df: pd.DataFrame) -> List[Document]:
    """Convert bearing data to LangChain Documents"""
    documents = []
    
    for idx, row in df.iterrows():
        try:
            content = f"""
Bearing Designation: {row.get('Designation', 'N/A')}
Category: {row.get('Category', 'N/A')}
Type: {row.get('Short_Description', 'N/A')}

Technical Specifications:
- Bore Diameter: {row.get('Bore diameter (mm)', 'N/A')} mm
- Outside Diameter: {row.get('Outside diameter (mm)', 'N/A')} mm
- Width: {row.get('Width (mm)', 'N/A')} mm
- Dynamic Load Rating: {row.get('Basic dynamic load rating (kN)', 'N/A')} kN
- Static Load Rating: {row.get('Basic static load rating (kN)', 'N/A')} kN
- Speed Limit: {row.get('Limiting speed (r/min)', 'N/A')} r/min
- Weight: {row.get('Product net weight (kg)', 'N/A')} kg
- Material: {row.get('Material, bearing', 'N/A')}
- Sealing: {row.get('Sealing', 'N/A')}

Key Benefits:
{row.get('Benefits', 'N/A')}
            """.strip()
            
            metadata = {
                'designation': str(row.get('Designation', 'N/A')),
                'category': str(row.get('Category', 'N/A')),
                'bore_diameter': float(row.get('Bore diameter (mm)', 0)) if pd.notna(row.get('Bore diameter (mm)')) else 0,
                'outside_diameter': float(row.get('Outside diameter (mm)', 0)) if pd.notna(row.get('Outside diameter (mm)')) else 0,
                'dynamic_load': float(row.get('Basic dynamic load rating (kN)', 0)) if pd.notna(row.get('Basic dynamic load rating (kN)')) else 0,
                'static_load': float(row.get('Basic static load rating (kN)', 0)) if pd.notna(row.get('Basic static load rating (kN)')) else 0,
                'speed_limit': float(row.get('Limiting speed (r/min)', 0)) if pd.notna(row.get('Limiting speed (r/min)')) else 0,
                'max_temp': float(row.get('Maximum operating temperature (°C)', 0)) if pd.notna(row.get('Maximum operating temperature (°C)')) else 0,
                'min_temp': float(row.get('Minimum operating temperature (°C)', 0)) if pd.notna(row.get('Minimum operating temperature (°C)')) else 0,
                'photo_url': str(row.get('Photo_URL', '')) if pd.notna(row.get('Photo_URL')) else '',
                'sealing': str(row.get('Sealing', 'N/A')) if pd.notna(row.get('Sealing')) else 'N/A',
                'index': idx
            }
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        except Exception as e:
            st.warning(f"Error processing row {idx}: {str(e)}")
            continue
    
    return documents


def get_safe_unique_values(df: pd.DataFrame, column_name: str) -> List[str]:
    """Safely extract unique values from a column, handling mixed types"""
    try:
        if column_name not in df.columns:
            return []
        
        unique_vals = []
        for val in df[column_name].unique():
            if pd.notna(val):
                str_val = str(val).strip()
                if str_val and str_val.lower() != 'nan':
                    unique_vals.append(str_val)
        
        unique_vals = list(set(unique_vals))
        unique_vals.sort()
        return unique_vals
    except Exception as e:
        st.warning(f"Error extracting unique values from {column_name}: {str(e)}")
        return []


@st.cache_resource
def setup_vector_store(documents: List[Document]):
    """Create and cache vector store"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        vector_store = LangChainFAISS.from_documents(documents, embeddings)
        return vector_store, embeddings
    except Exception as e:
        st.error(f"Error setting up vector store: {str(e)}")
        return None, None


@st.cache_resource
def setup_llm(api_key: str):
    """Initialize Mistral LLM"""
    try:
        return ChatMistralAI(
            api_key=api_key,
            model=MISTRAL_MODEL,
            temperature=0.3,
            max_tokens=500
        )
    except Exception as e:
        st.error(f"Error setting up LLM: {str(e)}")
        return None


def analyze_user_query(user_query: str, llm) -> Dict[str, Any]:
    """Analyze user input to extract requirements"""
    
    if not llm:
        return {
            "bearing_type": "All",
            "speed_requirement": "not specified",
            "load_requirement": "not specified",
            "confidence": 0.0
        }
    
    analysis_prompt = ChatPromptTemplate.from_template("""
You are a bearing specification analyzer. Analyze this query and extract key requirements.

Query: {query}

Return JSON with: bearing_type, speed_requirement, load_requirement, confidence (0-1).
Return ONLY valid JSON, no other text.
""")
    
    chain = analysis_prompt | llm
    
    try:
        response = chain.invoke({"query": user_query})
        content = response.content if hasattr(response, 'content') else str(response)
        
        import json
        import re
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {}
    except Exception as e:
        st.warning(f"Query analysis error: {str(e)}")
        return {}


def apply_filters(results: List[Tuple], filters: Dict, df: pd.DataFrame) -> List[Tuple]:
    """Apply user-selected filters to results"""
    
    if not filters or not any(filters.values()):
        return results
    
    filtered = []
    
    for idx, row, score in results:
        keep = True
        
        if keep and filters.get('category') and filters['category'] != 'All':
            try:
                if str(row.get('Category', 'N/A')).strip() != filters['category']:
                    keep = False
            except:
                pass
        
        if keep and filters.get('bore_min') is not None:
            try:
                bore = float(row.get('Bore diameter (mm)', 0))
                if bore < filters['bore_min']:
                    keep = False
            except:
                pass
        
        if keep and filters.get('bore_max') is not None:
            try:
                bore = float(row.get('Bore diameter (mm)', 0))
                if bore > filters['bore_max']:
                    keep = False
            except:
                pass
        
        if keep and filters.get('speed_min') is not None and filters['speed_min'] > 0:
            try:
                speed = float(row.get('Limiting speed (r/min)', 0))
                if speed < filters['speed_min']:
                    keep = False
            except:
                pass
        
        if keep and filters.get('load_min') is not None and filters['load_min'] > 0:
            try:
                load = float(row.get('Basic dynamic load rating (kN)', 0))
                if load < filters['load_min']:
                    keep = False
            except:
                pass
        
        if keep and filters.get('sealing') and filters['sealing'] != 'All':
            try:
                bearing_sealing = str(row.get('Sealing', 'N/A')).lower().strip()
                filter_sealing = str(filters['sealing']).lower().strip()
                if bearing_sealing != filter_sealing:
                    keep = False
            except:
                pass
        
        if keep:
            filtered.append((idx, row, score))
    
    return filtered


def rank_results(results: List[Tuple], df: pd.DataFrame) -> List[Tuple]:
    """Simple ranking by vector similarity"""
    
    scored = []
    for idx, row, score in results:
        normalized = 1.0 / (1.0 + score)
        scored.append((idx, row, float(normalized * 100)))
    
    scored.sort(key=lambda x: x[2], reverse=True)
    
    return scored


def display_bearing_grid(rank: int, row: pd.Series):
    """Display bearing in grid card format"""
    
    photo_url = row.get('Photo_URL', '')
    
    # Image
    if pd.notna(photo_url) and photo_url and photo_url != '':
        try:
            st.image(photo_url, use_column_width=True)
        except:
            st.markdown('<div style="background: #f3f4f6; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 0.5rem;">📷 No Image</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background: #f3f4f6; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 0.5rem;">📷 No Image</div>', unsafe_allow_html=True)
    
    # Content
    st.markdown(f"### {row.get('Designation', 'N/A')}")
    st.markdown(f'<span class="product-category">{row.get("Category", "N/A")}</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Bore:** {row.get('Bore diameter (mm)', 'N/A')} mm")
    with col2:
        st.markdown(f"**Load:** {row.get('Basic dynamic load rating (kN)', 'N/A')} kN")


def display_bearing_list(rank: int, row: pd.Series):
    """Display bearing in list format"""
    
    col1, col2, col3 = st.columns([1, 3, 2])
    
    with col1:
        photo_url = row.get('Photo_URL', '')
        if pd.notna(photo_url) and photo_url and photo_url != '':
            try:
                st.image(photo_url, use_column_width=True)
            except:
                st.markdown('<div style="background: #f3f4f6; height: 100px; display: flex; align-items: center; justify-content: center; border-radius: 0.5rem;">📷</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background: #f3f4f6; height: 100px; display: flex; align-items: center; justify-content: center; border-radius: 0.5rem;">📷</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"### #{rank}. {row.get('Designation', 'N/A')}")
        st.markdown(f'<span class="product-category">{row.get("Category", "N/A")}</span>', unsafe_allow_html=True)
        st.caption(row.get('Short_Description', 'N/A'))
    
    with col3:
        st.markdown(f"**Bore:** {row.get('Bore diameter (mm)', 'N/A')} mm")
        st.markdown(f"**Outer:** {row.get('Outside diameter (mm)', 'N/A')} mm")
        st.markdown(f"**Load:** {row.get('Basic dynamic load rating (kN)', 'N/A')} kN")
        st.markdown(f"**Speed:** {row.get('Limiting speed (r/min)', 'N/A')} r/min")
    
    with st.expander("📝 View Full Details"):
        st.write(f"**Material:** {row.get('Material, bearing', 'N/A')}")
        st.write(f"**Sealing:** {row.get('Sealing', 'N/A')}")
        st.write(f"**Width:** {row.get('Width (mm)', 'N/A')} mm")
        st.write(f"**Weight:** {row.get('Product net weight (kg)', 'N/A')} kg")
        st.write(f"**Benefits:** {row.get('Benefits', 'N/A')}")
        st.write(f"**Description:** {row.get('Long_Description', 'N/A')}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.set_page_config(
        page_title="KYVO AI Product Finder",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'grid'
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ''
    if 'selected_prompt' not in st.session_state:
        st.session_state.selected_prompt = ''
    
    # Header
    st.markdown("""
    <div class="header-container">
        <span class="logo-text">⚙️ KYVO AI</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading bearing database..."):
        df = load_bearing_data(CSV_FILES)
        
        if df.empty:
            st.error("❌ Failed to load bearing data. Please check your CSV files.")
            return
        
        documents = create_bearing_documents(df)
        
        if not documents:
            st.error("❌ No documents created from data.")
            return
        
        vector_store, embeddings = setup_vector_store(documents)
        
        if vector_store is None or embeddings is None:
            st.error("❌ Failed to set up vector store.")
            return
        
        llm = setup_llm(MISTRAL_API_KEY)
    
    # Hero Section (only show if no search performed)
    if not st.session_state.search_performed:
        st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">Electronic Component<br>Search Engine</h1>
            <p class="hero-subtitle">Instantly search across millions of components using natural language</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Search bar
        query = st.text_input(
            "Search",
            value=st.session_state.selected_prompt,
            placeholder="Show me Accelerometers with a low power mode...",
            label_visibility="collapsed",
            key="search_input"
        )
        
        # Sample prompts
        st.markdown('<div class="sample-prompts"><p class="sample-prompt-label">💡 Try these examples:</p></div>', unsafe_allow_html=True)
        
        cols = st.columns(2)
        for idx, prompt in enumerate(SAMPLE_PROMPTS):
            with cols[idx % 2]:
                if st.button(prompt, key=f"prompt_{idx}", use_container_width=True):
                    st.session_state.selected_prompt = prompt
                    st.session_state.current_query = prompt
                    st.session_state.search_performed = True
                    st.rerun()
        
        # How it works
        with st.expander("ℹ️ How It Works"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("### 1️⃣")
                st.markdown("**Describe Your Need**")
                st.caption("Use natural language to describe what you're looking for")
            
            with col2:
                st.markdown("### 2️⃣")
                st.markdown("**AI Analyzes**")
                st.caption("Our AI processes your query and understands requirements")
            
            with col3:
                st.markdown("### 3️⃣")
                st.markdown("**Refine Results**")
                st.caption("Apply filters to narrow down your search")
            
            with col4:
                st.markdown("### 4️⃣")
                st.markdown("**Get Matches**")
                st.caption("View detailed specs and images")
        
        # Search button
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query and query.strip():
                st.session_state.search_performed = True
                st.session_state.current_query = query
                st.session_state.selected_prompt = ''
                st.rerun()
    
    # Results view
    else:
        # Search bar (compact version)
        col1, col2 = st.columns([5, 1])
        with col1:
            new_query = st.text_input(
                "Search",
                value=st.session_state.get('current_query', ''),
                placeholder="Search for bearings...",
                label_visibility="collapsed",
                key="search_results_input"
            )
        with col2:
            if st.button("🔍 Search", type="primary", use_container_width=True):
                if new_query and new_query.strip():
                    st.session_state.current_query = new_query
                    st.rerun()
        
        query = st.session_state.get('current_query', '')
        
        if query and query.strip():
            with st.spinner("🔍 Searching..."):
                try:
                    # Vector search
                    docs_with_scores = vector_store.similarity_search_with_score(query, k=30)
                    
                    results = []
                    seen = set()
                    for doc, score in docs_with_scores:
                        desig = doc.metadata.get('designation', 'unknown')
                        if desig not in seen:
                            idx = doc.metadata.get('index', -1)
                            if 0 <= idx < len(df):
                                row = df.iloc[idx]
                                results.append((idx, row, score))
                                seen.add(desig)
                                if len(results) >= 30:
                                    break
                    
                    st.session_state.search_results = results
                
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
                    return
        
        results = st.session_state.search_results
        
        # Filters and View Controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"### Search Results ({len(results)} products found)")
        
        with col2:
            view_mode = st.selectbox(
                "View",
                ["Grid View", "List View"],
                index=0 if st.session_state.view_mode == 'grid' else 1,
                label_visibility="collapsed"
            )
            st.session_state.view_mode = 'grid' if view_mode == "Grid View" else 'list'
        
        with col3:
            show_filters = st.checkbox("🔎 Filters", value=False)
        
        st.markdown("---")
        
        # Main content area
        if show_filters:
            filter_col, results_col = st.columns([1, 3])
            
            with filter_col:
                st.markdown("### 🔎 Filters")
                
                # Category filter
                categories = ['All'] + get_safe_unique_values(df, 'Category')
                selected_category = st.selectbox("Bearing Type", categories, key="cat_filter")
                
                # Bore diameter
                st.markdown("**Bore Diameter (mm)**")
                bore_col1, bore_col2 = st.columns(2)
                with bore_col1:
                    bore_min = st.number_input("Min", value=0.0, key="bore_min", label_visibility="collapsed")
                with bore_col2:
                    bore_max = st.number_input("Max", value=1000.0, key="bore_max", label_visibility="collapsed")
                
                # Speed
                speed_min = st.number_input("Min Speed (r/min)", value=0, step=1000, key="speed")
                
                # Load
                load_min = st.number_input("Min Load (kN)", value=0.0, key="load")
                
                # Sealing
                sealings = ['All'] + get_safe_unique_values(df, 'Sealing')
                selected_sealing = st.selectbox("Sealing Type", sealings, key="seal_filter")
                
                # Apply filters button
                if st.button("Apply Filters", type="primary", use_container_width=True):
                    filters = {
                        'category': selected_category,
                        'bore_min': bore_min if bore_min > 0 else None,
                        'bore_max': bore_max if bore_max < 1000 else None,
                        'speed_min': speed_min if speed_min > 0 else None,
                        'load_min': load_min if load_min > 0 else None,
                        'sealing': selected_sealing,
                    }
                    
                    results = apply_filters(results, filters, df)
                    results = rank_results(results, df)
                    st.session_state.search_results = results
                    st.rerun()
            
            results_container = results_col
        else:
            results_container = st.container()
        
        # Display results
        with results_container:
            if results:
                if st.session_state.view_mode == 'grid':
                    # Grid view
                    cols_per_row = 3
                    for i in range(0, len(results), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j in range(cols_per_row):
                            if i + j < len(results):
                                idx, row, score = results[i + j]
                                with cols[j]:
                                    with st.container():
                                        st.markdown('<div class="grid-card">', unsafe_allow_html=True)
                                        display_bearing_grid(i + j + 1, row)
                                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # List view
                    for rank, (idx, row, score) in enumerate(results, 1):
                        with st.container():
                            st.markdown('<div class="product-card">', unsafe_allow_html=True)
                            display_bearing_list(rank, row)
                            st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.warning("❌ No bearings match your criteria. Try adjusting your search or filters.")
        
        # Back to search button
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("← New Search", use_container_width=False):
            st.session_state.search_performed = False
            st.session_state.search_results = []
            st.session_state.current_query = ""
            st.session_state.selected_prompt = ""
            st.rerun()


if __name__ == "__main__":
    main()