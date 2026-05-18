import streamlit as st
import os
import re
from supabase import create_client, Client
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.supabase import SupabaseVectorStore

# ==========================================
# 1. 페이지 설정 (Page Configuration)
# ==========================================
st.set_page_config(page_title="사업보고서 RAG 챗봇", page_icon="📊", layout="wide")
st.title("📊 사업보고서 RAG 챗봇 (Groq x Supabase)")

# ==========================================
# 2. 시크릿 키 관리 (Secrets Management)
# ==========================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    SUPABASE_DB_CONNECTION = st.secrets["SUPABASE_DB_CONNECTION"]
except KeyError as e:
    st.error(f"시크릿 키 설정 오류: st.secrets 환경변수를 확인하세요. 누락된 키: {e}")
    st.stop()

# ==========================================
# 3. 캐싱 및 초기화 (Caching & Initialization)
# ==========================================
@st.cache_resource
def init_supabase() -> Client:
    """Supabase API 클라이언트 초기화"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def init_llama_index():
    """LlamaIndex 전역 Settings (LLM, Embedding, Chunk) 설정"""
    # LLM 설정: Groq의 Llama 3 8B 모델 사용 (매우 빠르고 수치 정확성 높음)
    Settings.llm = Groq(
        model="llama3-8b-8192", 
        api_key=GROQ_API_KEY,
        temperature=0.1
    )
    
    # 임베딩 설정: HuggingFace의 오픈소스 임베딩 모델 사용 (768 차원 호환)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5"
    )
    
    # 청크 설정
    Settings.chunk_size = 500
    Settings.chunk_overlap = 50

# 서비스 초기화 실행
supabase_client = init_supabase()
init_llama_index()

def get_vector_store(company_name: str) -> SupabaseVectorStore:
    """회사명을 기반으로 Supabase Vector Store 연결 및 생성"""
    # 특수문자 제거 후 소문자 변환 규칙 적용
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', company_name).lower()
    if not clean_name.startswith('_'):
        clean_name = f"c_{clean_name}" # 테이블명 안전성 확보
        
    vector_store = SupabaseVectorStore(
        postgres_connection_string=SUPABASE_DB_CONNECTION,
        collection_name=clean_name,
        dimension=768
    )
    return vector_store

# ==========================================
# 4. 탭 구성 (3 Tabs)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📁 업로드", "💬 챗봇", "📜 채팅 기록"])

# ----- TAB 1: 업로드 -----
with tab1:
    st.header("📄 사업보고서 PDF 업로드")
    company_name = st.text_input("회사명 입력 (영어 또는 숫자 권장, 예: samsung, kakao):", placeholder="회사명을 입력하세요.")
    uploaded_file = st.file_input("사업보고서 PDF 파일을 선택하세요:", type=["pdf"])
    
    if st.button("Vector DB에 저장하기"):
        if not company_name or not uploaded_file:
            st.warning("회사명과 PDF 파일을 모두 입력해주세요.")
        else:
            with st.spinner("PDF 분석 및 Supabase 벡터 저장 중... 잠시만 기다려주세요."):
                try:
                    temp_dir = "./temp_data"
                    os.makedirs(temp_dir, exist_ok=True)
                    filepath = os.path.join(temp_dir, uploaded_file.name)
                    with open(filepath, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    reader = SimpleDirectoryReader(input_files=[filepath])
                    documents = reader.load_data()
                    
                    vector_store = get_vector_store(company_name)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    
                    VectorStoreIndex.from_documents(
                        documents, 
                        storage_context=storage_context,
                        show_progress=True
                    )
                    
                    os.remove(filepath)
                    st.success(f"🎉 '{company_name}' 사업보고서가 Supabase 벡터 DB에 성공적으로 저장되었습니다!")
                    
                except Exception as e:
                    st.error(f"업로드 중 에러 발생: {e}")

# ----- TAB 2: 챗봇 -----
with tab2:
    st.header("💬 AI 사업보고서 분석관")
    target_company = st.text_input("분석할 회사명 입력:", key="search_company", placeholder="업로드했던 회사명을 입력하세요.")
    user_question = st.text_input("💡 질문을 입력하세요:", placeholder="예: What is the total R&D expense for this year?")
    
    if st.button("질문하기"):
        if not target_company or not user_question:
            st.warning("회사명과 질문을 모두 입력해주세요.")
        else:
            with st.spinner("답변 생성 중..."):
                try:
                    vector_store = get_vector_store(target_company)
                    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
                    
                    query_engine = index.as_query_engine(similarity_top_k=3)
                    # 한국어로 답변해달라는 프롬프트 강제 추가
                    prompt_question = f"{user_question} (Please answer in Korean)"
                    response = query_engine.query(prompt_question)
                    
                    st.markdown("### 🤖 AI 답변:")
                    st.write(response.response)
                    
                    sources_list = []
                    st.markdown("### 📌 참고 출처 (Sources):")
                    for node in response.source_nodes:
                        page_num = node.node.metadata.get('page_label', 'Unknown')
                        text_preview = node.node.get_content()[:150] + "..."
                        
                        sources_list.append({"page": page_num, "text": text_preview})
                        st.caption(f"📄 **Page {page_num}** — {text_preview}")
                    
                    history_data = {
                        "question": user_question,
                        "answer": response.response,
                        "sources": sources_list,
                        "company_name": target_company
                    }
                    supabase_client.table("chat_history").insert(history_data).execute()
                    
                except Exception as e:
                    st.error(f"답변 조회 중 에러가 발생했습니다. 회사명을 정확히 입력했는지 확인하세요.\n에러 내용: {e}")

# ----- TAB 3: 채팅 기록 -----
with tab3:
    st.header("📜 유저 대화 이력")
    if st.button("기록 새로고침"):
        try:
            res = supabase_client.table("chat_history").select("*").order("created_at", desc=True).limit(20).execute()
            history = res.data
            
            if not history:
                st.info("저장된 대화 기록이 없습니다.")
            for chat in history:
                with st.expander(f"🏢 [{chat['company_name']}] {chat['question'][:30]}..."):
                    st.write(f"**질문:** {chat['question']}")
                    st.write(f"**답변:** {chat['answer']}")
                    st.markdown("**참고한 페이지 목록:**")
                    for src in chat.get('sources', []):
                        st.caption(f"- Page {src.get('page')}: {src.get('text')}")
        except Exception as e:
            st.error(f"기록을 불러오는 중 에러 발생: {e}")