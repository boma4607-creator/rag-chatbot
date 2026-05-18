import streamlit as st
import os
import re
import tempfile
import json
from supabase import create_client, Client
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.supabase import SupabaseVectorStore

# --- 1. 페이지 설정 및 시크릿 로드 ---
st.set_page_config(page_title="사업보고서 RAG 챗봇", page_icon="📊", layout="wide")

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    SUPABASE_DB_CONNECTION = st.secrets["SUPABASE_DB_CONNECTION"]
except KeyError as e:
    st.error(f"시크릿 설정이 누락되었습니다: {e}")
    st.stop()

# --- 2. 캐싱 및 리소스 최적화 ---

@st.cache_resource
def init_llama_index():
    """LlamaIndex 전역 설정 (LLM, 임베딩, 청크 사이즈)"""
    # LLM 설정: Gemini 1.5 Flash (속도와 비용 효율성, 2.5는 현재 미출시로 최신 안정버전 사용)
    llm = Gemini(model="models/gemini-1.5-flash", api_key=GEMINI_API_KEY, temperature=0.1)
    
    # 임베딩 설정: Gemini Embedding
    embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=GEMINI_API_KEY)
    
    # 전역 Settings에 등록 (LlamaIndex v0.10 방식)
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 500
    Settings.chunk_overlap = 50
    return llm, embed_model

@st.cache_resource
def init_supabase() -> Client:
    """Supabase API 클라이언트 초기화"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def clean_collection_name(name):
    """회사명을 DB 테이블/컬렉션 이름 규칙에 맞게 정제"""
    # 영어/숫자/한글/언더바를 제외한 문자는 '_'로 변경 후 소문자화
    clean = re.sub(r'[^a-zA-Z0-9가-힣_]', '_', name)
    return clean.lower()

def get_vector_store(company_name):
    """해당 회사의 Supabase Vector Store 생성"""
    table_name = f"vec_{clean_collection_name(company_name)}"
    vector_store = SupabaseVectorStore(
        postgres_connection_string=SUPABASE_DB_CONNECTION,
        collection_name=table_name,
        dimension=768 # Gemini Embedding 출력 차원
    )
    return vector_store

# 초기화 실행
llm, embed_model = init_llama_index()
supabase_client = init_supabase()

# --- 3. UI 화면 구성 (탭) ---
st.title("📊 사업보고서 분석 AI 비서")
st.markdown("PDF 사업보고서를 업로드하고 궁금한 점을 질문하세요.")

tab1, tab2, tab3 = st.tabs(["📁 보고서 업로드", "💬 AI 챗봇", "📜 채팅 기록"])

# [탭 1: 보고서 업로드]
with tab1:
    st.header("보고서 데이터 학습")
    up_company = st.text_input("회사명 입력 (예: 삼성전자)", key="up_name")
    uploaded_file = st.file_uploader("사업보고서 PDF 파일을 선택하세요", type="pdf")
    
    if st.button("학습 시작"):
        if not up_company or not uploaded_file:
            st.warning("회사명과 파일을 모두 입력해주세요.")
        else:
            with st.spinner(f"'{up_company}' 보고서를 분석 중입니다. 잠시만 기다려주세요..."):
                # 임시 폴더에 파일 저장
                with tempfile.TemporaryDirectory() as tmp_dir:
                    file_path = os.path.join(tmp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # 1. 문서 로드
                    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
                    
                    # 2. Vector Store 연결 및 인덱싱
                    vector_store = get_vector_store(up_company)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    index = VectorStoreIndex.from_documents(
                        documents, storage_context=storage_context, show_progress=True
                    )
                    st.success(f"'{up_company}' 보고서 학습 완료! 이제 챗봇 탭에서 질문하세요.")

# [탭 2: 챗봇]
with tab2:
    st.header("사업보고서 Q&A")
    q_company = st.text_input("질문할 회사명 입력", key="q_name")
    question = st.text_input("질문을 입력하세요 (예: 올해 매출액과 영업이익은 얼마인가요?)")
    
    if st.button("질문하기"):
        if not q_company or not question:
            st.warning("회사명과 질문을 입력해주세요.")
        else:
            with st.spinner("답변을 생성 중입니다..."):
                try:
                    # 1. 인덱스 로드
                    vector_store = get_vector_store(q_company)
                    index = VectorStoreIndex.from_vector_store(vector_store)
                    
                    # 2. 쿼리 엔진 생성 (Top 3 유사 문장 참고)
                    query_engine = index.as_query_engine(similarity_top_k=3)
                    response = query_engine.query(question)
                    
                    # 3. 결과 출력
                    st.subheader("AI 답변")
                    st.write(response.response)
                    
                    # 4. 출처 정보 정리
                    st.subheader("참고한 정보")
                    sources = []
                    for node in response.source_nodes:
                        page_num = node.metadata.get('page_label', '알 수 없음')
                        content = node.get_content()[:200] + "..."
                        sources.append({"page": page_num, "text": content})
                        with st.expander(f"페이지 {page_num} 내용 보기"):
                            st.write(content)
                    
                    # 5. Supabase에 대화 기록 저장
                    history_data = {
                        "question": question,
                        "answer": response.response,
                        "sources": sources, # list를 JSONB로 저장
                        "company_name": q_company
                    }
                    supabase_client.table("chat_history").insert(history_data).execute()
                    
                except Exception as e:
                    st.error(f"데이터를 찾을 수 없습니다. 먼저 업로드 탭에서 학습을 진행해주세요. (에러: {e})")

# [탭 3: 채팅 기록]
with tab3:
    st.header("최근 질문 기록")
    if st.button("기록 불러오기"):
        # 최신 20개 기록 가져오기
        response = supabase_client.table("chat_history") \
            .select("*") \
            .order("created_at", desc=True) \
            .limit(20) \
            .execute()
        
        records = response.data
        if not records:
            st.info("저장된 기록이 없습니다.")
        else:
            for rec in records:
                with st.expander(f"[{rec['company_name']}] {rec['question']} ({rec['created_at'][:10]})"):
                    st.write(f"**Q:** {rec['question']}")
                    st.write(f"**A:** {rec['answer']}")
                    st.markdown("---")
                    st.write("**참고 페이지:**")
                    for src in rec['sources']:
                        st.write(f"- Page {src['page']}")
