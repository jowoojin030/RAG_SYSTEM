# app.py
import os
import json
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma

# ======================
# 기본 설정 / API KEY
# ======================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="카드 추천 챗봇", page_icon="💳", layout="wide")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY가 .env에 설정되지 않았습니다.")
    st.stop()

CREDIT_JSON_PATH = "cardgorilla_crd_806_fee.json"
CHECK_JSON_PATH = "cardgorilla__check_crd_374.json"

def assert_file_exists(path: str):
    if not os.path.exists(path):
        st.error(f"❌ 파일을 찾을 수 없습니다: {path}\n\n현재 작업 폴더: {os.getcwd()}")
        st.stop()

assert_file_exists(CREDIT_JSON_PATH)
assert_file_exists(CHECK_JSON_PATH)

# ======================
# ✨ CSS (디자인 핵심)
# ======================
st.markdown(
    """
<style>
/* 전체 폭/여백 */
.block-container { padding-top: 2.0rem; padding-bottom: 2.5rem; max-width: 1200px; }

/* 사이드바 */
section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.08); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.8rem; }

/* 타이포 */
h1, h2, h3 { letter-spacing: -0.02em; }
.small-muted { color: rgba(255,255,255,0.65); font-size: 0.92rem; line-height: 1.5; }

/* 히어로 카드 */
.hero {
  border: 1px solid rgba(255,255,255,0.10);
  background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border-radius: 18px;
  padding: 18px 18px 16px 18px;
  margin-bottom: 18px;
}
.hero-title { font-size: 2.0rem; font-weight: 800; margin: 0 0 6px 0; }
.hero-sub { margin: 0; }
.chips { margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  font-size: 0.86rem;
  color: rgba(255,255,255,0.78);
}

/* 카드(추천 3개) */
.card {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 18px;
  padding: 14px 14px 10px 14px;
}
.card h3 { margin: 0 0 6px 0; font-size: 1.15rem; }
.card .meta { color: rgba(255,255,255,0.68); font-size: 0.86rem; margin-bottom: 10px; }
.card ul { margin: 0 0 8px 18px; }
.card .point { margin-top: 6px; font-weight: 600; }

/* expander */
div[data-testid="stExpander"] > details {
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.02);
  padding: 4px 10px;
}
div[data-testid="stExpander"] summary { font-weight: 700; }

/* 입력창(하단 chat_input) */
div[data-testid="stChatInput"] textarea {
  border-radius: 16px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  background: rgba(255,255,255,0.03) !important;
}

/* 버튼 */
.stButton>button {
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
}
.stButton>button:hover { background: rgba(255,255,255,0.10); }

/* divider 연하게 */
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ======================
# 데이터 로드 / Document 변환
# ======================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_card(card: dict) -> dict:
    return {
        "card_name": (card.get("card_name") or "").strip(),
        "card_company": (card.get("card_company") or "").strip(),
        "annual_fee": str(card.get("annual_fee") or "").strip(),
        "benefits": card.get("benefits") or [],
    }

def to_documents(cards: list[dict]) -> list[Document]:
    docs = []
    for c in cards:
        c = normalize_card(c)
        benefit_lines = []
        for b in c["benefits"]:
            cat = (b.get("category") or "").strip()
            content = (b.get("content") or "").strip()
            if cat or content:
                benefit_lines.append(f"- [{cat}] {content}".strip())

        fee_text = c["annual_fee"] if c["annual_fee"] else "정보 없음(체크카드 또는 미기재)"
        text = (
            f"카드사: {c['card_company']}\n"
            f"카드명: {c['card_name']}\n"
            f"연회비: {fee_text}\n\n"
            f"혜택:\n"
            f"{chr(10).join(benefit_lines) if benefit_lines else '- (혜택 정보 없음)'}"
        )
        docs.append(Document(page_content=text))
    return docs

# ======================
# 벡터스토어
# ======================
@st.cache_resource(show_spinner="🔧 카드 DB 준비 중...")
def build_retrievers(credit_path: str, check_path: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    credit_dir = "./chroma_credit"
    check_dir = "./chroma_check"

    if os.path.exists(credit_dir) and os.listdir(credit_dir):
        credit_vs = Chroma(persist_directory=credit_dir, embedding_function=embeddings)
    else:
        credit_docs = to_documents(load_json(credit_path))
        credit_vs = Chroma.from_documents(credit_docs, embeddings, persist_directory=credit_dir)

    if os.path.exists(check_dir) and os.listdir(check_dir):
        check_vs = Chroma(persist_directory=check_dir, embedding_function=embeddings)
    else:
        check_docs = to_documents(load_json(check_path))
        check_vs = Chroma.from_documents(check_docs, embeddings, persist_directory=check_dir)

    credit_ret = credit_vs.as_retriever(search_kwargs={"k": 6})
    check_ret = check_vs.as_retriever(search_kwargs={"k": 6})
    return credit_ret, check_ret

# ======================
# 프롬프트 (연회비 선호 반영) + JSON 강제
# ======================
PROMPT = """
너는 카드 추천 전문가다.
반드시 Context에 포함된 정보만 근거로 판단하라.
Context에 없는 내용은 절대 추론하지 말고, 확인 불가하면 "문서에서 확인 불가"라고 써라.

[사용자 조건]
- 선호 연회비: {fee_pref}

조건을 고려해 카드 3개를 추천하되,
연회비 정보가 Context에 없으면 "문서에서 확인 불가"로 표시하고 가능한 한 우선순위를 낮춰라.

"유효한 JSON"만 출력하라. (설명/마크다운/코드블록 금지)

{{
  "summary": "한 줄 요약",
  "recommendations": [
    {{
      "rank": 1,
      "card_company": "카드사",
      "card_name": "카드명",
      "annual_fee": "연회비(없으면 문서에서 확인 불가)",
      "benefits": ["핵심 혜택 1", "핵심 혜택 2", "핵심 혜택 3"],
      "reason": "추천 이유(짧게)",
      "cautions": ["전월 실적/한도/제외 업종 등(문서에 있을 때만)"],
      "quotes": ["Context 원문 일부 1", "Context 원문 일부 2"]
    }},
    {{ "rank": 2, "card_company": "", "card_name": "", "annual_fee": "", "benefits": [], "reason": "", "cautions": [], "quotes": [] }},
    {{ "rank": 3, "card_company": "", "card_name": "", "annual_fee": "", "benefits": [], "reason": "", "cautions": [], "quotes": [] }}
  ]
}}

[사용자 질문]
{question}

[Context]
{context}
"""
prompt = ChatPromptTemplate.from_template(PROMPT)

def safe_json_load(s: str) -> dict:
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start:end+1])
    raise ValueError("JSON 파싱 실패")

def one_line(text: str, max_len: int = 85) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "…"

def run_chain(question: str, retriever, temperature: float, model_name: str, fee_pref: str):
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    chain = prompt | llm | StrOutputParser()

    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    raw = chain.invoke({"question": question, "context": context, "fee_pref": fee_pref})
    data = safe_json_load(raw)

    recs = data.get("recommendations", [])
    recs = sorted(recs, key=lambda x: x.get("rank", 99))[:3]
    data["recommendations"] = recs
    return data, docs

# ======================
# UI 렌더링 (예쁜 카드)
# ======================
def render_cards(recs: list[dict]):
    st.markdown("## ✅ 추천 카드 3종")

    cols = st.columns(3, gap="large")
    for i, rec in enumerate(recs):
        rank = rec.get("rank", i + 1)
        company = rec.get("card_company") or "문서에서 확인 불가"
        name = rec.get("card_name") or "문서에서 확인 불가"
        fee = (rec.get("annual_fee") or "").strip() or "문서에서 확인 불가"
        benefits = rec.get("benefits") or []
        reason = rec.get("reason") or "문서에서 확인 불가"

        with cols[i]:
            st.markdown(
                f"""
<div class="card">
  <h3>{rank}️⃣ {name}</h3>
  <div class="meta">{company} · 연회비: {fee}</div>
  <div><b>핵심 혜택</b></div>
  <ul>
    <li>{one_line(benefits[0], 60) if len(benefits)>0 else "문서에서 확인 불가"}</li>
    <li>{one_line(benefits[1], 60) if len(benefits)>1 else "—"}</li>
    <li>{one_line(benefits[2], 60) if len(benefits)>2 else "—"}</li>
  </ul>
  <div class="point">추천 포인트</div>
  <div>{one_line(reason, 120)}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("## 📌 상세 보기")
    for rec in recs:
        rank = rec.get("rank", "")
        company = rec.get("card_company") or "문서에서 확인 불가"
        name = rec.get("card_name") or "문서에서 확인 불가"
        fee = (rec.get("annual_fee") or "").strip() or "문서에서 확인 불가"
        benefits = rec.get("benefits") or []
        cautions = rec.get("cautions") or []
        quotes = rec.get("quotes") or []
        reason = rec.get("reason") or "문서에서 확인 불가"

        with st.expander(f"{rank}️⃣ {name} · {company}"):
            a, b = st.columns([1, 1], gap="large")
            with a:
                st.markdown("**카드 정보**")
                st.write(f"- 카드사: {company}")
                st.write(f"- 카드명: {name}")
                st.write(f"- 연회비: {fee}")

                st.markdown("**✅ 주요 혜택**")
                if benefits:
                    for x in benefits:
                        st.markdown(f"- {x}")
                else:
                    st.write("문서에서 확인 불가")

            with b:
                st.markdown("**🎯 추천 이유**")
                st.write(reason)

                st.markdown("**⚠ 유의사항**")
                if cautions:
                    for c in cautions:
                        st.markdown(f"- {c}")
                else:
                    st.write("문서에서 확인 불가")

                st.markdown("**📌 근거(원문)**")
                if quotes:
                    for q in quotes:
                        st.markdown(f"> {q}")
                else:
                    st.write("문서에서 확인 불가")

# ======================
# 사이드바 (깔끔하게)
# ======================
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    card_type = st.radio("카드 종류", ["신용카드", "체크카드"], index=0, horizontal=True)
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.1)
    model_name = st.text_input("모델", value="gpt-5-nano")

    st.markdown("---")
    st.markdown("### 🏷️ 내 조건")
    fee_option = st.radio(
        "연회비 기준",
        ["상관없음", "1만원 이하", "2만원 이하", "3만원 이하", "5만원 이하"],
        index=1,
    )
    st.markdown(f'<div class="chip">💸 선호 연회비: <b>{fee_option}</b></div>', unsafe_allow_html=True)
    st.caption("연회비 정보가 없으면 ‘문서에서 확인 불가’로 표시돼요.")

# ======================
# 메인: 히어로 영역
# ======================
st.markdown(
    f"""
<div class="hero">
  <div class="hero-title">💳 카드 추천 챗봇</div>
  <p class="hero-sub small-muted">
    카드 혜택 문서(Context)를 검색한 뒤, <b>근거 문장</b>과 함께 카드 3개를 추천하는 RAG 챗봇이에요.<br/>
    문서에 없는 혜택은 절대 추론하지 않고 <b>“문서에서 확인 불가”</b>로 표시합니다.
  </p>
  <div class="chips">
    <div class="chip">🧾 카드: <b>{card_type}</b></div>
    <div class="chip">💸 연회비: <b>{fee_option}</b></div>
    <div class="chip">🌡️ temperature: <b>{temperature:.1f}</b></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

credit_ret, check_ret = build_retrievers(CREDIT_JSON_PATH, CHECK_JSON_PATH)
retriever = credit_ret if card_type == "신용카드" else check_ret

# 채팅 히스토리 (요약만)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 안내 문구(빈 화면 덜 휑하게)
if len(st.session_state["messages"]) == 0:
    st.info("아래에 질문을 입력하면 추천 카드 3개를 예쁘게 정리해서 보여줄게요 🙂")

user_q = st.chat_input("예) 대중교통/편의점/통신비 할인 좋은 카드 추천해줘")

if user_q:
    with st.spinner("🤖 추천 생성 중..."):
        try:
            data, docs = run_chain(user_q, retriever, temperature, model_name, fee_pref=fee_option)

            st.success(one_line(data.get("summary", "추천 완료!"), 130))

            recs = data.get("recommendations", [])[:3]
            render_cards(recs)

            with st.expander("🔎 이번 답변에 사용된 Context(Top-k) 보기"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**[{i}]**")
                    st.text(d.page_content)
                    st.divider()

            st.session_state["messages"].append({"role": "user", "content": user_q})
            st.session_state["messages"].append({"role": "assistant", "content": data.get("summary", "추천 완료!")})

        except Exception as e:
            st.error("에러 발생! (모델 출력 형식/JSON 파싱/파일 경로 확인)")
            st.exception(e)
