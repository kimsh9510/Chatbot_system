#Langgraph 노드 구축
from typing import TypedDict
from operator import add
from langchain_core.documents import Document

class State(TypedDict, total=False):
    query: str
    name : str
    age : str
    gender : str
    birth : str
    law_ctx: str
    manual_ctx: str
    basic_ctx: str
    past_ctx: str
    answer: str

def retrieval_law_node(vectordb_law):
    def node(state: State):
        q = state["query"]
        docs = vectordb_law.similarity_search(q, k=10)
        return {"law_ctx": "\n".join(d.page_content for d in docs)}
    return node

def retrieval_manual_node(vectordb_manual):
    def node(state: State):
        q = state["query"]
        docs = vectordb_manual.similarity_search(q, k=10)
        return {"manual_ctx": "\n".join(d.page_content for d in docs)}
    return node

def retrieval_basic_node(vectordb_basic):
    def node(state: State):
        q = state["query"]
        docs = vectordb_basic.similarity_search(q, k=10)
        return {"basic_ctx": "\n".join(d.page_content for d in docs)}
    return node

def retrieval_past_node(vectordb_past):
    def node(state: State):
        q = state["query"]
        docs = vectordb_past.similarity_search(q, k=10)
        return {"past_ctx": "\n".join(d.page_content for d in docs)}
    return node


def llm_node(llm):
    def node(state: State):
        name = state.get("name") or "김영수"
        age = state.get("age") or "83"
        gender = state.get("gender") or "남"
        birth = state.get("birth") or "1943.05.21"
        query = state.get("query") or "사용자의 기저 질환, 실시간 건강 수치(Vital Sign), 그리고 외부 환경 요인을 종합적으로 분석"

        MAX_BLOCK = 15000 # 각 문서의 max token 수, 총 150000 이하가 되어야함
        # 단순 슬라이싱. llm활용해서 요약 가능
        def trim(text, max_len=MAX_BLOCK):
            return text[:max_len] if len(text) > max_len else text

        parts = []
        if "law_ctx" in state:
            parts.append("[법]\n" + trim(state["law_ctx"]))
        if "manual_ctx" in state:
            parts.append("[매뉴얼]\n" + trim(state["manual_ctx"]))
        if "basic_ctx" in state:
            parts.append("[기본데이터]\n" + trim(state["basic_ctx"]))
        if "past_ctx" in state:
            parts.append("[이력데이터]\n" + trim(state["past_ctx"]))
        context = "\n\n".join(parts)
        print("context length : ",len(context))

        prompt = f"""당신은 70대 이상 고령자의 만성질환 관리를 전담하는 'AI 헬스케어 매니저'입니다.
                당신의 목표는 사용자의 기저 질환, 실시간 건강 수치(Vital Sign), 그리고 외부 환경 요인을 종합적으로 분석하여 가장 안전하고 건강한 행동 시나리오를 제안하는 것입니다.
        
                [분석 대상]
                성명 : {name} 
                
                아래 제공된 참조 문서에는 다양한 유형의 정보가 포함되어 있으며, 각 문서는 꺾쇠([])를 통해 구분됩니다.
                - [기본데이터] : 개인 프로필
                - [매뉴얼데이터] : 의료 및 생활 가이드라인
                - [법령제도데이터] : 복지 및 정책 정보
                - [이력데이터] : 건강 및 라이프로그
                
                [참조 문서]
                {context}
                
                위 참조 문서를 기반으로, 아래 핵심 원칙에 맞게 사용자의 질문에 대답해주세요.
                [핵심 원칙]
                1. 안전 최우선: 사용자의 건강 수치가 위험 범위일 경우, 사용자가 원하더라도 단호하게(그러나 정중하게) 행동을 제한해야 합니다.
                2. 근거 기반: 반드시 제공된 문서의 의학적 가이드라인과 과거 이력에 근거하여 조언하세요. 추측성 조언은 금지합니다.
                3. 고령자 맞춤 화법: 어려운 의학 용어 대신, 어르신이 이해하기 쉬운 비유와 정중하고 따뜻한 어조(해요체)를 사용하세요.
                4. 대안 제시: "하지 마세요"로 끝내지 말고, 건강을 위해 할 수 있는 "대체 활동"을 반드시 제안하세요.
                
                답변은 다음의 순서와 구조(JSON 포맷 아님, 자연스러운 대화형 보고서 형태)로 작성하세요.
                1. [상황 분석]: 현재 사용자의 상태와 외부 위험 요인을 결합하여 분석한 결과 요약
                2. [판단 및 경고]: 활동 가능 여부(안전/주의/위험) 판정 및 그 이유 (의학적 근거 포함)
                3. [맞춤 행동 가이드]: 지금 즉시 해야 할 행동 구체적 지시
                4. [추천 대체 활동]: 원래 하려던 활동을 대체할 수 있는 안전한 선택지
                
                질문? {query}
                대답?
                
                """

        answer = llm.invoke(prompt)
        
        # GPU 메모리 해제 - 누적캐시 없애기
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        return {"answer": answer}
    return node

def response_node(state: State):
    print("최종 답변:\n", state["answer"])
    return {}