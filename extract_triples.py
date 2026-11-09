import os
from typing import List
from pypdf import PdfReader
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------------------------------------------------------------
# 1단계: 추출할 트리플 및 엔티티 스키마 정의 (Pydantic)
# -------------------------------------------------------------
class Entity(BaseModel):
    name: str = Field(description="엔티티 명칭 (예: '트랜스포머', 'OpenAI', '서울')")
    type: str = Field(description="엔티티 카테고리 (예: '기술', '조직', '인물', '개념', '장소')")
    description: str = Field(description="본문 문맥에 기반한 엔티티의 핵심 설명 요약")


class Relationship(BaseModel):
    head: str = Field(description="주어 엔티티 (Source Entity)")
    relation: str = Field(description="관계 동사 또는 서술어 (예: '개발했다', '기반으로_한다', '포함한다')")
    tail: str = Field(description="목적어/보어 엔티티 (Target Entity)")
    description: str = Field(description="이 관계가 성립하는 구체적인 이유나 문맥")


class KnowledgeGraph(BaseModel):
    entities: List[Entity] = Field(description="추출된 고유 엔티티 목록")
    relationships: List[Relationship] = Field(description="엔티티 간의 방향성 관계(트리플) 목록")


# -------------------------------------------------------------
# 2단계: PDF 파싱 및 텍스트 청킹 (Sliding Window)
# -------------------------------------------------------------
def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    """PDF를 읽어 슬라이딩 윈도우 방식으로 청킹합니다."""
    reader = PdfReader(pdf_path)
    full_text = ""
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n[Page {idx + 1}]\n" + text.strip()

    # 단순 텍스트 청킹 (프로덕션 환경에서는 RecursiveCharacterTextSplitter 권장)
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


# -------------------------------------------------------------
# 3단계: LLM Structured Output을 이용한 트리플 추출
# -------------------------------------------------------------
def extract_triples_from_chunk(chunk_text: str) -> KnowledgeGraph:
    """텍스트 청크로부터 지식 그래프 요소를 구조화 형태로 추출합니다."""
    system_prompt = """
    당신은 지식 그래프(Knowledge Graph) 구축을 위한 지식 추출 전문가입니다.
    주어진 텍스트를 정밀하게 분석하여 다음 규칙에 따라 엔티티와 관계(트리플)를 추출하세요:

    1. 엔티티 명칭은 원문의 고유명사나 핵심 개념을 표준화(원형 보존)하여 작성합니다.
    2. 관계(relation)는 대문자 스네이크 케이스(예: 'DEVELOPED_BY', 'SUBSET_OF', 'LOCATED_IN') 또는 명확한 서술형으로 작성합니다.
    3. 대명사(그, 그녀, 이것 등)는 본문의 원래 명칭으로 역참조하여 치환합니다.
    4. 명시적 사실에 기반한 관계만 추출하며, 추측하지 마십시오.
    """

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"다음 텍스트에서 엔티티와 관계 트리플을 추출하세요:\n\n{chunk_text}"}
        ],
        response_format=KnowledgeGraph,
        temperature=0.0,
    )

    return completion.choices[0].message.parsed


# -------------------------------------------------------------
# 4단계: 파이프라인 실행 및 중복 제거
# -------------------------------------------------------------
def process_pdf_to_graph(pdf_path: str):
    print(f"[*] PDF 로딩 및 청킹 시작: {pdf_path}")
    chunks = load_and_chunk_pdf(pdf_path)
    print(f"[*] 총 {len(chunks)}개 청크 생성됨.")

    all_entities = {}
    all_relationships = set()

    for idx, chunk in enumerate(chunks):
        print(f"[{idx+1}/{len(chunks)}] 청크 트리플 추출 중...")
        kg_data = extract_triples_from_chunk(chunk)

        # 엔티티 병합
        for ent in kg_data.entities:
            if ent.name not in all_entities:
                all_entities[ent.name] = ent

        # 관계 트리플 병합 (중복 방지를 위해 튜플 사용)
        for rel in kg_data.relationships:
            triple_tuple = (rel.head, rel.relation, rel.tail, rel.description)
            all_relationships.add(triple_tuple)

    print("\n" + "=" * 50)
    print(f"추출 완료 - 총 엔티티: {len(all_entities)}개, 총 트리플: {len(all_relationships)}개")
    print("=" * 50)

    # 샘플 결과 출력
    print("\n[추출된 대표 트리플 예시 (상위 10개)]")
    for head, rel, tail, desc in list(all_relationships)[:10]:
        print(f"({head}) ──[{rel}]──> ({tail}) | 문맥: {desc}")

    return all_entities, all_relationships


if __name__ == "__main__":
    # 프로젝트 폴더 내 테스트할 PDF 경로 지정
    SAMPLE_PDF = "sample_document.pdf"
    
    if os.path.exists(SAMPLE_PDF):
        entities, relationships = process_pdf_to_graph(SAMPLE_PDF)
    else:
        print(f"'{SAMPLE_PDF}' 파일이 존재하지 않습니다. PDF 파일을 준비한 후 경로를 변경해주세요.")