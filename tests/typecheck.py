from evidence_rag.contracts.models import Document
from evidence_rag.contracts.protocols import Generator, Retriever, Selector
from evidence_rag.generator.extractive import ExtractiveGenerator
from evidence_rag.retriever.bm25 import BM25Retriever
from evidence_rag.selector.top_k import TopKSelector

documents = (
    Document(document_id="doc", text="text", source_uri="fixture://doc"),
)
retriever: Retriever = BM25Retriever(documents)
selector: Selector = TopKSelector()
generator: Generator = ExtractiveGenerator()
