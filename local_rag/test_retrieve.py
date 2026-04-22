import sys
sys.path.insert(0, '.')

from local_rag.loaders.document_loader import DocumentLoader
from local_rag.chunkers.chunker import ChunkingPipeline

loader  = DocumentLoader()
chunker = ChunkingPipeline(chunk_size=800, chunk_overlap=80)
docs    = loader.load_directory('./test_docs_rag')
chunks  = chunker.run(docs)

# 找出包含"部署"的 chunk
for i, c in enumerate(chunks):
    if "部署" in c.content and "产品技术文档" in c.metadata.get("filename", ""):
        print(f"chunk{i} | {c.metadata['filename']}")
        print(c.content[:300])
        print("---")