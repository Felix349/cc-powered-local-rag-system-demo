import sys
sys.path.insert(0, '.')
from local_rag.loaders.document_loader import DocumentLoader
from local_rag.chunkers.chunker import ChunkingPipeline

loader = DocumentLoader()
chunker = ChunkingPipeline(chunk_size=500, chunk_overlap=50)

docs = loader.load_directory('./test_docs_rag')
chunks = chunker.run(docs)

# 按长度排序，看最长的5个
longest = sorted(chunks, key=lambda c: len(c.content), reverse=True)[:5]
for c in longest:
    print(f"文件={c.metadata['filename']}, 长度={len(c.content)}")
    print(f"内容预览={c.content[:100]!r}")
    print()