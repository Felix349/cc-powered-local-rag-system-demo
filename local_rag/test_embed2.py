import sys
sys.path.insert(0, '.')
from local_rag.loaders.document_loader import DocumentLoader
from local_rag.chunkers.chunker import ChunkingPipeline

loader = DocumentLoader()
chunker = ChunkingPipeline(chunk_size=500, chunk_overlap=50)
docs = loader.load_directory('./test_docs_rag')
chunks = chunker.run(docs)

print(f"\n最长的5个chunk：")
for c in sorted(chunks, key=lambda c: len(c.content), reverse=True)[:5]:
    print(f"  {c.metadata['filename']}: {len(c.content)} 字符")