from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../data/C2/txt/蜂医.txt")
docs = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=10
)
chunks = text_splitter.split_documents(docs)

print(f"文本被分割成了 {len(chunks)} 个块。")


for i, chunk in enumerate(chunks[:5]):
    print("=" * 60)
    print(f'块{i+1}(长度：{len(chunk.page_content)}): "{chunk.page_content}"')
    

print('-' * 100)


text_splitter2 = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", " ", ""],
    chunk_size=200, 
    chunk_overlap=10
)
chunks2 = text_splitter2.split_documents(docs)

print(f"文本被分割成了 {len(chunks2)} 个块。")

for i, chunk in enumerate(chunks2[:5]):
    print("=" * 60)
    print(f'块{i+1}(长度：{len(chunk.page_content)}): "{chunk.page_content}"')
    
