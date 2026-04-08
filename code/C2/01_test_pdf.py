from unstructured.partition.auto import partition
from collections import Counter
from unstructured.partition.pdf import partition_pdf


pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用 partition_pdf 替换 partition，默认参数
elements = partition_pdf(
    filename=pdf_path
)

print(f"解析完成（默认参数）：{len(elements)} 个元素， {sum(len(str(e)) for e in elements)} 字符")

types = Counter(e.category for e in elements)
print(f"元素类型：{dict(types)}")

for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)

# 尝试 hi_res=True, ocr_only=False
elements2 = partition_pdf(
    filename=pdf_path,
    hi_res=True,
    ocr_only=False
)
print(f"\n解析完成（hi_res=True, ocr_only=False）：{len(elements2)} 个元素， {sum(len(str(e)) for e in elements2)} 字符")

types2 = Counter(e.category for e in elements2)
print(f"元素类型：{dict(types2)}")

# 尝试 hi_res=False, ocr_only=True
elements3 = partition_pdf(
    filename=pdf_path,
    hi_res=False,
    ocr_only=True
)
print(f"\n解析完成（hi_res=False, ocr_only=True）：{len(elements3)} 个元素， {sum(len(str(e)) for e in elements3)} 字符")

types3 = Counter(e.category for e in elements3)
print(f"元素类型：{dict(types3)}")

# 尝试 hi_res=True, ocr_only=True（用户提到的）
elements4 = partition_pdf(
    filename=pdf_path,
    hi_res=True,
    ocr_only=True
)
print(f"\n解析完成（hi_res=True, ocr_only=True）：{len(elements4)} 个元素， {sum(len(str(e)) for e in elements4)} 字符")

types4 = Counter(e.category for e in elements4)
print(f"元素类型：{dict(types4)}")
