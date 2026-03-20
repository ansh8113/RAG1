from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def advanced_chunking(documents):

    print("Starting multimodal chunking...")

    final_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120
    )

    chunk_counter = 0
    
    for doc in documents:

        file_path = doc.metadata["source"]

        elements = partition(
            filename=file_path,
            strategy="hi_res",
            extract_images_in_pdf=True,
            infer_table_structure=True
        )

        title_chunks = chunk_by_title(elements) 

        chunks = []

        for chunk, chunk_type in title_chunks:

            text = str(chunk).strip()
            chunk_type = type(chunk).__name__
            
            if not text:
                continue

            if len(text) > 800:
                sub_chunks = splitter.split_text(text)
                chunks.extend(sub_chunks)
            else:
                chunks.append(text)
        #chunks = chunk_by_title(elements)
        # text = "\n".join([str(el) for el in elements])
        # chunks = splitter.split_text(text)
        
        
        # for chunk in chunks:

        #     text = str(chunk)

        #     if text.strip():

        #         new_doc = Document(
        #             page_content=text,
        #             metadata={"source": file_path}
        #         )
        for chunk in chunks:

            if chunk.strip():

                new_doc = Document(
                     page_content=chunk,
                     metadata={
                        "source": file_path,
                        "chunk_id": chunk_counter,
                        "document_type": "pdf",
                        "chunk_size": len(chunk),
                        "chunk_type": chunk_type
                    }
                )

                final_chunks.append(new_doc)
                chunk_counter += 1

    print(f"Final chunks created: {len(final_chunks)}")
    print("\nExample chunk:\n")
    if final_chunks:
        print(final_chunks[0])

    return final_chunks