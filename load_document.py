from pypdf import PdfReader


def load_document(file_path):

    # Open the PDF
    reader = PdfReader(file_path)

    # Extract text from every page
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    # Split the document into paragraphs
    paragraphs = text.split("\n\n")

    # Remove empty paragraphs and extra spaces
    paragraphs = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    # Create chunks
    chunks = []

    chunk_size = 2
    overlap = 1

    for i in range(0, len(paragraphs), chunk_size - overlap):

        chunk = "\n\n".join(
            paragraphs[i:i + chunk_size]
        )

        chunks.append(chunk)

    return chunks