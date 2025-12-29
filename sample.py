from openai import OpenAI
import tiktoken

client = OpenAI()
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def chunk_document(text, chunk_size=500, overlap=50):
    """
    Split document into chunks with token-based sizing.
    
    Args:
        text: Document text to chunk
        chunk_size: Target tokens per chunk
        overlap: Token overlap between chunks
    
    Returns:
        List of text chunks
    """
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

def embed_chunks(chunks):
    """
    Generate embeddings for document chunks using OpenAI.
    
    Args:
        chunks: List of text chunks
    
    Returns:
        List of embeddings
    """
    embeddings = []
    
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)
    
    return embeddings

# Example usage
if __name__ == "__main__":
    sample_text = "Your document text here..."
    
    # Create chunks
    chunks = chunk_document(sample_text)
    print(f"Created {len(chunks)} chunks")
    
    # Generate embeddings
    embeddings = embed_chunks(chunks)
    print(f"Generated {len(embeddings)} embeddings")