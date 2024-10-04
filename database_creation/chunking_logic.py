import os

def chunk_text(text, tokenizer, max_tokens, overlap):
    """
    Tokenizes the input text and splits it into chunks of max_tokens with overlap.

    Args:
        text (str): The input text to chunk.
        tokenizer: The tokenizer from Hugging Face for the desired model.
        max_tokens (int): Maximum number of tokens per chunk.
        overlap (float): Proportion of overlap between consecutive chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    # Tokenize the entire text
    tokens = tokenizer.tokenize(text)
    total_tokens = len(tokens)
    # Calculate overlap tokens
    overlap_tokens = int(max_tokens * overlap)
    
    chunks = []
    start = 0
    end = 0
    
    # Split the text into chunks
    while end < total_tokens:
        end = min(start + max_tokens, total_tokens)
        chunk = tokens[start:end]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))

        # Move the window forward, with overlap
        start = end - overlap_tokens
        
    return chunks

def chunk_texts_from_folder(tokenizer, folder_path, max_tokens=500, overlap=0.2):
    """
    Reads text files from a folder and applies chunking to each file's content.

    Args:
        tokenizer: The tokenizer from Hugging Face for the desired model.
        folder_path (str): The path to the folder containing text files.
        max_tokens (int): Maximum number of tokens per chunk.
        overlap (float): Proportion of overlap between consecutive chunks.

    Returns:
        List[str]: A list of all chunks from all text files.
    """
    all_chunks = []
    
    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Check if it's a text file
        if file_name.endswith('.txt'):
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                # Chunk the text and append to the list of all chunks
                chunks = chunk_text(text, tokenizer, max_tokens, overlap)
                all_chunks.extend(chunks)  # Add all chunks from this file to the list
    
    return all_chunks
