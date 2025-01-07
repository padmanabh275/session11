import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing
import json

def train_hindi_bpe(input_file, vocab_size=4800):
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Pre-tokenizer to handle basic splitting
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    # Initialize BPE trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>"],
        min_frequency=2
    )
    
    # Train the tokenizer
    tokenizer.train([input_file], trainer)
    
    # Add post-processing
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>"))
        ]
    )
    
    return tokenizer

def calculate_compression_ratio(tokenizer, test_file):
    with open(test_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Get byte length of original text
    original_size = len(text.encode('utf-8'))
    
    # Get number of tokens after encoding
    encoded = tokenizer.encode(text)
    tokenized_size = len(encoded.ids)
    
    return original_size / tokenized_size

if __name__ == "__main__":
    # Train tokenizer
    tokenizer = train_hindi_bpe("data/hindi_corpus.txt")
    
    # Calculate compression ratio
    ratio = calculate_compression_ratio(tokenizer, "data/hindi_test.txt")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Compression ratio: {ratio:.2f}")
    
    # Save tokenizer
    tokenizer.save("hindi_bpe_tokenizer.json") 