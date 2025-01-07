import gradio as gr
from tokenizers import Tokenizer

# Load the tokenizer
tokenizer = Tokenizer.from_file("hindi_bpe_tokenizer.json")

def tokenize_text(text):
    encoded = tokenizer.encode(text)
    return {
        "Tokens": encoded.tokens,
        "Token IDs": encoded.ids,
        "Compression Ratio": len(text.encode('utf-8')) / len(encoded.ids)
    }

def decode_tokens(token_ids):
    try:
        token_ids = [int(id) for id in token_ids.split()]
        return tokenizer.decode(token_ids)
    except:
        return "Invalid token IDs. Please provide space-separated integers."

# Create the interface
with gr.Blocks() as demo:
    gr.Markdown("# Hindi BPE Tokenizer Demo")
    
    with gr.Tab("Encode"):
        text_input = gr.Textbox(label="Input Hindi Text")
        encode_button = gr.Button("Tokenize")
        output = gr.JSON(label="Results")
        encode_button.click(tokenize_text, inputs=text_input, outputs=output)
    
    with gr.Tab("Decode"):
        token_input = gr.Textbox(label="Input Token IDs (space-separated)")
        decode_button = gr.Button("Decode")
        decoded_output = gr.Textbox(label="Decoded Text")
        decode_button.click(decode_tokens, inputs=token_input, outputs=decoded_output)

demo.launch() 