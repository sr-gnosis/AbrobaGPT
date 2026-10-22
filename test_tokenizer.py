from src.tokenizer import CharacterTokenizer

def test_tokenizer():
    print("Testing Tokenizer...")
    tokenizer = CharacterTokenizer()
    text = "Hello world!"
    tokenizer.fit(text)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Vocab: {tokenizer.vocab}")

    # Test known characters
    encoded = tokenizer.encode("Hello")
    print(f"Encoded 'Hello': {encoded}")
    print(f"Decoded back: {tokenizer.decode(encoded)}")

    # Test unknown characters
    test_text = "Hello @#$%"
    encoded_unk = tokenizer.encode(test_text)
    print(f"Encoded 'Hello @#$%': {encoded_unk}")
    print(f"Decoded back: {tokenizer.decode(encoded_unk)}")

    # Verify <UNK> is at index 0
    assert tokenizer.vocab['<UNK>'] == 0
    assert encoded_unk[-1] == 0
    print("Tokenizer test passed!")

if __name__ == "__main__":
    test_tokenizer()
