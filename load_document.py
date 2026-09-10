with open("data/company.txt", "r") as file:
    text = file.read()

chunks = text.split("\n\n")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)