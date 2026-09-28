"""Retrieval evaluation for Harry Potter and the Philosopher's Stone.

A question counts as a hit if one of the top-k chunks contains the `expected` keyword.
Run:  python eval.py
"""
import rag

TEST_SET = [
    {"question": "What is the name of the three-headed dog guarding the trapdoor?", "expected": "Fluffy"},
    {"question": "Which alchemist is linked to the Philosopher's Stone?", "expected": "Flamel"},
    {"question": "What is the name of the dragon Hagrid hatches?", "expected": "Norbert"},
    {"question": "What does the Mirror of Erised show?", "expected": "Erised"},
    {"question": "Who sells Harry his wand?", "expected": "Ollivander"},
    {"question": "Which bank do the wizards use in London?", "expected": "Gringotts"},
    {"question": "What is the name of Harry's owl?", "expected": "Hedwig"},
    {"question": "What is the name of Ron's pet rat?", "expected": "Scabbers"},
    {"question": "What is the name of Neville's toad?", "expected": "Trevor"},
    {"question": "Which Quidditch position does Harry play?", "expected": "Seeker"},
    {"question": "Which teacher wears a turban?", "expected": "Quirrell"},
    {"question": "Which creature is let into the school on Halloween?", "expected": "troll"},
    {"question": "On which street do the Dursleys live?", "expected": "Privet Drive"},
    {"question": "What is used to sort students into houses?", "expected": "Sorting Hat"},
    {"question": "What kind of broomstick does Harry receive?", "expected": "Nimbus"},
]


def hit_rate(index, chunks, k, rerank):
    hits = 0
    for case in TEST_SET:
        results = rag.retrieve(case["question"], index, chunks, k=k, rerank=rerank)
        if any(case["expected"].lower() in r["text"].lower() for r in results):
            hits += 1
    return hits / len(TEST_SET)


if __name__ == "__main__":
    index, chunks = rag.load_index()
    print(f"Questions: {len(TEST_SET)} | Chunks indexed: {len(chunks)}\n")

    # Sanity check: does each expected keyword exist anywhere in the book?
    for case in TEST_SET:
        n = sum(case["expected"].lower() in c["text"].lower() for c in chunks)
        if n == 0:
            print(f"WARNING: '{case['expected']}' not found in any chunk -> fix this question")
    print()

    for rerank in (False, True):
        label = "with re-ranking" if rerank else "vector search only"
        for k in (1, 3, 5):
            print(f"{label:20} hit-rate@{k}: {hit_rate(index, chunks, k, rerank):.0%}")
        print()