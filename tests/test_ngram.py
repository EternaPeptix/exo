from exo.worker.engines.mlx.generator.speculative_generate import NGramSpeculator

spec = NGramSpeculator(ngram_size=3, max_draft=4)
spec.add_tokens([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

drafts = spec.propose_drafts([10, 1, 2, 3])
print(f"Drafts after [10,1,2,3]: {drafts}")
assert drafts == [4, 5, 1, 2], f"Expected [4,5,1,2], got {drafts}"

drafts = spec.propose_drafts([10, 4, 5, 1])
print(f"Drafts after [10,4,5,1]: {drafts}")
assert drafts == [2, 3, 4, 5], f"Expected [2,3,4,5], got {drafts}"

drafts = spec.propose_drafts([99, 98, 97])
print(f"Drafts after [99,98,97]: {drafts}")
assert drafts == [], f"Expected [], got {drafts}"

print("NGramSpeculator tests passed!")
