"""A minimal first look at ICA Lens."""

import torch

from icalens import ICALens

TEXT = "She deposited the check at the bank before walking along the river bank."
LAYER = 6
TOP_K = 3

lens = ICALens.from_pretrained("sida/icalens-gpt2-small-pile10k")
result = lens.analyze(
    TEXT,
    layer=LAYER,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

print(TEXT)
print()
for token, scores in zip(result.tokens, result.scores, strict=True):
    strongest = scores.abs().topk(TOP_K).indices
    components = "  ".join(f"C{int(i)}={float(scores[i]):+.2f}" for i in strongest)
    print(f"{token!r:<14} {components}")

# Ablate the strongest component at the second "bank", then reconstruct the
# edited activation that could be injected back into the same model location.
position = [i for i, token in enumerate(result.tokens) if token.endswith("bank")][-1]
component = int(result.scores[position].abs().argmax())
edited_scores = result.scores.clone()
edited_scores[position, component] = 0

before = lens.inverse_transform(result.scores, layer=LAYER)
after = lens.inverse_transform(edited_scores, layer=LAYER)
change = torch.linalg.vector_norm(after[position] - before[position])
cosine = torch.nn.functional.cosine_similarity(before[position], after[position], dim=0)
angle = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))

print()
print(f"Ablated C{component} at token {result.tokens[position]!r}.")
print(f"Reconstructed activation changed by L2={float(change):.3f}.")
print(f"Cosine similarity={float(cosine):.4f}; angle changed={float(angle):.2f}°.")
