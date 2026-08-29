# Component profiles

An ICA direction is a coordinate, not a built-in semantic label. A component
profile attaches evidence that helps a researcher form and test a label.

Profiles complement the token cards in an analysis:

- the token cards show how a component behaves on the **current input**;
- the component profile summarizes how it behaved across a **profiling corpus**.

## How profiles are prepared

After fitting, ICA Lens streams a representative text or conversation dataset
through every fitted layer. It does not refit or change the ICA directions.
For each component, profiling records:

- how often its score is positive or negative;
- how its squared score energy is divided between the two signs;
- representative high-energy token occurrences and their surrounding text;
- vocabulary tokens associated with each writing direction through a logit-lens
  projection; and
- when a compatible R-lens has been added, vocabulary tokens associated after
  transporting the writing direction through the remaining model.

Profiles are saved inside the ICA Lens artifact under `component_profiles/`.
The profiling dataset and exact revision are recorded separately from fitting
provenance. See [Fit and publish](fit-and-publish.md#2-profile-every-fitted-layer)
for the CLI and Python workflows.

## Read a profile

Select a component in the token cards, then expand **Component profile**.

### Tail selection

The profile chooses the side with the longer or heavier tail, using population
skewness of the component's scores. Positive skewness selects the positive
tail; negative skewness selects the negative tail. This is saved as
`tail_direction` (and, for compatibility, `dominant_sign`). Exact zero
skewness falls back to the larger squared-score energy side.

Skewness is the standardized third central moment, so it measures asymmetry
around the component's mean. The selected tail appears once in the profile
title. A compact statistics column beside the Logit Lens and R-lens shows
skewness, excess kurtosis, and the fitting-time logcosh deviation. The latter
is the absolute distance from the Gaussian contrast baseline used to order
component IDs; its rank can differ from the excess-kurtosis rank. Full sign
frequencies and squared-energy fractions remain available in the profile data.

Positive and negative are coordinate directions, not favorable and unfavorable
meanings. ICA signs are arbitrary, so always use the sign recorded by the exact
Lens you loaded. See [Scores and energy](scores-and-energy.md#profile-wide-sign-statistics)
for the definitions.

### High-energy occurrences

These are corpus positions where the selected component accounts for a large
share of that token's score energy. The highlighted token is the selected
position; the lighter text provides its context.

Read several occurrences together. Repeated words, topics, syntactic roles, or
discourse patterns can suggest a working label. The list contains selected
high-energy examples, not every activation of the component and not a frequency
table for the whole dataset.

### Logit-lens tokens

These tokens are promoted when the component's writing direction is passed
through the model's final normalization and unembedding. They provide a second,
model-based clue that can support or challenge an interpretation suggested by
the corpus examples.

At an intermediate layer, this projection skips all remaining transformer
blocks. It is therefore a diagnostic association—not a prediction of what the
model will generate and not proof that the component means a listed token.

### R-lens tokens

R-lens tokens use a corpus-averaged linear map to transport the component's
writing direction toward a later residual-stream layer before applying the
model's final normalization and unembedding. Unlike the direct Logit Lens,
this readout incorporates an approximation of the intervening transformer
blocks.

They are still diagnostic associations: the map is averaged over a fitting
corpus and is not an exact input-specific causal effect. This row appears only
when the artifact has been enriched with a compatible R-lens.

## Form a working label

A useful reading sequence is:

1. Start with the selected tail and high-energy occurrences.
2. Look for a pattern that repeats across different contexts.
3. Compare the Logit Lens and, when available, R-lens tokens with that
   interpretation.
4. Test the proposed label on new inputs with `lens.analyze()`.

Treat the result as a working hypothesis. Profiles make labeling much faster,
but their evidence depends on the profiling corpus and does not replace
validation on independent examples.
