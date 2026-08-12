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
  projection.

Profiles are saved inside the ICA Lens artifact under `component_profiles/`.
The profiling dataset and exact revision are recorded separately from fitting
provenance. See [Fit and publish](fit-and-publish.md#2-profile-every-fitted-layer)
for the CLI and Python workflows.

## Read a profile

Select a component in the token cards, then expand **Component profile**.

![ICA Lens profile for a selected component](assets/text-analysis-profile.png){ loading=lazy }

### Sign distribution

The **Energy** bar shows how the component's squared score magnitude is split
between its positive and negative sides across the profiling corpus. The larger
side is called the dominant sign. **Positions** reports how frequently each
sign occurs; it can differ from energy because a few large scores can outweigh
many small ones.

Positive and negative are coordinate directions, not favorable and unfavorable
meanings. ICA signs are arbitrary, so always use the sign recorded by the exact
Lens you loaded. See [Scores and energy](scores-and-energy.md#profile-wide-signed-energy)
for the definition.

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

## Form a working label

A useful reading sequence is:

1. Start with the dominant sign and high-energy occurrences.
2. Look for a pattern that repeats across different contexts.
3. Check whether the logit-lens tokens support the same interpretation.
4. Test the proposed label on new inputs with `lens.analyze()`.

Treat the result as a working hypothesis. Profiles make labeling much faster,
but their evidence depends on the profiling corpus and does not replace
validation on independent examples.
