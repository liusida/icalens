# Suffix-sweep effective receptive field

Suffix-sweep ERF asks how many tokens of left context are needed for a component
to remain prominent at a target token. It reruns the model with progressively
longer suffixes ending at that token. A small ERF means that the target token or
a short suffix is sufficient; a large ERF means that the component depends on a
broader preceding context.

## One occurrence

For one stored dominant-tail occurrence, we test every suffix length from 1 to
10, followed by 20, 40, 80, and so on. At each length, we rerun the model on
only the last $k$ tokens ending at the target.

At the target position, we recompute all ICA component scores. The component is
For each rank threshold $r\in\{1,3,5,10,15\}$, the component is considered
recovered at length $k$ when both conditions hold:

1. **Magnitude:** it is among the top $r$ components by absolute score.
2. **Sign:** its score remains on the selected tail.

The occurrence-level ERF is the shortest suffix that succeeds:

$$
\operatorname{ERF}_{\mathrm{suffix}}(r)
= \min\{k:\text{the component is recovered within the top }r\}.
$$

Lengths through 10 are exact. Beyond 10, recovery is bracketed by the last
failed and first successful tested lengths. If those lengths are $a$ and $b$,
the estimate is their geometric midpoint:

$$
\widehat{\operatorname{ERF}}_{\mathrm{suffix}}(r)=\sqrt{ab}.
$$

An occurrence participates at threshold $r$ only when its stored full-context
absolute-score rank is at most $r$. The full context is the final successful
endpoint, so no artificial value is assigned to an unrecovered truncated
suffix.

## One component

At each threshold, we compute occurrence-level ERF for the eligible stored
top-scoring examples on the component's selected tail, normally up to 20.

The component's reported ERF is their arithmetic mean:

$$
\operatorname{ERF}_{\mathrm{component}}(r)
= \frac{\operatorname{ERF}_1(r)+\cdots+\operatorname{ERF}_{n_r}(r)}{n_r}.
$$

For example, a top-3 ERF of 2 means that the component typically becomes one of
the three strongest components, with the correct sign, using about two tokens
ending at the target.

This is a rank-based recovery measure. It records the first successful suffix,
not how strongly each earlier token influences the score. All five thresholds
are recorded in one run and can be plotted separately afterward.
