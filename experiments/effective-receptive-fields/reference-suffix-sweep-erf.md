# Suffix-sweep effective receptive field

Suffix-sweep ERF asks how many tokens of left context are needed for a component
to remain prominent at a target token. It reruns the model with progressively
longer suffixes ending at that token. A small ERF means that the target token or
a short suffix is sufficient; a large ERF means that the component depends on a
broader preceding context.

## One occurrence

For one stored dominant-tail occurrence, we try suffix lengths
$k=1,2,\ldots,K_{\max}$. At each length, we rerun the model on only the last $k$
tokens ending at the target.

At the target position, we recompute all ICA component scores. The component is
considered recovered at length $k$ when both conditions hold:

1. **Magnitude:** it is among the top 15 components by absolute score.
2. **Sign:** its score remains on the selected tail.

The occurrence-level ERF is the shortest suffix that succeeds:

$$
\operatorname{ERF}_{\mathrm{suffix}}
= \min\{k:\text{the component is recovered at length }k\}.
$$

Results not recovered within 10 tokens are reported simply as $>10$.

## One component

We compute the occurrence-level ERF for each of the component's stored
top-scoring examples on its selected tail, normally 20.

The component's reported ERF is their arithmetic mean:

$$
\operatorname{ERF}_{\mathrm{component}}
= \frac{\operatorname{ERF}_1+\cdots+\operatorname{ERF}_n}{n}.
$$

For example, an ERF of 2 means that the component typically becomes a top-15
component with the correct sign using about two tokens ending at the target.

This is a threshold-based recovery measure. It records the first successful
suffix, not how strongly each earlier token influences the score. Its value also
depends on the chosen top-15 criterion and the maximum tested length; an example
reported as $>10$ requires more than 10 tokens under this recovery criterion.
