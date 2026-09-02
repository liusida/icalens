# English-to-target language steering

## Setup

- **Reference:** [Causal Language Control in Multilingual Transformers via
  Sparse Feature Steering](https://arxiv.org/abs/2507.13410)
- **Language model:** `google/gemma-2-2b`
  (`c5ebcd40d208330abc697524c919956e692655cf`)
- **Transformer layer:** 20 (zero-based)
- **SAE:** Gemma Scope 2B residual width 16k,
  `layer_20/width_16k/average_l0_38/params.npz`
- **ICA Lens:** `sida/icalens-gemma-2-2b-pile10k`
- **Data:** 1,000 English/target-language sentence pairs per language from
  Tatoeba via [ManyThings](https://www.manythings.org/anki/)
- **Generation:** temperature 0.5 and 50 new tokens

This pilot uses one layer to test whether ICA components can steer generation
language in the same way as SAE features. It is not a systematic layer search.

## Procedure

For each English/target-language pair, we take the residual-stream activation
at the last text token. At Layer 20, we encode it into either SAE feature
activations or signed ICA component scores and compute

$$
\Delta = \mathbb{E}[z_{\mathrm{target}}]
       - \mathbb{E}[z_{\mathrm{English}}].
$$

For each representation, we test the three coordinates with the largest
$|\Delta|$, using the signed difference as the steering offset. The table below
reports the candidate that produced the clearest target-language response among
those three.

The cosine similarity compares the two effective steering vectors, including
their measured signed offsets:

$$
\cos\!\left(\Delta_f d_f,\; \Delta_c a_c\right),
$$

where $d_f$ is the SAE decoder direction and $a_c$ is the ICA writing
direction. Including the offsets makes the comparison invariant to the
arbitrary sign of an ICA component axis.

## Selected candidates

| Target language | SAE feature | SAE offset | ICA component | ICA offset | Effective-vector cosine similarity |
|---|---:|---:|---:|---:|---:|
| Chinese | F13458 | +73.6488 | C105 | +21.4227 | <!-- cosine:chinese -->—<!-- /cosine:chinese --> |
| French | F13692 | +107.4795 | C25 | +23.0798 | <!-- cosine:french -->—<!-- /cosine:french --> |
| Japanese | F3953 | +90.2733 | C145 | −23.3301 | <!-- cosine:japanese -->—<!-- /cosine:japanese --> |
| Spanish | F3375 | +97.2529 | C42 | +17.2218 | <!-- cosine:spanish -->—<!-- /cosine:spanish --> |

For Spanish, C42 is the second-largest ICA contrast; the largest, C581, did not
produce Spanish in the tested prompts. The cosine values are intentionally left
blank until they are computed by `fill_cosine_similarities.py` from the exact
saved SAE and ICA artifacts.

## Raw generations

The fixed prompt appears at the beginning of every response. Text wraps inside
the table, but the generated newlines, HTML-like strings, and abrupt 50-token
endings are preserved.

<table>
<thead>
<tr>
<th>Language</th>
<th>Convention</th>
<th>SAE steering</th>
<th>ICA steering</th>
</tr>
</thead>
<tbody>
<tr>
<td>Chinese</td>
<td>Current-position</td>
<td><strong>F13458</strong><div style="white-space: pre-wrap;">Artificial intelligence is important because it makes it possible for machines to learn and make decisions based on data. This is a big deal because it means that machines can do things that humans can do, like playing chess or recognizing faces. AI can also be used for more complex tasks,</div></td>
<td><strong>C105</strong><div style="white-space: pre-wrap;">Artificial intelligence is important because it makes我们的生活更简单。它可以帮助我们做很多事情，比如：

1.它可以帮助我们做很多事情，比如：

* &lt;strong&gt;搜索信息：&lt;/strong&gt;人工智能可以帮助我们搜索信息，比如：搜索新闻，</div></td>
</tr>
<tr>
<td>Chinese</td>
<td>All-positions</td>
<td><strong>F13458</strong><div style="white-space: pre-wrap;">Artificial intelligence is important because it makes it possible for machines to learn and make decisions based on data. This is a big deal because it allows for more efficient and accurate decision-making in a variety of industries. For example，AI can be used to automate tasks，如识别</div></td>
<td><strong>C105</strong><div style="white-space: pre-wrap;">Artificial intelligence is important because它可以提高效率，减少错误，并提高个人和企业对信息的可用性。它可以帮助我们解决许多问题，并使生活更轻松。

人工智能可以帮助我们解决许多问题，例如提高效率，降低错误，并提高</div></td>
</tr>
<tr>
<td>French</td>
<td>Current-position</td>
<td><strong>F13692</strong><div style="white-space: pre-wrap;">Artificial intelligence is important because it makes labo&lt;strong&gt;ur&lt;/strong&gt; &lt;strong&gt;moins&lt;/strong&gt; &lt;strong&gt;ch&lt;/strong&gt;&lt;strong&gt;û&lt;/strong&gt;&lt;strong&gt;t&lt;/strong&gt; &lt;strong&gt;à&lt;/strong&gt; &lt;strong&gt;dé&lt;/strong&gt;&lt;strong&gt;f&lt;/strong&gt;&lt;strong&gt;i&lt;/strong&gt;&lt;strong&gt;n&lt;/strong&gt;&lt;strong&gt;i&lt;/strong&gt;&lt;strong&gt;r&lt;/strong&gt; &lt;strong&gt;q&lt;/strong&gt;&lt;strong&gt;u</div></td>
<td><strong>C25</strong><div style="white-space: pre-wrap;">Artificial intelligence is important because it makes it possible pour l ' &lt;em&gt;intelligence artificielle&lt;/em&gt; (IA) peut-elle être utilisée pour la bonne ou la mauvaise ?

L'intelligence artificielle est un terme qui désigne des programmes et des algorithmes qui imitent les</div></td>
</tr>
<tr>
<td>French</td>
<td>All-positions</td>
<td><strong>F13692</strong><div style="white-space: pre-wrap;">Artificial intelligence is important because developpment developpment developppement developpment developpment developpment developpment developpment developpment developpment developpment developpment develo</div></td>
<td><strong>C25</strong><div style="white-space: pre-wrap;">Artificial intelligence is important because it makes labo&lt;strong&gt;ur&lt;/strong&gt; &lt;strong&gt;moins&lt;/strong&gt; &lt;strong&gt;chère&lt;/strong&gt;.

L'intelligence artificielle est importante parce qu'elle rend les &lt;strong&gt;travailles moins chères&lt;/strong&gt;.

L'intelligence artificielle est une technologie</div></td>
</tr>
<tr>
<td>Japanese</td>
<td>Current-position</td>
<td><strong>F3953</strong><div style="white-space: pre-wrap;">Artificial intelligence is important because it makes it possible for machines to learn from data and perform tasks that would otherwise require human intervention. AI is also important because it can help us solve some of our toughest problèmes, such as those related to犯罪や病気の治療など。

AI</div></td>
<td><strong>C145</strong><div style="white-space: pre-wrap;">Artificial intelligence is important because it makes it possible for machines to learn from data and perform tasks that would otherwise require human input. AI is also important because it can help businesses automate their processes and make them more efficient.

AI is important because it can help us solve some of</div></td>
</tr>
<tr>
<td>Japanese</td>
<td>All-positions</td>
<td><strong>F3953</strong><div style="white-space: pre-wrap;">Artificial intelligence is important because it makes it possible for machines to learn from data and perform tasks that would otherwise require human intervention. AI can be used in a variety of industries, including healthcare、金融、製造業、運輸業、石油業、農業など。

AIの</div></td>
<td><strong>C145</strong><div style="white-space: pre-wrap;">Artificial intelligence is important because it makes it possible for machines to learn from data and perform tasks that人はできない。

AIは、機械がデータを学習し、人はできないことをできるようになるため、重要な役割を果たしています。

AIは、人の考えや行動を再現する</div></td>
</tr>
<tr>
<td>Spanish</td>
<td>Current-position</td>
<td><strong>F3375</strong><div style="white-space: pre-wrap;">Artificial intelligence is important because it makes nuestra vida más fácil y nos ayuda en muchas cosas. A veces, los programas de inteligencia artificial se ven como una amenaza, pero también hay muchas cosas positivas que podrían llegar a ser.

Por ejemplo, las empresas pueden usar inteligencia artificial para</div></td>
<td><strong>C42</strong><div style="white-space: pre-wrap;">Artificial intelligence is important because it makes nuestra vida más fácil y nos ayuda en muchas cosas. A la vez, es capaz de hacer cosas que antes no se podía hacer. Por ejemplo, la inteligencia artificial puede ayudar a los científicos a entender un fenómeno que no se entiende.

</div></td>
</tr>
<tr>
<td>Spanish</td>
<td>All-positions</td>
<td><strong>F3375</strong><div style="white-space: pre-wrap;">Artificial intelligence is important because de-centralización de las empresas y la creación de aplicaciones para el consumo masivo.

&lt;h2&gt;¿Qué es el Inteligencia Artificial?&lt;/h2&gt;

La Inteligencia Artificial es la imitación de la capacidad de un ser humano para realizar tareas o procesos</div></td>
<td><strong>C42</strong><div style="white-space: pre-wrap;">Artificial intelligence is important because de-humanización de las tareas y la falta de empatía en la interacción con el usuario son problemas que la IA puede resolver.

La inteligencia artificial (IA) es la imitación de la capacidad de un sistema para pensar, aprender, razon</div></td>
</tr>
</tbody>
</table>

## Interpretation

The Layer-20 examples show that individual ICA components can causally shift
generation toward the same target languages studied with SAE steering. The
effect is not uniformly clean: some outputs remain code-mixed, malformed, or
repetitive. This pilot demonstrates steering capability; it does not provide
the original paper's full FastText language-success and LaBSE semantic-fidelity
evaluation.
