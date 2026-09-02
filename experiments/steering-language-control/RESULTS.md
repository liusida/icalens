# English-to-target language steering

## Setup

- **Reference:** [Causal Language Control in Multilingual Transformers via
  Sparse Feature Steering](https://arxiv.org/abs/2507.13410)
- **Language model:** `google/gemma-2-2b`
  (`c5ebcd40d208330abc697524c919956e692655cf`)
- **Transformer layer:** 20 (zero-based)
- **SAE:** Gemma Scope 2B residual width 16k,
  `layer_20/width_16k/average_l0_71/params.npz`
- **ICA Lens:** `sida/icalens-gemma-2-2b-pile10k`
- **Data:** 1,000 English/target-language sentence pairs per language from
  Tatoeba via [ManyThings](https://www.manythings.org/anki/)
- **Generation intervention:** primary results use `current-position`;
  `all-positions` is compared below for Chinese. Temperature 0.5, 50 new tokens

## Procedure

For each English/target-language pair, we take the residual-stream activation
at the last text token of each sentence. At layer 20, we encode it into either
SAE feature activations or signed ICA component scores. We then compute

$$
\Delta = \mathbb{E}[z_{\mathrm{target}}]
       - \mathbb{E}[z_{\mathrm{English}}].
$$

For each representation, we retain the three coordinates with the largest
$|\Delta|$ and preserve the sign of each difference. During generation, a
candidate's measured offset is added along its SAE decoder direction or ICA
writing direction at the runtime-current token position. We test the candidates
on four English prompts and report the candidate that produced the clearest
target-language shift.

The cosine similarity compares the two **effective steering vectors**, including
their measured signed offsets:

$$
\cos\!\left(\Delta_f d_f,\; \Delta_c a_c\right),
$$

where $d_f$ is the SAE decoder direction and $a_c$ is the ICA writing
direction. Including the offsets makes the comparison invariant to the
arbitrary sign of an ICA component axis.

## Successful candidates

| Target language | SAE feature | SAE offset | ICA component | ICA offset | Effective-vector cosine similarity |
|---|---:|---:|---:|---:|---:|
| Chinese | F6360 | +83.7958 | C105 | +21.4227 | 0.7896 |
| French | F12332 | +100.3192 | C25 | +23.0798 | 0.8956 |
| Japanese | F4945 | +91.6278 | C145 | -23.3301 | 0.6853 |
| Spanish | F8590 | +91.8189 | C42 | +17.2218 | 0.8537 |

## Concrete generations by convention

Every generation includes the complete fixed prompt, shown in gray, followed
by its generated continuation. SAE and ICA are shown side by side. Snippets
are line-wrapped, HTML tags are removed, and `…` marks truncation; the complete
outputs are in
[`multilingual-layer20`](runs/multilingual-layer20/) and
[`convention-comparison`](runs/convention-comparison/).

| Language | Convention | SAE steering | ICA steering |
|---|---|---|---|
| Chinese | `current-position` | **F6360:** <span style="color: #888;">Artificial intelligence is important because</span> it makes it possible for machines to learn and make decisions based on数据。 Artificial intelligence（AI）是一种技术，让计算机能够像人类一样学习和做出决策。AI可以应用于许多领域，例如医疗保健，交通运输和制造 | **C105:** <span style="color: #888;">Artificial intelligence is important because</span> it makes我们的生活更简单。它让我们的生活更轻松，更方便。它使我们能够做更多更有效的事情。它使我们能够更快地完成更多。它使我们能够更快地完成更多。它使我们能够更快 |
| Chinese | `all-positions` | **F6360:** <span style="color: #888;">Artificial intelligence is important because</span> it makes it possible for machines to学习，思考和做出决策。人工智能技术可以被应用于各种不同的领域，包括医疗保健，农业，能源和环境等。人工智能可以帮助人类解决一些棘手的问题，包括疾病的诊断和… | **C105:** <span style="color: #888;">Artificial intelligence is important because</span> 它可以提高效率，减少错误，并提高个人和企业对信息的可用性。它可以帮助我们解决许多问题，并使生活更轻松。人工智能可以帮助我们解决许多问题，例如提高效率，降低错误，并提高… |
| French | `current-position` | **F12332:** <span style="color: #888;">Artificial intelligence is important because</span> it makes it possible pour l'intelligence artificielle, l'intelligence artificielle peut être appliquée à des tâches qui ne peuvent être réalisées par les humains. Pour les personnes qui ont des handicaps, l'IA peut aider à améliorer la qualité de… | **C25:** <span style="color: #888;">Artificial intelligence is important because</span> it makes it possible pour l'intelligence artificielle (IA) peut-elle être utilisée pour la bonne ou la mauvaise ? L'intelligence artificielle est un terme qui désigne des programmes et des logiciels qui peuvent être utilisés pour effectuer… |
| French | `all-positions` | **F12332:** <span style="color: #888;">Artificial intelligence is important because</span> it makes it possible pour l'intelligence artificielle, l'intelligence artificielle peut être appliquée à des domaines divers et variés, tels que l'apprentissage automatique, l'intelligence des machines, le traitement de l'image et du son… | **C25:** <span style="color: #888;">Artificial intelligence is important because</span> it makes labour moins chère. L'intelligence artificielle est importante parce qu'elle rend les travailles moins chères. L'intelligence artificielle est une technologie… |
| Japanese | `current-position` | **F4945:** <span style="color: #888;">Artificial intelligence is important because</span> it makes it possible forコンピューター to do things that人間 cannot. It enablesコンピューター to learn fromデータ and to判断 what is正しい and what is誤り. It also enablesコンピューター to generate新しいデータ and to predict未来. | **C145:** <span style="color: #888;">Artificial intelligence is important because</span> it makes it possible for machines to learn from data and perform tasks that人はできない。AIは、機械がデータを学習し、人はできないことをできるようになるため、重要な役割を果たしています。AIは、人の考え方を再現し、予測… |
| Japanese | `all-positions` | **F4945:** <span style="color: #888;">Artificial intelligence is important because</span> it makes it possible forコンピューター to do things that人間 cannot. It enablesコンピューター to learn fromデータ and to判断 what is正しい and what is誤り. It also enablesコンピューター to do things that人間 cannot… | **C145:** <span style="color: #888;">Artificial intelligence is important because</span> it makes it possible to automate some of our身の周りに、さまざまな分野でAIが活用されている。AIは、人間の意思を介さずに、自動的に意思決定ができるという特技を持っている。AIは、人間の意思を介さずに… |
| Spanish | `current-position` | **F8590:** <span style="color: #888;">Artificial intelligence is important because</span> it makes nuestra vida más fácil y nos ayuda en muchas cosas. A continuación, explicaremos por qué es importante y cómo lo podemos usar para mejorar nuestro día a día. ¿Qué es la inteligencia artificial? La inteligencia artificial… | **C42:** <span style="color: #888;">Artificial intelligence is important because</span> it makes nuestra vida más fácil y nos ayuda en muchas cosas. A la vez, es capaz de hacer cosas que antes no se podía hacer. Por ejemplo, la inteligencia artificial puede ayudar a los científicos a entender un fenómeno que no se entiende. |
| Spanish | `all-positions` | **F8590:** <span style="color: #888;">Artificial intelligence is important because</span> it makes nuestra vida más fácil y nos ayuda en muchas cosas. A veces, los programas de inteligencia artificial se ven como algo mágico, pero el mundo real es mucho más sencillo de lo que parece. En este artículo, aprenderá todo sobre inteligencia… | **C42:** <span style="color: #888;">Artificial intelligence is important because</span> de-humanización de las tareas y la falta de interacción con las personas puede ser dañina para la salud mental. La inteligencia artificial es un campo de la tecnología que se centra en la creación de sistemas que puedan pensar y actuar… |

`all-positions` generally causes an earlier target-language transition. The
clearest difference is Chinese ICA, which changes from an English opening under
`current-position` to immediate Chinese under `all-positions`. It is not
uniformly better: across the other three prompts, stronger intervention also
produced more repetition, code-mixing, or loss of relevance in some cases.

All four language conditions produced target-language continuations for both
SAE and ICA steering. Chinese and French were the clearest. Japanese was more
frequently code-mixed, and some Spanish generations were repetitive or
code-mixed. For Spanish, the successful ICA component C42 was the
second-largest contrast candidate; the largest candidate, C581, did not steer
the tested prompts into Spanish.

These are qualitative four-prompt checks at one layer. They establish that the
selected directions are causally effective, but they are not the paper's full
all-layer FastText and LaBSE evaluation.
