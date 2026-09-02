# English-to-target language steering

Minimal SAE/ICA language-steering comparison based on
[Causal Language Control in Multilingual Transformers via Sparse Feature Steering](https://arxiv.org/abs/2507.13410).

- **Model:** `google/gemma-2-2b`
- **Layer:** 20
- **Calibration pairs:** 1000 per language
- **Final-token mode:** `text`
- **Steering:** `all-positions`
- **Evaluator:** `gpt-4.1-mini-2025-04-14`

## Selection

Evaluate each of the three largest activation contrasts on four prompts. Select
the candidate with the most passing outputs, then display its best passing sample.

| Language | SAE feature | SAE offset | ICA component | ICA offset | Signed cosine |
|---|---:|---:|---:|---:|---:|
| Chinese | F13458 | +73.6488 | C105 | +21.4227 | 0.7765 |
| French | F13692 | +107.4795 | C25 | +23.0798 | 0.9225 |
| Japanese | F3953 | +90.2733 | C145 | -23.3301 | 0.5970 |
| Spanish | F3375 | +97.2529 | C42 | +17.2218 | 0.8920 |

## Generations

<table style="width: 100%; table-layout: fixed;">
<colgroup>
<col style="width: 10%;">
<col style="width: 45%;">
<col style="width: 45%;">
</colgroup>
<thead><tr><th>Language</th><th>SAE</th><th>ICA</th></tr></thead>
<tbody>
<tr><td style="vertical-align: top;">Chinese</td><td style="vertical-align: top;"><strong>F13458</strong><br><span style="color: #777;">The difference between weather and climate is</span> that weather is the统计分析与预测过程，而气候是统计分析与预测过程。气候是天气持续的时间。天气是短期的，而气候是长期的。天气是随机的，而气候是规律性的。<br><br>天气是指</td><td style="vertical-align: top;"><strong>C105</strong><br><span style="color: #777;">Artificial intelligence is important because</span>它可以提高效率，减少错误，并提高个人和企业对信息的可用性。它可以帮助我们解决许多问题，并使生活更轻松。<br><br>人工智能可以帮助我们解决许多问题，例如提高效率，降低错误，并提高</td></tr>
<tr><td style="vertical-align: top;">French</td><td style="vertical-align: top;"><strong>F13692</strong><br><span style="color: #777;">Leaves change color in autumn because</span> les sont pigmentes, qui sont des molécules qui absorbent les rayons du soleil. Les végétaux produisent de la chlorophylle, qui est un pigment qui permet la photosynthèse.<br><br>Les feuilles changent de couleur en automne parce</td><td style="vertical-align: top;"><strong>C25</strong><br><span style="color: #777;">Artificial intelligence is important because</span> it makes labo&lt;strong&gt;ur&lt;/strong&gt; &lt;strong&gt;moins&lt;/strong&gt; &lt;strong&gt;chère&lt;/strong&gt;.<br><br>L&#x27;intelligence artificielle est importante parce qu&#x27;elle rend les &lt;strong&gt;travailles moins chères&lt;/strong&gt;.<br><br>L&#x27;intelligence artificielle est une technologie</td></tr>
<tr><td style="vertical-align: top;">Japanese</td><td style="vertical-align: top;"><strong>F3953</strong><br><span style="color: #777;">The difference between weather and climate is</span> that weather is the短時間（短期間）の気象状況（気候）の事。一方、気候は季節（季節性）や広域（広域性）で、その長期的な（長期的な）気象</td><td style="vertical-align: top;"><strong>C145</strong><br><span style="color: #777;">Artificial intelligence is important because</span> it makes it possible for machines to learn from data and perform tasks that人はできない。<br><br>AIは、機械がデータを学習し、人はできないことをできるようになるため、重要な役割を果たしています。<br><br>AIは、人の考えや行動を再現する</td></tr>
<tr><td style="vertical-align: top;">Spanish</td><td style="vertical-align: top;"><strong>F3375</strong><br><span style="color: #777;">A practical way to organize a small kitchen is</span> a cocina americana, o sea, una cocina americana.<br><br>La cocina americana es un espacio que se encuentra al lado del comedor, y de ahí su nombre.<br><br>Es una zona muy popular en los países de habla hispánica, ya que es</td><td style="vertical-align: top;"><strong>C42</strong><br><span style="color: #777;">Artificial intelligence is important because</span> de-humanización de las tareas y la falta de empatía en la interacción con el usuario son problemas que la IA puede resolver.<br><br>La inteligencia artificial (IA) es la imitación de la capacidad de un sistema para pensar, aprender, razon</td></tr>
</tbody>
</table>
