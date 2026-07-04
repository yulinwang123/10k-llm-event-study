# Prompt v1: Earnings Call Semantic Narrative Measures

Use this prompt for transcript-level or segment-level LLM inference. Replace `{segment_type}` and `{text}` at runtime.

```text
You are extracting structured semantic features from an earnings call transcript segment for an accounting and finance research project.

Research goal:
We study how investors weight hard earnings numbers versus managerial narrative during earnings announcements. Your task is only to measure the managerial narrative in the transcript text. Do not predict stock returns, market reactions, investor reactions, analyst reactions, or future realized outcomes.

Domain context:
This is an earnings call transcript. "Performance" refers to firm operating and financial performance, including revenue, margins, EPS, demand, costs, cash flow, guidance, segment performance, pricing, volume, supply chain, labor, foreign exchange, interest rates, regulation, competition, and macroeconomic conditions.

"Hard earnings news" refers to quantitative earnings surprise. It is not included in this prompt. Measure only the narrative content in the text below.

"Managerial narrative" refers to how management explains performance and outlook, including tone, uncertainty, specificity, defensiveness, informativeness, and attribution.

Use only the transcript text below. Do not use outside knowledge. Do not infer whether the stock price should go up or down.

Segment type: {segment_type}

Transcript text:
{text}

Scoring rule:
Most dimensions use a continuous score from 0 to 1 because they measure latent intensity rather than discrete categories. Use 0 when the feature is absent, 0.5 when it is moderate, and 1 when it is dominant. Intermediate values such as 0.35 or 0.68 are allowed when the evidence falls between anchors.

General anchors for all [0, 1] dimensions:
- 0.00 = absent or essentially not present
- 0.25 = weak or occasional presence
- 0.50 = moderate presence
- 0.75 = strong and repeated presence
- 1.00 = dominant feature of the segment

Do not default to 0.50 unless the evidence is genuinely mixed or moderate. Use the full range when the text clearly supports it.

Score the following dimensions:

1. contextual_sentiment [-1, 1]

Definition:
The overall semantic tone of management's discussion of current performance and future outlook. This should capture meaning in context, not simple positive or negative word counts.

Scale:
- -1.00 = strongly negative: major deterioration, losses, demand collapse, severe margin pressure, liquidity stress, or clearly pessimistic outlook
- -0.50 = moderately negative: noticeable weakness, pressure, underperformance, caution, or negative outlook, but not severe
-  0.00 = neutral or mixed: balanced discussion, factual reporting, or positive and negative information offset each other
- +0.50 = moderately positive: solid performance, improvement, constructive outlook, but with some caveats
- +1.00 = strongly positive: broad-based strength, accelerating demand, expanding margins, confident outlook, or unusually positive substantive framing

High-score example:
"Demand accelerated across all regions, margins expanded, and we expect continued growth next quarter."

Low-score example:
"Revenue declined sharply, margins compressed, and demand visibility remains poor."

Important:
Do not assign high positive sentiment merely because the language sounds polished or promotional. Focus on substantive meaning.

2. uncertainty [0, 1]

Definition:
The extent to which the segment discusses uncertainty in business conditions, demand, costs, macro environment, guidance, visibility, regulation, supply chain, or future performance.

Scale:
- 0.00 = no meaningful uncertainty; conditions are described as clear or stable
- 0.25 = minor uncertainty mentioned briefly
- 0.50 = moderate uncertainty affecting some business areas or outlook
- 0.75 = repeated uncertainty across multiple drivers, markets, or time periods
- 1.00 = uncertainty is central to the segment; management repeatedly emphasizes limited visibility, volatility, or unpredictable conditions

High-score example:
"Customer demand remains difficult to forecast, and we have limited visibility into second-half orders."

Low-score example:
"Order trends have been stable, and we have good visibility into the next quarter."

Not necessarily high uncertainty:
"We operate in a competitive environment." This is generic risk language unless tied to current uncertainty.

Distinguish from hedging:
Uncertainty is about the business environment or information state. Hedging is about cautious or non-committal language.

3. hedging [0, 1]

Definition:
The extent to which management uses cautious, qualified, vague, or non-committal language, regardless of whether the underlying news is good or bad.

Examples of hedging language:
"may", "could", "approximately", "we believe", "we expect but cannot assure", "subject to", "depending on", "too early to tell", "we are cautiously optimistic".

Scale:
- 0.00 = direct, specific, and committed language
- 0.25 = occasional qualifications but mostly direct
- 0.50 = several qualifications or cautious statements
- 0.75 = frequent hedging; management avoids firm commitments
- 1.00 = highly hedged or vague throughout; little concrete commitment

High-score example:
"We may see some improvement, depending on customer behavior, but it is too early to quantify."

Low-score example:
"We expect revenue to grow 6% to 8% next quarter, driven by signed contracts already in backlog."

Distinguish from uncertainty:
A segment can describe uncertainty directly without hedging. A segment can also hedge even when discussing positive results.

4. forward_looking_specificity [0, 1]

Definition:
The extent to which the segment provides concrete, forward-looking information about future performance, guidance, strategic plans, expected drivers, timing, or operational actions.

Scale:
- 0.00 = no forward-looking content, or only generic phrases like "we are optimistic about the future"
- 0.25 = broad future-oriented statements with little detail
- 0.50 = some future drivers or plans are discussed, but without much precision
- 0.75 = specific future drivers, operational plans, time frames, or directional guidance
- 1.00 = detailed and concrete forward-looking information, such as numeric guidance, margin targets, revenue drivers, cost plans, timelines, segment-specific expectations, or clear strategic milestones

High-score example:
"We expect second-quarter revenue to grow in the mid-single digits, led by enterprise demand and a 150 basis point improvement in gross margin from lower freight costs."

Low-score example:
"We remain excited about our long-term future."

Important:
Do not score generic optimism as high forward-looking specificity. The key is specificity, not positivity.

5. informativeness [0, 1]

Definition:
The extent to which the segment explains the economic drivers behind current earnings performance or future outlook. High informativeness means the narrative helps explain why earnings changed or what mechanisms drive future performance.

Relevant drivers include:
revenue growth, pricing, volume, demand, costs, margins, product mix, customer behavior, segment performance, supply chain, labor, foreign exchange, interest rates, regulation, competition, investment, restructuring, or macro conditions.

Scale:
- 0.00 = boilerplate, slogans, repetition of headline numbers, or no explanation of drivers
- 0.25 = minimal explanation; mostly generic statements
- 0.50 = some drivers identified but limited depth
- 0.75 = clear explanation of several key drivers
- 1.00 = highly informative; gives detailed, coherent, and economically meaningful explanation of performance or outlook

High-score example:
"Gross margin declined by 180 basis points because freight costs increased, product mix shifted toward lower-margin hardware, and pricing actions lagged input-cost inflation."

Low-score example:
"We faced some challenges but remain confident in our strategy."

Important:
A segment can be negative and highly informative. Informativeness is not the same as positive sentiment.

6. defensiveness [0, 1]

Definition:
The extent to which management appears to defend, justify, minimize, reframe, or shift attention away from negative performance, weak guidance, analyst concerns, or operational problems.

Signals of defensiveness:
- blaming external factors without much internal accountability
- emphasizing temporary factors to downplay weakness
- repeatedly saying issues are transitory
- redirecting from weak metrics to unrelated positives
- justifying missed expectations
- resisting or softening analyst criticism

Scale:
- 0.00 = no defensive framing; management is direct and balanced
- 0.25 = mild justification or brief downplaying
- 0.50 = moderate defensive framing around some issues
- 0.75 = frequent attempts to justify, reframe, or minimize negative information
- 1.00 = defensiveness dominates the segment; management persistently avoids accountability or shifts attention

High-score example:
"The miss was entirely due to temporary macro headwinds, and we do not believe it reflects any execution issues."

Low-score example:
"Margins declined because we mispriced several contracts, and we are taking corrective action."

Important:
Defensiveness is not the same as negative sentiment. A negative but candid explanation can have low defensiveness.

7. qa_evasiveness [0, 1 or null]

Definition:
For Q&A segments only, the extent to which management avoids directly answering analyst questions.

Signals of evasiveness:
- does not answer the specific question asked
- gives generic talking points instead of details
- redirects to another topic
- refuses to quantify when quantification is reasonable
- gives overly long but low-information answers
- says "we do not disclose that" repeatedly without alternative explanation
- answers only the easiest part of a multi-part question

Scale:
- 0.00 = direct, specific, and responsive answers
- 0.25 = mostly responsive with minor omissions
- 0.50 = partially responsive; some avoidance or vagueness
- 0.75 = often evasive or redirects important questions
- 1.00 = highly evasive; repeatedly avoids direct answers

High-score example:
Analyst asks about margin guidance, and management gives a generic answer about long-term strategy without addressing margins.

Low-score example:
Analyst asks about margin guidance, and management explains expected margin drivers, timing, and uncertainty.

For prepared remarks:
Return null.

Important:
A refusal to disclose is not always evasive if management gives a clear and reasonable explanation. Score high only when the answer avoids substantive engagement.

8. self_serving_attribution [0, 1]

Definition:
The extent to which management asymmetrically attributes good outcomes to internal skill, strategy, execution, or discipline, while attributing bad outcomes to external, temporary, or uncontrollable factors.

Signals:
- good performance credited to "our strategy", "execution", "discipline", "team", "innovation"
- bad performance blamed on macro, foreign exchange, weather, supply chain, regulation, timing, customers, or temporary headwinds
- selective emphasis of internal causes for success and external causes for weakness

Scale:
- 0.00 = no attribution pattern or balanced attribution
- 0.25 = mild self-serving attribution
- 0.50 = noticeable asymmetric attribution
- 0.75 = repeated self-serving attribution across topics
- 1.00 = strong and systematic self-serving attribution dominates the narrative

High-score example:
"Our strong execution drove revenue growth, while the margin shortfall was entirely due to temporary external cost pressures."

Low-score example:
"Revenue growth benefited from both strong execution and favorable demand, while margin pressure reflected both our pricing decisions and higher input costs."

Important:
Do not score high simply because management mentions external factors. Score high when there is an asymmetric pattern: internal credit for good news, external blame for bad news.

Return valid JSON only. Use numeric values, not strings. Do not include comments outside JSON. The short_rationale should be at most 40 words and should cite narrative features only, not stock-market implications.

JSON schema:
{
  "contextual_sentiment": number,
  "uncertainty": number,
  "hedging": number,
  "forward_looking_specificity": number,
  "informativeness": number,
  "defensiveness": number,
  "qa_evasiveness": number or null,
  "self_serving_attribution": number,
  "short_rationale": string
}
```

