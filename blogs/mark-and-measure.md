# Mark and Measure

Reference notes for a future engineering blog. This is not the published essay. It records the training-data method built in this project, why it was chosen, the tricks that are easy to skip, and what must not be overclaimed.

Working name: **Mark and Measure**.

Family it belongs to: synthetic data generation, the name NVIDIA uses (SDG). The specific mechanism, a writer model brackets only the evidence and a program measures the labels, does not have a single name at Google or NVIDIA.

## The sentence

A writer model invents a messy human document and puts brackets around the evidence. A program deletes the brackets, counts where the words landed, calculates anything that was not written, and throws the document away if the calculation was already printed. The student model later sees only the clean document.

## Why this exists

The target is a Clairic-style extraction model on Gemma 4 E2B: one forward pass, no generated answer text, four small heads on a frozen reader.

[Clairic](https://blog.cloudflare.com/clairic-extraction-model) (Cloudflare, 19 October 2026) reads a document once and returns typed values: a span of the original text, a number, a real calendar value, a duration, a list, an estimate with an 80% band, or a short forecast. It does not write a JSON reply. The weights and the head code are not public. The output shapes are.

Gemma 4 E2B is the student: about 2.3 billion effective parameters, 5.1 billion once the per-layer embedding tables are counted, 128k context, text and image. Those embedding tables stay frozen. The chat model is not asked to emit the answer.

Paths that were considered and set aside:

- Asking Gemma for JSON. Easy, and still a generator. It can invent an order id. Cloudflare's own note said a plain Gemma 4 run failed when it had to invent a repair cost.
- A grammar that only forces valid JSON. The shape is legal. The content can still be wrong.
- A LoRA that memorizes one fixed schema. Useful for one document type. A new field needs a new tune.
- One shared pointer head for every field. Enough for text that is copied. A pointer cannot create `2413.25` from three payments, cannot force a real calendar day, and cannot return a band.
- Filling one template with random facts, including with Gemma 31B as the writer. The labels are easy, and the language repeats. The user does not have a 31B model. Thirty thousand copies of one shape were rejected for the same reason.
- Letting the writer report character indexes or totals from memory. Language models write the story well and count badly.

Mark and Measure is the path that remained. The writer is Grok 4.7, the normal one, not a fast variant and not the student. The checker is `extractor/audit.py`.

## What to call it, and what other labs call

Use **Mark and Measure** for this method. Say it in one line: the model marks the evidence, the program measures the answer.

Do not claim the whole idea is new. These are the relatives, and the difference matters in the blog:

- **Synthetic data generation (SDG).** NVIDIA's name for LLM-written training sets. Nemotron-4 340B used synthetic text for about 98% of its alignment data. A second model, Nemotron-4 340B-Reward, ranks replies and the low-scoring ones are dropped. NeMo Data Designer is their later tool: seed columns, a generator model, then Python or LLM-judge validators. That is "generate, then keep the good rows." Their validators check schema and quality. They do not rebuild character spans or own the arithmetic.
- **Self-Instruct.** An earlier research pattern: a language model writes instruction examples for a smaller model. The writer and the student are different roles. There is no span measurement.
- **Data programming (Snorkel).** Code rules, called labeling functions, vote on a label. The program is the labeler. The text usually already exists. In Mark and Measure the text is invented, and the program is not voting. It either measures an exact label or drops the row.
- **Template span synthesis.** Tools such as synthspan fill `{CITY}` slots in a fixed sentence and record the character span because the program inserted the word. Labels are perfect. The sentences share a skeleton. That is the deterministic generator this project refused as the main method.
- **LLM picks the string, code finds the offset.** Small NER tools do this: the model names the entity, a search function returns the start and end. Mark and Measure is that idea, plus calculated answers the search cannot find, plus a rule that a leaked calculation kills the row.

The blog can honestly say: the family is verified synthetic data. The piece that is particular here is evidence brackets plus a measuring program that owns every derived label for an extraction head.

## The three roles

Keep these separate in the writing. Collapsing them is how the method gets misunderstood.

1. **Writer.** Grok 4.7 invents a situation a person might actually send: a blouse job in Vadodara, a cello complaint in Belo Horizonte. It wraps evidence in `[[name]]words[[/]]`. It does not report character numbers. It does not write the sum, the day count, the due date, the median, or the forecast that the student is supposed to learn.
2. **Auditor.** `extractor/audit.py` strips the brackets, checks the slice, parses dates and amounts, computes the derived answers, and drops the whole document on failure. It has no taste and no memory of the story.
3. **Student.** Gemma 4 E2B, frozen, with small heads. It sees the clean text and the questions. It never sees a bracket. Round 1 has been trained. The numbers are in the section below. They are not a production score.

The fascination people feel is the handoff. The writer knows which words are the evidence because it chose them. The program knows where they sit because it counted after the brackets were removed. Neither is asked to do the other's job.

## The four heads

One Gemma read produces the token states. The field type picks the head. Field names share a head. "Job code" and "ceremony date" are not two models.

| Head | Learns | Right answer | Derived by |
| --- | --- | --- | --- |
| Point | Start and end of a real slice, a list of slices, or missing | `HT-3391` at characters 134–141, or empty | The bracketed quote, measured after the brackets are gone |
| Number | The amount itself, including one that is not printed | `2413.25`, or `15` days | Parsed from a quote, or sum, product, subtract, or days-between |
| Calendar | A legal year, month, day, and time | `2026-04-19`, or a due date the text does not print | Parsed from a quote, or issue date plus N days |
| Spread | A chance across bins the user set | The middle of an even list of prices, or the hidden tail of a series | Median of comps, or the forecast lines removed from the student text |

The point head is shared because the question text already says what to find. A separate head per field name would repeat the same pointing skill. The other heads exist because a pointer cannot mint a number, a legal date, or a band.

Every scalar head also has a missing case. The gold label is `missing: true` and `value: null`. That is a hard target, not a calibrated probability. Clairic's `p_null` and `confidence` still need a later calibration stage. This dataset does not teach honest probabilities yet.

## The pipeline, on one document

Raw writer output, `data/raw/support.jsonl`, id `support-01`. Brackets are the contract.

```text
The thread ceremony is [[ceremony]]April 19, 2026[[/]]
and blouse [[order_id]]HT-3391[[/]] is still at your table.
You cut it on [[cut_on]]2026-03-18[[/]].
I am writing on [[written_on]]2 April 2026[[/]] ...
GPay sent [[part]]1736.40[[/]], I left cash of [[part]]582.15[[/]],
and the button loops were another [[part]]94.70[[/]].
Still to do: [[alter]]sleeve shortened[[/]], [[alter]]neckline raised[[/]],
[[alter]]hook moved[[/]].
A measurement chart number was never read back to me.
Old job HT-908 was Amma's saree fall, leave it.
```

The `ask` list tells the auditor which head each field trains, and which marks feed it. The writer does not fill in `start`, `end`, or `2413.25`.

The auditor then:

1. Finds each `[[name]] ... [[/]]`. A broken or nested bracket drops the document.
2. Deletes the brackets and stitches the prose back together. Offsets are Unicode code points in that clean string, the same indexes Python uses. They are not token indexes and not UTF-16.
3. Checks `clean[start:end] == inner`. If the slice drifted, the row is dropped.
4. Parses dates with an explicit calendar, including `Sept`. Slash dates are ambiguous, so the field carries `dayfirst`. `02/04/2026` with `dayfirst: true` is 2 April, not 4 February.
5. Parses amounts as `Decimal`, after stripping commas and a leading currency sign. Rounding is half up to the field's decimals.
6. Computes derived fields from the marks only.
7. Rejects the document if a computed answer already appears as its own number in the clean text.
8. Writes the clean text and the measured fields to `data/audited/`.

Saved labels for Hetal:

- Point, job code: `HT-3391`, characters 134–141. `HT-908` is in the text and is not the label.
- Point, list: `sleeve shortened`, `neckline raised`, `hook moved`, in that order.
- Point, missing: chart number is null. The phone `0265 231 7844` is a decoy.
- Number, sum: `1736.40 + 582.15 + 94.70 = 2413.25`. The total is not in the message.
- Number, duration: 18 March 2026 to 2 April 2026 is 15 days. The message does not say 15.
- Calendar: `April 19, 2026` becomes `2026-04-19`.

`HT-908` is the clever negative. It looks like a job code. It has no brackets, so the program never promotes it to a label. At training time, a point head that returns `HT-908` is wrong.

## Correct, wrong, and dropped

Three different outcomes. The blog should not blur them.

- **Correct label.** The brackets survived, the slice matches, the date parses, and any calculated value is absent from the prose. The row is gold for that head.
- **Wrong student answer, later.** The row is kept. During training, the head guesses another span or another total. The loss is against the auditor's label. The decoy is what makes that guess tempting.
- **Dropped document.** The auditor refuses the whole row, not one bad field. A partial row would teach a lie next to a truth. Current drop reasons include broken brackets, a missing mark, an unparsable date or amount, a negative duration, a leaked answer, an odd count of comps, a forecast tail that is still visible, a banned string, and a duplicate id.

The leak check is deliberately fussy. `number_present` looks for the rendered number on token boundaries. `12` counts as present in `old ref 12 only` and does not count as present inside `120`. Otherwise a room number would kill a true total of 12, or a leaked `15` hiding inside `150` would slip through.

## What is random, and what is the shift

The situation is chosen by the writer, not rolled by a fact sheet. The support file alone has twelve different lives: a blouse, a goat vet, a dental crown, a damp machiya, a cello. The other files add hospital bills, a Colombo repair bench, Kothrud rents, and a Howrah tea stall. That is the randomization. There is no seed that swaps names inside one skeleton.

The label is not random. After the words exist, the program either measures them or drops them.

The **shift** is the thing the student has to survive: the same kind of answer sitting in a new sentence.

- In Hetal's WhatsApp, the job code starts at character 134 and the money is a sum of three payments.
- In `support-10`, Helena in Belo Horizonte writes in Portuguese. The cello contract `BH-5529` sits in a different place. The money is a subtraction, `730.60 - 185.40 = 545.20`. The loan is 72 days, from 20 January 2026 to 2 April 2026. An unmarked decoy, `BH-17` and `318` reais for the case, is in the letter and is not the label.

Train, val, and test are whole documents. Fields from one story are never split across the sets, because the model would then be tested on a paragraph it already read. After sorting ids, index mod 10 chooses the bucket: 8 is val, 9 is test, everything else is train. The current test documents are `bills-10`, `ops-08`, `series-06`, and `support-04`. Val is `bills-09`, `ops-07`, `series-05`, and `support-03`.

The shift inside the support file is real and still narrow: a head can memorize "one code, one list, one money calculation, one day count, one missing id." The bills, ops, and series files are the wider shift. A hospital sheet with a computed due date, a Colombo repair card with hours times a rate, and a Pune rent comparison are not Hetal's WhatsApp. Passing those is a better exam. It is still one writer's batch per file, not a real inbox.

## Calculated answers the search cannot see

These are specified in `ask` and implemented in the auditor. The writer marks only the inputs.

- `sum` of every mark with one shared name, at least two parts.
- `product` of a left mark and a right mark, such as hours times a rate.
- `subtract` of left minus right, such as a deposit minus a fee.
- `days_between` two date marks. The day count must not be printed.
- `add_days` on a calendar field. "Credit runs 14 days from 2 Apr 2026" becomes `2026-04-16`, and that due date is not printed. This is the Clairic "net 30" case. `bills-01` is one kept example: Bhosale Memorial Hospital, admitted `2026-04-02T14:30`, pay-by computed as 16 April 2026.
- `estimate`: an even number of `comp` marks, at least four. Gold is the average of the two middle values after sorting, quantized. An even count is deliberate. With three prices the median is one of the prices, and the head can cheat by copying the middle line. The median must sit inside `min` and `max` and must not be printed.
- `forecast`: each observation is `[[obs]]YYYY-MM-DD|value[[/]]` on its own line. The auditor deletes the last `horizon` lines from the student text and saves those values as the label. The history stays. The hidden numbers must not appear in the history or the intro. The answer is absent on purpose. Code does not "find" it in the student text. It held it back.

Estimate gold is the median of the listed comps, not an appraiser's judgment. Forecast gold is the withheld tail of a series the writer composed, not a future that happened in the world. Say that plainly in the blog. The labels are exact inside the rules. They are not a hidden real-world truth.

## Clever details worth keeping in the essay

- The brackets are not the label. They are a promise that those exact characters are the evidence. The label is the measurement after deletion.
- Decoys are ordinary words with no brackets. The program has no list of traps to memorize. Absence of a mark is what makes them traps.
- A missing field is an `ask` entry with `mark: null`, often with `must_not_contain` or a document-level `banned` list, so a null label cannot hide a code that was actually written.
- One bad field drops the whole document. Mixed gold and garbage in one row is worse than a shorter dataset.
- Dates are parsed by code, not believed as ISO strings the writer typed beside the prose. `April 19, 2026` and `2026-04-19` can be the same day. `Sept` is accepted. Impossible days fail.
- Money uses `Decimal`, not binary floats, then quantizes. The saved JSON still stores a JSON number, which is a float. For cents that is fine after quantization. The blog should not pretend the file is a decimal type on disk.
- The leak check ignores a number that is only a prefix of a longer number. That one regex is the difference between a usable sum label and a pile of false drops.
- Forecast removal is line-based. An observation written inline cannot be deleted cleanly, so the auditor drops `obs_not_on_own_line` instead of guessing.
- The student will need a second mapping, from these character offsets to Gemma tokens. The dataset does not store token indexes, because those depend on the tokenizer. Build that map at train time and check that the decoded tokens still equal the slice.
- The writer and the student are different models on purpose. A student who writes its own textbook copies its own mistakes. Grok 4.7 writes. Gemma 4 E2B, later, only reads.
- The first twelve rows were written by a Grok 4.7 sub-agent, one situation at a time, not by a template expander. Later writers are the same model, the non-fast one.

## How training will use a row

Not started. The intended loop, when a row is honest:

1. Freeze Gemma 4 E2B, including the large per-layer embedding tables. Run one forward pass on the clean text plus the questions. Turn thinking off.
2. Point head: loss on the start token and the end token of each gold slice, in order for lists, and a missing score when the gold is null. A predicted slice of `HT-908` is a miss.
3. Number head: loss on the digits of `2413.25` and of `15`, one shot, not a written sentence. Evidence spans can be an auxiliary target. They are not the answer.
4. Calendar head: loss on year, month, and day as legal choices. Month 13 is not in the output space.
5. Spread head: loss on the bin of the median, or on each withheld forecast step. Six estimates and six forecasts are in the audited set.
6. Score the heads on the four test documents only after training: `bills-10`, `ops-08`, `series-06`, and `support-04`. Those documents are not in the training file.

Confidence comes later. A sharp softmax is not a probability until it is checked on documents the generator never saw, including real ones. Clairic says the same thing in their preview notes: intervals can run narrow on unfamiliar data.

## Quality of the data on disk today

Four writers, four files, all audited together. `data/audited/report.json`:

- 48 kept, 0 dropped.
- 40 train, 4 val, 4 test.
- Fields: 65 text spans, 31 missing text fields, 34 lists, 53 numbers, 44 dates, 4 times, 5 datetimes, 21 durations, 6 estimates, 6 forecasts.

Every head now has gold. The spread head has six rent-or-price comparisons and six withheld series. The number head has printed amounts, sums, products, and subtractions. The calendar head has quoted dates, times, datetimes, and due dates computed with `add_days`.

Checked examples worth quoting:

- `series-01`: six Kothrud rents, gold median `25375`, and that number is not in the prose.
- `series-07`: cup counts beside Howrah. The student text keeps 40, 42, 38, 55. The label is the hidden tail, 61 then 63.
- `ops-02`: 6 hours at 850 rupees. Gold labour is `5100`, a product. Quote `Q-77` said 2200 and is a decoy.
- `bills-01`: ancillary charges sum to `4331.85`. Pay-by is 16 April 2026, computed, not printed. The grand total `9149.20` is printed, so it is a copy label, not a sum label.

The labels match the text because the auditor built them. The prose is specific: Pune, Colombo, Howrah, Belo Horizonte.

The set is still a pilot. Forty training documents can show a head move. They cannot settle it. Within the support file the field shape repeats. The four files came from four separate writer passes, so the variety is real and still one voice per file. Copying any of them with new numbers would inflate the count and teach the skeleton.

Real customer mail is still required before anyone calls the heads production-ready. This pipeline cannot invent that mail. A few hundred real documents, run through the same measurement rules where the evidence can be marked, are the shift that matters.

One operational miss belongs in the engineering notes, not in the method. Each writer committed its file on its own git branch. A checkout of the support branch showed only `support.jsonl`, which looked like the other files had never been written. They had. The auditor has to be run on the assembled `data/raw/*.jsonl`, not on whichever branch the workspace happens to be sitting on.

## Round 1, what actually ran

The audited 48 stayed frozen. A separate 24-document exam was written by Grok 4.7 (the normal one, not a fast variant), audited 24 kept and 0 dropped, and stored at `data/generalization/`. It was not used to train and not used to pick the epoch.

No Kaggle token was on this machine. `~/.kaggle` was missing, and the environment had no Kaggle variable. The same public model, `google/gemma-4-E2B-it`, was read from the Hugging Face snapshot already on disk. The checkpoint file is 10.2GB. This machine has no GPU and about 6GB free RAM, so the full file cannot be loaded.

The forward that did run:

- Tokenize every document first. The 72 documents used 2,721 distinct token ids, not the 262,144-row tables.
- Copy only those rows out of `embed_tokens` and `embed_tokens_per_layer`. The per-layer table is the expensive one: each row is 35 × 256. Copying 2,721 rows is about 50MB. Copying the whole table would be about 4.7GB.
- Copy the text-tower layer weights, about 2.5GB in bf16, into a `Gemma4TextModel` whose vocabulary is those 2,721 rows. The raw rows are copied. The embedding module still multiplies by `sqrt(hidden)` in its forward. Do not bake that scale into the copied rows.
- The document is the causal prefix and every question is packed after it. One forward per document is enough, because a later question cannot change the document's hidden states. The head reads the last token of the question, the token that has seen the document and the whole question.
- A forward of a few hundred tokens took under a second on four CPU cores. Cached states are in `outputs/cache/docs/` and are not committed.

Two measurement facts that are easy to miss:

- Gemma's tokenizer often glues the preceding space onto the first piece of an answer. On the audited point spans, 135 of 160 gold character spans begin in the middle of a token. The covering token text, with whitespace stripped, still equals the gold string in every one of those 160. The head is trained on the covering token indexes. Scores report token exact, stripped-text exact, and raw character-offset exact separately. Character-offset exact stays low. That is the tokenizer, not a miss.
- Lists become "Item 1", "Item 2", and one extra item whose correct answer is missing. Forecast steps become "Step 1", "Step 2". A field is correct only when every one of those questions is correct. A document is correct only when every field is correct.

The point head's first loss was about 88 per question. Gemma's states have standard deviation around 4, and an unnormalized dot product over 64 dimensions then has logits hundreds wide. Cross-entropy against the wrong token becomes huge, and the step barely moves. The fix, chosen before the scored run, is a LayerNorm on the states plus the usual `1/sqrt(inner)` scale on the start and end scores. After that, the initial point loss was about 8, and training field exact-match climbed past 90%.

The spread loss is a Gaussian negative log likelihood. It can keep falling forever if the width shrinks, even when the snapped number is already wrong. The saved epoch is the earliest epoch with the highest training field exact-match, not the lowest loss. Validation was not used to pick the epoch. Four documents are too few: one field is 4% of the score. Validation only picked the missing threshold from 0.5, 0.3, and 0.7. The chosen threshold was 0.5.

Round 1 scores, field exact-match, after that frozen choice. The "before" column is the same heads at random initialization.

| Split | Fields | Before | After | Documents after |
| --- | --- | --- | --- | --- |
| Train, 40 docs | 221 | 6/221 | 213/221 | 32/40 |
| Val, 4 docs | 24 | 0/24 | 4/24 | 0/4 |
| Test, 4 docs, scored once | 24 | 0/24 | 3/24 | 0/4 |
| Generalization exam, 24 docs, scored once | 112 | 3/112 | 29/112 | 0/24 |

By head on the exam: point 17/59, number 2/19, calendar 10/28, spread 0/6. On the training documents the point, number, and calendar fields were 100%. The spread fields were 2/10. The heads memorized the forty documents. They did not carry the skill to new cities. The same-source test, 3/24, says the same thing. Do not quote the 213/221 training number as accuracy.

The test split has no forecast, no time, and no datetime. Only the exam can speak for those. The original 48 are still the only same-source set. Real customer mail is still not on disk.

Because point, number, and calendar are clearly below half on an exam with enough fields, and spread never fit the training median, the plan's next step is about 100 new training documents aimed at those heads, plus a new 24-document exam. The first exam is not reused to claim that the second model improved.

## Round 2

Grok 4.7, the normal one, wrote 98 more training documents and a fresh 24-document exam. The auditor kept all 122. None of the ids overlap the original 48 or the first exam. The 98 are training rows only. They were not mixed into val or test. The fresh exam is `data/exam2/locked.jsonl`. The first exam was not scored again.

The new training mix, measured by the auditor: 152 text spans, 60 missing text fields, 72 lists, 84 numbers, 124 dates, 12 times, 18 durations, 14 estimates, 12 forecasts.

Same training rule as round 1. Epoch 93 had the highest training field exact-match, 744/769. The missing threshold stayed 0.5. Random initialization is the "before" column.

| Split | Fields | Before | After | Documents after |
| --- | --- | --- | --- | --- |
| Train, 138 docs | 769 | 17/769 | 744/769 | 113/138 |
| Val, the original 4 | 24 | 0/24 | 4/24 | 0/4 |
| Test, second look at the original 4 | 24 | 0/24 | 4/24 | 0/4 |
| Fresh exam, 24 docs, scored once | 95 | 3/95 | 14/95 | 0/24 |

By head on the fresh exam: point 14/51, number 0/16, calendar 0/22, spread 0/6. On the training documents, point, number, and calendar were again 100%. Estimates were 8/18 and forecasts 3/18. Text spans on the fresh exam were 13/39. Dates were 0/20. Lists were 1/12. No full document was right.

The test moving from 3/24 to 4/24 is one field on a second look. It is not a new exam.

The number and calendar heads in that run never saw the document tokens. They read only the last question token. "Bill date" looks similar from one bill to the next, and the date itself sits in the document tokens those heads did not receive. A head can memorize which training question went with which date, score 159/159 on the training dates, and then write the wrong day on every new bill. The point head is different: it compares the question with every document token, which is why copied text was the only field that moved on a new document.

## Round 3

Number, calendar, and spread now pool the document tokens with weights that depend on the question. The pool is a small attention block: LayerNorm on the document and on the question, a query and a key of width 64, a value of the full 1536-wide state, scores scaled by `1/sqrt(64)`, then softmax. That pool is added to the question vector, LayerNorm is applied again, and the same small network writes the digits, the calendar indexes, or the mean and width. The point head already compared the question with every document token, so it was left as it was. Gemma stays frozen. Thinking stays off. The training documents are still the same 138. The epoch rule and the learning rate are unchanged.

Epoch 73 of 80 had the highest training field exact-match, 720/769. The run did not extend, because that epoch was not inside the last 5 of the budget. The missing threshold stayed 0.5. Random initialization is the "before" column.

| Split | Fields | Before | After | Documents after |
| --- | --- | --- | --- | --- |
| Train, 138 docs | 769 | 15/769 | 720/769 | 90/138 |
| Val, the original 4 | 24 | 0/24 | 5/24 | 0/4 |
| Fresh exam, 24 docs, scored once | 95 | 3/95 | 15/95 | 0/24 |

The claim is `data/exam3/locked.jsonl`. The auditor kept all 24. None of the ids overlap the original 48, the added 98, or the first two exams. Exam 1 and exam 2 were not scored again. The original test was not scored again.

By head on this exam: point 14/51, number 0/16, calendar 0/22, spread 1/6. Text spans were 14/39. Lists were 0/12. Dates were 0/20. Copied and calculated amounts were 0/12. Durations were 0/4. Estimates were 1/3. Forecasts were 0/3. No full document was right. On the training documents the same checkpoint scored point 390/390, number 156/164, calendar 169/179, and spread 5/36. Forecast steps on the training documents were 0/18.

A wrong amount on this exam is a different money figure. `e3-bills-01` asks for 1460.75 and the head wrote 22.00. A wrong date is a different day, still in 2026 or 2027. `e3-bills-01` asks for 2026-08-11 and the head wrote 2027-06-14. On a training bill the head can also swap two dates that share a document: `bills-01` billed_on was predicted as 2026-04-16, which is that same bill's pay-by date, while pay-by itself was right. Seeing the document tokens was not enough for the head to pick the asked amount or the asked day on a new note.

The pre-registered fail rule: a head fails when the fresh exam's field exact-match is under 50% and that head has at least 8 fields. Point, number, and calendar fail that test. Spread has 6 fields on this exam, so the count is reported and those 6 fields are not, by themselves, a reason to add a hundred notes. Forecasts are already 0/18 on the training documents, so another pile of the same forecast shape is not what the exact step-snap is missing.

Training is 138 documents. The ceiling for this synthetic set is about 200. The next measured step is the remaining documents, about 62, aimed at point, number, and calendar, plus a fourth 24-document exam written by the same Grok 4.7 and locked before it is scored. Exam 3 will not be scored again to claim that later model improved. Real customer mail is still not on disk.

## What the blog must not say

- That Cloudflare published Clairic's head code. They published the output types. The four heads are our reading of those types.
- That Mark and Measure is how NVIDIA or Google builds data. They do SDG with validators. The bracket-and-measure step is this project's method.
- That the writer "just knows the character offsets." It knows the words. The program counts.
- That 48 audited rows, or a future 30,000 cloned rows, are enough.
- That estimate and forecast labels are real-world truth. They are rules applied to numbers the writer placed in the document.
- That any training round is a production accuracy. Round 1 fit 213 of 221 training fields and 29 of 112 exam fields. Round 2 fit 744 of 769 training fields and 14 of 95 fields on a fresh exam. Round 3 fit 720 of 769 training fields and 15 of 95 fields on a third exam. Number and calendar were still right on none of that third exam's dates or amounts.
- That a passed auditor means the English is diverse. The auditor has no opinion about style. The repeated field shape is a quality bug it cannot see.

## File map

- `extractor/audit.py` — the measuring program.
- `tests/test_audit.py` — spans, leaked sums, date shift, even-comp median, forecast tail removal, and the keep/drop split.
- `data/raw/support.jsonl`, `bills.jsonl`, `ops.jsonl`, `series.jsonl` — writer output, brackets still present. 12 documents each.
- `data/audited/train.jsonl`, `val.jsonl`, `test.jsonl` — clean text and measured labels.
- `data/audited/report.json` — counts and drop reasons.
- `data/generalization/raw/` — the locked 24-document exam, brackets still present. `locked.jsonl` is the audited form.
- `extractor/labels.py`, `heads.py`, `gemma_cache.py`, `train_heads.py`, `score_heads.py` — question packing, the four heads, the tiny-vocabulary forward, training, and exact-match scoring.
- `data/round2/raw/` — 98 added training documents, brackets still present. `data/round2/audited.jsonl` is the measured form. All of it is train. None of it was put into val or test.
- `data/exam2/raw/` and `data/exam2/locked.jsonl` — the fresh 24-document exam. This is the round 2 claim. The first exam was not scored again.
- `data/exam3/raw/` and `data/exam3/locked.jsonl` — the third 24-document exam. This is the round 3 claim. The first two exams were not scored again.
- `outputs/heads.pt`, `outputs/scores.json` — round 1 weights and the one-shot scores. `outputs/heads_round2.pt` and `outputs/scores_round2.json` are the second run. `outputs/heads_round3.pt` and `outputs/scores_round3.json` are the document-reader run, scored once on exam 3. `outputs/cache/docs/` is the hidden-state cache on disk and is gitignored.
- `blogs/mark-and-measure.md` — this note.

## Open threads for the essay

- Show the bracketed Hetal message and the clean training row side by side. That pair is the whole method.
- Then show a shift that is not another WhatsApp: the Colombo repair card (`6 × 850 = 5100`, with the 2200 quote left unmarked) or the Howrah cups, where 61 and 63 are deleted from the student text.
- Name the prior art in a short paragraph so the coinage stays honest.
- Spend time on the leak rule and the decoy. Those two are the difference between a labeled story and a labeled exam.
- The spread head is fed, and its gold is a rule: a median, or a withheld tail. Say that before anyone reads it as a price the market produced.
- End on the limit: the program can certify a label inside a document it can parse. It cannot certify that the document looks like the user's real inbox.
