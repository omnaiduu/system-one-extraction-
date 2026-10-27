# Handoff for the next agent

This is the operational note. `blogs/mark-and-measure.md` is the essay log of the same work: the data method, the round-by-round scores, and the claims that must not be made. Read that for the method. Read this before changing the model.

The current student is a frozen Gemma 4 E2B, one forward, one point head, and a program that reads the marked characters. On the latest 24-document exam the program's value is **42 of 74** fields, **4 of 24** documents. The same head is right on **968 of 970** training fields. The last change that retrained anything moved that exam by one field. The two changes after it were rules on scores that were already trained, and they are used up.

A value that is not written on the page cannot be a span. The span method only covers a computation when a program was already coded for that exact rule and the inputs are text that program can parse. A number or a date written in words, or a computation outside those rules, needs a model trained to emit the value. The earlier conclusion that said "leave dates and amounts on the program" is withdrawn. The reason is in "Where that conclusion is wrong."

## What was fixed before any of this code

The target is a Clairic-style extractor. [Cloudflare's note](https://blog.cloudflare.com/clairic-extraction-model) (19 October 2026) describes the outputs: a span of the original text, a number, a real calendar value, a duration, a list, an estimate with a band, or a short forecast. It reads the document once and does not write a JSON reply. The weights and the head code are not public. The four heads in this repo are this project's reading of those output types. Do not write that Cloudflare published them.

Gemma 4 E2B-it is the student (`google/gemma-4-E2B-it`). It stays frozen, including the per-layer embedding tables. Thinking stays off. The chat model is not asked to emit the answer.

Paths that were considered and set aside, before training:

- Asking Gemma for JSON. That is still a generator. It can invent an order id. Cloudflare's note said a plain Gemma 4 run failed when it had to invent a repair cost.
- A grammar that only forces valid JSON. The shape is legal. The content can still be wrong.
- A LoRA that memorizes one fixed schema. A new field needs a new tune.
- One shared pointer for every field, with no program. A pointer can copy `HT-3391`. It cannot add three payments, force a real calendar day, or return a band.
- Filling one template with random facts. The labels are easy and the language repeats. Thirty thousand copies of one skeleton were rejected for that reason.
- Letting the writer report character indexes or totals from memory. The writer writes the story and counts badly.

What remained is Mark and Measure. A writer model invents a document and wraps evidence in `[[name]]words[[/]]`. `extractor/audit.py` deletes the brackets, counts the characters, computes anything that was not written, and drops the whole document if the calculation was already printed. The student sees only the clean text.

Three roles, and they stay separate:

1. **Writer.** Grok 4.7, the normal one. It does not report character numbers. It does not write the sum, the day count, the due date, the median, or the forecast the student is supposed to learn.
2. **Auditor.** No taste and no memory of the story. One bad field drops the whole document.
3. **Student.** Frozen Gemma, small heads. It never sees a bracket.

The user is out of writer tokens. The training set is locked at the 200 documents already on disk. The exams are the four 24-document sets already on disk. Forecasts stay out of the highlighter score, because the answer was removed from the text and there is no span to mark. Do not write new documents to chase the remaining misses. Do not score a new exam and then use it to pick the next change.

## The data the student actually sees

Character offsets are Unicode code points in the clean string, the same indexes Python uses. They are not token indexes. Token indexes are built at cache time and checked against the slice.

A decoy is an ordinary unmarked string. `HT-908` sits in Hetal's message and is not the job code. Absence of a bracket is what makes it a trap. The auditor has no list of traps.

Derived answers the writer must not print:

- `sum` of at least two marks, `product` of two marks, `subtract` of left minus right.
- `days_between` two dates. The day count must not be printed.
- `add_days`. "Credit runs 14 days from 2 Apr 2026" becomes `2026-04-16`, and that date is not printed.
- `estimate`: an even number of comparison marks, at least four. Gold is the average of the two middle values after sorting. An odd count would make the median one of the printed prices, and the head could cheat by copying it. The median must sit inside `min` and `max` and must not be printed. This gold is a rule applied to numbers the writer placed. It is not an appraiser's judgment.
- `forecast`: the last `horizon` observation lines are deleted from the student text and saved as the label. The answer is absent on purpose. Forecast gold is the withheld tail of a series the writer composed, not a future that happened.

Train, val, and test are whole documents. After sorting the original ids, index mod 10 chooses the bucket: 8 is val, 9 is test, everything else is train. Val is `bills-09`, `ops-07`, `series-05`, `support-03`. Test is `bills-10`, `ops-08`, `series-06`, `support-04`. Later additions were not mixed into val or test.

Counts:

| Split | What it is | Used for |
| --- | --- | --- |
| Train, 200 docs | Original 40, plus 98 from round 2, plus 62 from round 4 | The only set that picks the epoch |
| Val, 4 docs | Original holdout | Picks the missing threshold, and only as a tie-break |
| Test, 4 docs | Original holdout | Scored after the choice is frozen |
| Generalization, exam2, exam3, exam4 | Four locked 24-document exams | Scored after the choice is frozen. Exam 4 is the claim |

Round 4's mix, measured by the auditor on the 62 new training documents: 30 text spans, 22 missing text fields, 36 lists, 43 numbers, 63 dates, 6 datetimes, 7 times, 9 durations, 3 estimates, 3 forecasts. Another 60 notes of that same shape were judged not to be the missing piece. The head could already see the document. It fit the notes it trained on and missed the amount and the day on a bill it had not seen.

## How a forward is built

`extractor/gemma_cache.py` does not load the 10GB checkpoint. The text tower is about 2.5GB. The token and per-layer embedding tables are larger than this machine. Each document is tokenized first. Only the rows for tokens that appear are copied into a small embedding table. Layer weights are copied from the safetensors file one tensor at a time. The local snapshot path is the `SNAPSHOT` constant in that file, overridable with `GEMMA_SNAPSHOT`.

The document is the prefix of one packed prompt. Questions are appended:

```text
{clean document}

QUESTION
{field}: {instruction}
```

One forward covers every question. The document states are the last-layer vectors at the document token positions. The question state is the last token of that question body. Width is 1536. Cached states are float16. The head trains in float32 on those cached states. The backbone is not in the training step.

Question wording, from `extractor/highlight.py`:

- A copied string, a printed date, or a printed number uses the field instruction alone. Example: `Printed charge on the slip`.
- A list is `Item 1`, `Item 2`, … and one extra item whose gold is missing, so the head can learn where to stop. Example: `Each supply line printed on the bill Item 1.`
- A sum, product, subtraction, or median uses `Part 1`, `Part 2`, … and one extra part whose gold is missing.
- A day count appends `The start date.` and `The end date.`
- A due date appends `The start date.` and `The day count.`

`Item 1` and `Item 2` are the same instruction plus a different number. Their question vectors are close. That becomes important below.

There are two caches. `outputs/cache/highlight/` is the open causal read: the question token has attended through the document. `outputs/cache/highlight_blocked/` is the blocked read. Both are gitignored. The weight files and the score JSON files under `outputs/` are tracked.

## The point head, the loss, and the decoder

`PointHead` in `extractor/heads.py`:

- LayerNorm on the document states and on the question vector.
- A query and two keys, inner width 64. Start score and end score are scaled dot products of the question against every document token.
- Missing logit = `missing_question(question) + missing_match([start.amax(), end.amax()])`. The match features are the single best start token and the single best end token, not the endpoints of the span the decoder later picks.

`point_loss` is binary cross-entropy on the missing bit, plus cross-entropy on the gold start index, plus cross-entropy on the gold end index, and only when the gold is present. Start and end are two separate token classifications. Nothing in the loss scores a span as a pair, and nothing says two parts of a list must not overlap.

At score time, `choose_span` picks the pair with the highest start score plus end score whose length is inside a cap. `choose_spans` places several of those pairs in text order. Those two functions are not in the loss. The missing threshold is 0.5. A part the missing score drops is never handed to the chain.

The program in `extractor/highlight.py` then reads the characters and applies the auditor's rule: copy, sum, product, subtract, parse a date, count days, add days, or the median. `verify_corpus` checks that the program reproduces every audited value from the gold marks. When an exam field is wrong, the mark is wrong, except for two exam-4 fields noted below where a slightly wrong mark still parses to the gold value.

`value_exact` can be true when `span_exact` is false. On the current exam-4 score, the token span is 40/74 and the program's value is 42/74. The two extras are `e4-bills-02` settle-by (the day-count tokens are not the gold tokens, the characters still read as 21 days, the date is 2026-06-25) and `e4-ops-06` ancillary (the middle tokens are not the gold tokens, and 14.65 + 26.8 + 7.35 is still 48.80). Optimizing the value number hides span errors. Report both.

## What was tried, in order

### 1. Four heads, rounds 1–4

One Gemma read, then the field type picks a head. Field names share a head. "Job code" and "ceremony date" are not two models.

| Head | What it emits | Why it exists |
| --- | --- | --- |
| Point | Start token, end token, missing bit | The answer is a slice of the text, a list of slices, or empty |
| Number | Sign and digits | A pointer cannot mint `2413.25` from three payments |
| Calendar | Legal year, month, day, hour, minute | Month 13 is not in the output space |
| Spread | A mean and a width inside the caller's min/max | The median, or a forecast step |

Round 3 added `DocumentReader`: the question produces attention weights over the document tokens, and that pooled vector is added to the question before the number, calendar, and spread heads. The reason was that a date or an amount lives in the document, and those heads were only seeing the question vector.

Training rule, all four rounds: earliest epoch with the highest training field exact-match. Validation is four documents, so it does not rank epochs. It only picks the missing threshold from {0.5, 0.3, 0.7}. Seed 7, AdamW, learning rate 1e-3, weight decay 0.01, gradient clip 1.0, 80 epochs, extend by 20 up to 120 if the best epoch is still inside the last five. Exams are scored once, after that choice is frozen.

| Round | Train | Fresh exam | What the exam said |
| --- | --- | --- | --- |
| 1 | 213/221 fields | 29/112 | First locked exam |
| 2 | 744/769 | 14/95 on exam 2 | Extra 98 training documents |
| 3 | 720/769 | 15/95 on exam 3 | Document reader. This exam is not the same exam as round 2 |
| 4 | 948/991, 158/200 docs, epoch 92 | 9/77 on exam 4, 0/24 docs | 200 training documents |

Exam 4 by head: point 8/33, number 0/16, calendar 1/22, spread 0/6. Amounts 0/12. Dates 1/20. The one right date is `e4-series-03`, 4 March 2026. A wrong amount is a different money figure. `e4-bills-01` asks for 1486.75 and the number head wrote 2186.75. `e4-bills-05` asks for a balance of 2825.50 and the head wrote 17.00. `e4-bills-01` asks for 17 March 2026 and the calendar head wrote 4 July 2026.

On the training documents the same checkpoint scored point 474/478, number 214/216, calendar 253/255, spread 7/42. Forecast steps on the training documents were 0/21.

9/77 and the earlier 15/95 are two different exams, each scored once. The pair does not say the model got worse, and it does not say the extra 62 documents helped. It says the heads fit the notes they trained on and still miss the amount and the day on a note they have not seen.

200 synthetic documents is the stop for this set. The miss was no longer "the head cannot see the document."

### 2. One highlighter

The number head and the calendar head were dropped. One point head remains. Every printed value, and every piece of a sum, product, subtraction, day count, due date, or median, becomes a span question. The program reads the marked characters and applies the same rule the auditor used.

The reason: a wrong number head invents a nearby amount. A wrong calendar head invents a legal day. A wrong mark gives the program nothing, or the wrong characters, and the arithmetic is then checkable. Forecasts are skipped. Exam 4 has 3 of those, which is why this score has 74 fields and round 4 had 77.

No new documents. The four exams were scored after the epoch was frozen. They had already been read when this run was chosen, so it is a re-check, not a new exam.

Epoch 70 of 80. Training value 959/970, 183/200 documents. Exam 4 value **28/74**, one document fully right (`e4-bills-05`). Dates 14/20. Amounts 4/12. Lists 0/12. Sums 0/3. Missing fields 5/8.

The date move against round 4 (1/20 to 14/20) is the copied span. A printed day was already the kind of thing the point head could mark. The calendar head had been inventing a day from the question vector. The amount move is smaller, 0/12 to 4/12. On `e4-bills-01` the fee span was empty, so the program returned nothing, where round 4 had written 2186.75 for a fee of 1486.75.

Lists grabbed a long stretch of the note. `e4-bills-02` marked most of the bill as the second charge name. Estimates were 0/3 here and 21/21 on the training notes. The highlighter fit the 200 texts. It did not learn to mark a new note.

Weights: `outputs/heads_highlighter.pt`. Scores: `outputs/scores_highlighter.json`.

### 3. Question block

The question token in the open forward has already attended through the document. Its vector is a question that has read the note, including the answer. The block stops that.

`apply_question_block` sets `mask[:, :, doc_tokens:, :doc_tokens]` to the dtype's minimum. A question token can attend to itself and to earlier question tokens. It cannot attend to document tokens. Document rows keep the ordinary causal mask, so they do not attend to questions either. Later questions still see the text of earlier questions (`Item 1` is visible to `Item 2`). They still do not see the note.

On the first blocked note, with one CPU thread, the largest absolute gap between the blocked document states and an open read was 0. The question states did move. The largest absolute gap there was 69.5. The cache refuses to continue if the document gap is above 1e-3 or the question gap is 0.

Several CPU threads make this bf16 forward disagree with itself. The same open forward, run twice on many threads, can differ by about 17–25 in max abs. The blocked cache sets `torch.set_num_threads(1)` before the model is built, and it must stay there. Raising the count and then lowering it still lets the forward disagree with itself. Do not "fix" a failing equality check by loosening it until a multi-thread run passes.

The missing score also gained the match features described above. That was retrained. Same 200 texts. The run extended from 80 to 100 because epoch 80 was still inside the rising window. Best epoch 85, training span 962/970. All three thresholds scored 2/24 on the four validation notes, so 0.5 stayed. Training value 963/970, 186/200 documents. The one field whose span is wrong and whose value is still right is `r3-bills-14` balance: the marked tokens are off, and the characters still subtract to 1259.50.

Exam 4 value **29/74**. Lists stayed 0/12. Sums stayed 0/3. Missing fields went from 5/8 to 8/8. Copied strings went from 3/13 to 5/13. Dates fell from 14/20 to 12/20. Amounts fell from 4/12 to 2/12. `e4-bills-05` had been the one fully right document. Its balance span came back empty, so no fourth-exam document was fully right. Exam 3 moved more, 24/92 to 38/92.

This was the check the run was for. The question state can be trained without having already read the answer in that same vector. The lists still ran past the item. On `e4-bills-02` the third charge name was the right span and the extra item was correctly missing, and the first two names continued through the rest of the bill.

Weights: `outputs/heads_highlighter_blocked.pt`. This file is the last set of trained weights. Everything after it is a rescore.

### 4. Span cap

The blocked head picked the start and the end separately. Each one was the single highest score. On a short item the end often landed many tokens later, or before the start.

The training notes already say how long these spans are. The readout keeps the pair with the highest combined score whose length is inside that maximum. Ties keep the earlier start, then the earlier end. Nothing was retrained. The cap was not used to choose epoch 85 or the threshold.

| Operation | Max training tokens |
| --- | --- |
| copy | 10 |
| list | 8 |
| read_number | 9 |
| sum | 11 |
| read_datetime | 16 |
| read_date | 10 |
| add_days | 10 |
| product | 8 |
| read_time | 5 |
| subtract | 10 |
| days_between | 10 |
| median | 6 |

No gold training span is longer than its cap. Two gold spans on exam 4 are. Both are printed amounts with a currency prefix, and the tokenizer glues the leading space onto `USD`:

- `e4-bills-01` fee, ` USD 1,486.75`, 10 tokens, cap 9.
- `e4-bills-03` fee, ` USD 2,075.40`, 10 tokens, cap 9.

The cap cannot mark those two, even if the head's logits on those tokens are perfect. The best legal pair is then some other span. On `e4-bills-01` that pair is the date `17 March 2026`, so the program returns nothing for 1486.75. On `e4-bills-03` the fee is called missing.

Exam 4 value **39/74**. Lists 1/12 (`e4-support-01`: reef spray, gust front, low cloud). Sums 1/3 (`e4-bills-06` materials, 553.75). Amounts 6/12. Dates 14/20. Three documents fully right: `e4-bills-05`, `e4-ops-01`, `e4-support-01`. No field that was already right went wrong. Training value 967/970, 191/200 documents.

The same cap on the earlier, unblocked head moves exam 4 from 28/74 to 34/74. Lists there are 1/12 and sums stay 0/3. Most of the jump from 28 to 39 is the decoder. The block adds a smaller amount on top of the same rule.

The cap removes marks that run away to the bottom of the note. It does not decide which short span is the item. Eleven of the twelve lists were still wrong.

### 5. Ordered parts

A list and a sum were still choosing each part on its own. Two parts could mark the same short span, and a later part could sit to the left of an earlier one. A pass over the remaining wrong lists and the remaining wrong sum showed those collisions. `e4-support-05` and `e4-bills-06` were the examples that moved.

The missing bit is unchanged. The parts marked present are one chain. Each later part starts after the previous part ends. Each span stays inside the cap. The chain is the one with the highest total of start score plus end score. A copied string, a date, a product, and a due date stay on the separate picker. A due date's day count can sit before the date, so a text-order chain would make that gold pair illegal. Nothing was retrained. The chain was not used to choose the epoch.

Exam 4 value **42/74**. Token span 40/74. Lists 3/12. Sums 2/3. Amounts 7/12. Dates stay 14/20. Text stays 15/21. Four documents fully right: `e4-bills-05`, `e4-bills-06`, `e4-ops-01`, `e4-support-01`. Training value 968/970, 192/200 documents. Training lists 135/136. Training sums 31/31. No field that was already right went wrong.

The two new lists are `e4-bills-06` items (cedar sill board, galvanised hinge pair, roofing felt roll) and `e4-support-05` tasks (linen count, candle stubs, guest cards). The new sum is `e4-ops-06` ancillary, 48.80, from tokens that are not all the gold tokens. The new training field is `r3-support-13` notes: it had repeated its last item, and the chain now marks reset the door lock, cleaned the sill track, tightened the rope clamp, and tested the alarm bell. The one training list still wrong is `r2-support-16` to return, which shortens `marbled endpaper` to `paper`. That short span does not sit on another part, so the chain leaves it.

The chain moves a repeated item and an item that sat out of order. It leaves a short edge that does not collide, a distractor that is already in order, and a part the missing score drops.

## Scoreboard

Program value, exact field match. The span count is one or two fields lower where a wrong mark still parses.

| Run | Weights | Decoder | Train | Exam 4 | Lists | Sums | Amounts | Dates | Docs fully right |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Four heads, round 4 | `heads_round4.pt` | each head emits its type | 948/991 | 9/77 | 0/12 | — | 0/12 | 1/20 | 0/24 |
| Highlighter, question can see the text | `heads_highlighter.pt` | independent argmax | 959/970 | 28/74 | 0/12 | 0/3 | 4/12 | 14/20 | 1/24 |
| Question blocked | `heads_highlighter_blocked.pt` | independent argmax | 963/970 | 29/74 | 0/12 | 0/3 | 2/12 | 12/20 | 0/24 |
| Same blocked head | same file | span cap | 967/970 | 39/74 | 1/12 | 1/3 | 6/12 | 14/20 | 3/24 |
| Same blocked head | same file | span cap, list and sum chained | 968/970 | 42/74 | 3/12 | 2/3 | 7/12 | 14/20 | 4/24 |
| Unblocked head, cap only | `heads_highlighter.pt` | span cap | — | 34/74 | 1/12 | 0/3 | — | — | — |

Other exams, blocked head, cap then chain. These were not used to pick anything.

| Split | Cap, value | Chain, value |
| --- | --- | --- |
| Val, 4 docs | 5/24 | 5/24 |
| Test, 4 docs | 9/24 | 9/24 |
| First exam | 48/109 | 48/109 |
| Exam 2 | 39/92 | 42/92 |
| Exam 3 | 48/92 | 49/92 |

Exam 4 by operation on the chained score: printed date 13/18, due date 1/2, day count 1/4, copied string 7/13, list 3/12, median 0/3, missing field 8/8, product 3/4, printed number 1/3, time 1/1, subtraction 1/2, sum 2/3.

## The 32 exam-4 fields still wrong

Taken from `outputs/predictions_highlighter_blocked_chained.json`. Gold, then what the program returned.

Clipped edge of the right phrase. The chain has nothing to repair, because the short span does not collide with another part.

| Id | Field | Gold | Got |
| --- | --- | --- | --- |
| `e4-bills-02` | lines | harbour chart folio, pilot lamp set, waxed envelope pack | harbour chart, pilot lamp set, waxed envelope pack |
| `e4-ops-02` | parts | lagoon pump seal, brass cotter pin, split hose clamp | lagoon pump, brass cotter pin, split hose clamp |
| `e4-ops-03` | notes | keel zinc swap, packing gland reset, anode bond wire | keel zinc, packing gland, anode bond wire |
| `e4-series-06` | place | Saint-Georges | Saint |
| `e4-support-03` | bench_tag | BP-NSY-6084 | BP-NSY-608 |

A real part called missing, so the field is blank or short. The chain never sees a part the missing score has dropped.

| Id | Field | What was dropped |
| --- | --- | --- |
| `e4-bills-03` | fee | the whole amount, called missing. The gold span is also over the 9-token cap |
| `e4-ops-04` | packets | salt-stained logbook and copper earth strap called missing. spare fuse card clipped to spare fuse |
| `e4-ops-05` | ticket | PAP-6407 called missing |
| `e4-ops-06` | jobs | night latch check called missing. The first mark is `total under them. A crate wash`, the second is `ice top` |
| `e4-series-05` | place | Leticia called missing |
| `e4-support-02` | flap_phrases | blue twine called missing. The first mark is the code AL-BIN-3096, the second is `seal` |
| `e4-support-06` | fitted | bracket screws called missing. cable ties and fuse caps are right |

The right kind of span, the wrong instance. Often a decoy that is already in order, so the chain keeps it.

| Id | Field | Gold | What was marked |
| --- | --- | --- | --- |
| `e4-bills-01` | fee | 1486.75 | the date 17 March 2026. Gold span is over the cap |
| `e4-bills-03` | due | 2027-02-27 | start date is right, day count is `1,99` from the decoy 1,990.00, so the date lands on 2027-08-27 |
| `e4-bills-04` | jobs | hinge pin replacement, deck cleat weld, locker latch reset | `5519 covers hinge pin replacement`, deck cleat weld, `locker`, and the extra item was not called missing |
| `e4-ops-02` | issued | 2026-05-21 | 5.75 |
| `e4-ops-03` | balance | 716.75 | 8 and 147.80, so the subtraction is -139.80 |
| `e4-ops-03` | docked | 12 days | start date right, end mark is `28` |
| `e4-ops-05` | opened | 2026-06-14 | the token `14` |
| `e4-ops-05` | closed | 2026-06-29 | the opening date 14/06/2026 |
| `e4-series-02` | place | Sinnamary | 640 |
| `e4-series-04` | place | Oiapoque | Neri |
| `e4-support-03` | cover_date | 2027-01-14 | 9 August 2026 |
| `e4-support-03` | tray_span | 25 days | start date right, end mark is `026-09-03` |
| `e4-support-04` | store_lines | mooring rope, lamp oil, wharf chalk | mooring rope, 27.15, 9.60 |
| `e4-support-04` | charges_sum | 55.15 | 61.20 and 27.15, third part called missing |
| `e4-support-05` | roster_span | 21 days | the same date twice, 06/11/2026, so the count is 0 |
| `e4-support-05` | cover_date | 2027-02-08 | 06/11/2026 |
| `e4-support-06` | labour | 91.65 | 5 and 18, so the product is 90.00 |

Estimates, 0/3. Training estimates are 21/21. The question is `Part 1` … `Part 8` plus a stop. Later parts are called missing, and a date or a glued pair gets into the early parts, so the median cannot be computed.

| Id | Gold | What was marked |
| --- | --- | --- |
| `e4-series-01` | 140 | 30; `30, 50`; 210; then five parts called missing |
| `e4-series-02` | 850 | 640 and 780 right, 920 and 1100 called missing |
| `e4-series-03` | 16 | `4 March 2`; `18, 25`; then four parts called missing |

## Why this is a bottleneck

The training set is memorized. 968 of 970 fields, 192 of 200 documents. The exam is about half the fields and 4 of 24 documents. More epochs on these 200 notes polish a set the head has already fit.

The last retraining, the question block, moved exam 4 from 28/74 to 29/74 and left lists and sums at zero. The length cap and the chain then moved 29 to 39 to 42 without touching a weight. They forbid a mark that is too long, a mark that repeats an earlier part, and a mark that sits out of order. The remaining 32 fields are the head pointing at the wrong tokens, clipping the edge of the right tokens, or calling a real span missing.

A further readout rule would be a name for one of those 32. "Snap to the word boundary" is the clipped list. "If the score is high, do not call it missing" is `bracket screws` and `PAP-6407`. "A fee question may not return a date" is `e4-bills-01`. Each of those describes a miss already on this exam. Designing it from this exam uses the exam twice.

The loss and the decoder disagree. Training asks for one start token and one end token. Scoring asks for the best legal pair, then for a chain. The missing logit is trained on the best single start and the best single end, which can be an illegal pair, and the decoder then picks different endpoints.

List and sum questions barely differ. `Item 1` and `Item 2` share the instruction. The chain papers over that by handing the peaks out from left to right. It works when the peaks are the items. It keeps the price, the code, and the other date when those are the peaks.

The question vector, under the block, has not seen the note. The only selection is that vector, matched in 64 dimensions against each document token. The document states themselves never see the question. If those states do not already separate "the fee" from "the date on the same slip," no rule on top of the scores will.

## Decisions that are easy to undo by accident

- The saved blocked checkpoint was selected at epoch 85 by independent argmax, before the cap and the chain existed. `train_highlighter.py` now calls `choose_span` and, for lists and sums, `choose_spans` inside every epoch's evaluation. A fresh `--blocked` training run would pick its epoch under the cap and the chain. That is a different experiment. Do not describe it as a reproduction of `heads_highlighter_blocked.pt`. The rescore path (`--rescore`) keeps the saved weights and the saved threshold.
- The cap and the chain were not used to choose the epoch or the threshold. The score files say so. Leave that true, or say plainly that a new run broke it.
- Val is four documents. It picks a threshold only when the three candidates disagree. On the blocked run they agreed, at 2/24. Do not start ranking epochs by val.
- Exam 4, and the three earlier exams, are not a development set. They were scored after the epoch was frozen. A rule written to fix a named miss on this list has used the exam.
- Do not write new documents. The user is out of writer tokens, and round 4 already called 200 the stop for this skeleton.
- Do not put forecasts back into the highlighter. There is no span in the student text. The four-head forecast steps were 0/21 even on the training documents.
- Do not loosen the blocked-cache equality check to get a multi-thread run to pass. Set one thread before the forward, and keep it there.
- `value_exact` is the program. `span_exact` is the mark. Quote both.
- Two exam-4 gold spans are longer than the training maximum. A perfect head still cannot mark them under the current cap.
- The essay log's "what the blog must not say" section still applies. In particular: these rounds are not a production accuracy, a passed auditor says nothing about English variety, and estimate labels are a rule on writer-placed numbers.

## What a different approach has to change

The shape that is still right: mark characters that are in the text, and let a program do the arithmetic. The four-head number and calendar outputs invented a nearby amount and a legal day. The highlighter stopped that kind of error. Keep the program.

The piece that is exhausted is this matcher plus another legality rule. A next experiment has to change what is trained, not how a frozen score is trimmed. The mismatches to close, if that is the direction:

- The loss scores a start token and an end token. The decoder scores a span, then a chain. Those should be the same object.
- A list is N copies of one instruction. The head is not asked which item is which, except by a number at the end of the question. One question that emits an ordered set of spans is a different head.
- The blocked question vector never sees the document, and the document states never see the question. The 64-wide dot product is the entire meeting place. If the next idea lets them meet earlier, it has to say what the question is allowed to see, because the open forward's question vector had already read the answer and the lists were still 0/12.
- The two over-cap amounts are a property of the decoder, not of the head. `USD 1,486.75` is 10 tokens because of the currency prefix and the leading space the tokenizer attached. Any cap taken from the training maximum will refuse them.

More notes of the same skeleton were already the round-4 conclusion, and the answer was no. More epochs on the current loss will move the training number that is already 968/970.

## Computed fields, and whether to go back to the four heads

A proposal was put to this note. Cloudflare's outputs include a copied span, a number, a calendar value, a duration, a list, an estimate, and a forecast. Some of those values are not written on the page. They are computed. The proposal said the span method cannot do a computed value, text is the only field the network must copy, dates and amounts showed no real difference, and the next agent should go back to the four heads.

The exam does not support that. The comparison below is the same 74 fields. The four-head score is usually written 9/77 because it also counts 3 forecasts, all wrong. The span runs do not score forecasts. The plain span column is `outputs/predictions_highlighter.json`: the dot-product head, the question can see the note, no length cap, no chain. The last column is the blocked head plus the cap and the chain, `outputs/predictions_highlighter_blocked_chained.json`.

| Kind | Four heads | Span, dot product only | Blocked span, cap, and chain |
| --- | --- | --- | --- |
| Dates | 1/20 | 14/20 | 14/20 |
| Amounts | 0/12 | 4/12 | 7/12 |
| Text | 8/21 | 8/21 | 15/21 |
| Lists | 0/12 | 0/12 | 3/12 |
| Estimates | 0/3 | 0/3 | 0/3 |
| Shared fields | 9/74 | 28/74 | 42/74 |

Six fields were right in both the four-head run and the plain span run. The plain span gained 22 and lost 3. Those 3 are missing text fields the four-head run had right: `e4-bills-01` insurer, `e4-ops-01` insurer, `e4-ops-04` permit. The chained score got those three back. Forty-three fields were wrong in both the four-head run and the plain span run.

Printed dates are the gap. On the plain span run, `read_date` is 13/18. The calendar head is 1/22. A printed day is on the page. The program does not invent it. The calendar head emitted a day from a blended vector, and on `e4-bills-01` it wrote 4 July 2026 for a stamp that says 17 March 2026.

Text did not move between the four heads and the plain span head. Both are 8/21. Copied strings on the plain span run are 3/13. Text is not the field the span method won. The later 15/21 includes the question block, the cap, and the chain. It is not the plain dot product.

A computed value, in the span method, is the program. The network marks the pieces that are written. The program adds, multiplies, subtracts, counts the days, or adds the days. On the plain span run, before either rule: product 2/4, subtract 1/2, a due date 1/2, a day count 1/4, a sum 0/3, a median 0/3, a printed number 1/3. The four-head number head was 0/16 on this exam. `e4-bills-01` asks for 1486.75 and that head wrote 2186.75. The plain span run returned nothing for that fee, because the mark was empty. Both are wrong. The invented number is the failure the span method was built to stop.

After the cap and the chain, the computed counts are product 3/4, subtract 1/2, sum 2/3, due date 1/2, day count 1/4, median 0/3. Two of the sum and amount wins are a right total from tokens that are not all the gold tokens (`e4-ops-06` ancillary, `e4-bills-02` settle-by). Count those as program values, and keep the span count beside them.

Lists are words on the page. The four-head point head and the plain span head both scored 0/12. The chain later scored 3/12. Estimates are the median of printed numbers. Both designs scored 0/3. Forecasts are not on the student page: the auditor deleted the tail. The span method skips them. The spread head scored 0/21 forecast steps on the training documents and 0/3 on this exam. Marking the page cannot be the whole answer for a forecast.

The table above is the score. It is not a decision about fields whose answer is not a span.

## Where that conclusion is wrong

The conclusion that used to stand here said: leave dates and amounts off the number head and the calendar head, let the program compute, and send the network back only to mark inputs. That conclusion is withdrawn. It is logically flawed.

It treats "the answer is not printed" as "the answer is a sum of clean numbers the program already knows." This repository only programs six computations: sum, product, subtract, days between two dates, add days, and the median of an even list of figures. Each one needs the inputs as text that `parse_number` or `parse_date` can read. A digit string. A date the parser already accepts.

That is not the general case.

- Sometimes the number is not on the page. The sum, the balance, the due amount was deliberately left unprinted. There is no span of the answer. Pointing at tokens cannot emit it. The program can emit it only if someone already coded that formula and the parts are spans it can parse.
- Sometimes the date is not on the page. "Credit runs 14 days from 2 Apr 2026" has a start and a count. The due date itself was not written. A span head has nothing to mark for the due date. A calendar head is the output that can emit `2026-04-16`.
- Sometimes the number is written in words. "Fourteen hundred," "two lakh," "net thirty." A span of those words is still words. No parser in this repo turns arbitrary wording into a number. Adding another length cap does not do it either.
- Sometimes the date is written in words, or in a form the parser does not accept. A span returns the words. It does not return a calendar value.
- Any computation that is not one of those six ops has no program. The span method then has no answer at all.

The exam scores were used as if they closed this. They do not. The 14 of 20 dates are mostly printed dates, `read_date` 13/18. The amounts the span run got right are digit strings, or digit strings run through a programmed op. This synthetic set was built so the writer never prints the total and always leaves clean parts. Winning on that set shows the program can add the cases it was coded for. It does not show the program can read a number out of prose, or compute a rule nobody wrote down.

Using "the calendar head scored 1/20 on printed dates" to forbid a trained head for dates that are not printed is the flaw. Those are different outputs. A printed day can be a span. A day that is not on the page cannot.

Forecasts were already this hole. The answer was deleted from the page, so the span method skips them, and the spread head also scored 0. The same hole is every unprinted number and every unprinted date, once the wording or the formula leaves the six coded ops.

What stands, as the correction:

- Text that is copied, and only that, has to be a span. There is nothing to compute. The characters are the answer.
- A number or a date that is not on the page cannot be spanned. The output has to come from a model trained to emit it: the number head, the calendar head, or a later head with that job. A program is not a substitute unless the formula and the parse are already coded for that exact case.
- A number or a date written in words is the same problem. The span is words. The required output is a number or a calendar value. This codebase cannot do that conversion programmatically for the ways people write them.
- The span-plus-program path is not capable of those edge cases. More decode rules will not make it capable. The model has to be trained to output the computed value.
- The score table stays in the note so the next agent can see what this particular exam measured. It is not a reason to keep dates and amounts off a head that emits them.

## How to reproduce the current numbers

From the repo root, with `.venv` (torch is not on the system Python):

```bash
# Blocked states. One thread is set inside the script. Skip if outputs/cache/highlight_blocked already has every document.
.venv/bin/python -m extractor.gemma_cache --highlighter --blocked

# Rescore the saved blocked head. Does not train. Writes the chained score files.
.venv/bin/python -m extractor.train_highlighter --blocked --rescore
```

Expect:

```text
train span 967/970 value 968/970
exam4 span 40/74 value 42/74
train list 135/136
train sum 31/31
exam4 list 3/12
exam4 sum 2/3
```

The span count on train is one below the value count because of `r3-bills-14`. The fresh-training command, which would select a new epoch under today's decoder, is `.venv/bin/python -m extractor.train_highlighter --blocked`. Do not run it to "confirm" the table above.

Decoder checks live in `tests/test_heads.py`: a far end loses to a short pair, an end before the start is repaired, a repeated item is refused, and the first span can shorten so the second fits. `tests/test_highlight.py` checks that the program reproduces audited values from gold marks.

## Where the files are

- `extractor/audit.py` — brackets, measurement, leak check, drop reasons.
- `extractor/labels.py` — the four-head questions and the packed prompt.
- `extractor/highlight.py` — span questions, the program, judging.
- `extractor/heads.py` — `PointHead`, `choose_span`, `choose_spans`, the four older heads.
- `extractor/gemma_cache.py` — tiny-vocabulary forward and the question block.
- `extractor/train_heads.py` — four-head training.
- `extractor/train_highlighter.py` — highlighter training and `--rescore`.
- `data/audited/`, `data/round2/audited.jsonl`, `data/round3/audited.jsonl` — the 200 training documents, plus the original val and test.
- `data/generalization/locked.jsonl`, `data/exam2/locked.jsonl`, `data/exam3/locked.jsonl`, `data/exam4/locked.jsonl` — the four exams.
- `outputs/heads_round4.pt`, `outputs/heads_highlighter.pt`, `outputs/heads_highlighter_blocked.pt` — the three weight files that matter for this note.
- `outputs/scores_highlighter_blocked_chained.json` and `outputs/predictions_highlighter_blocked_chained.json` — the current claim.
- `outputs/cache/` — hidden states, gitignored, required for a rescore.
- `blogs/mark-and-measure.md` — the essay log, including the per-round tables this note compresses.

Branch at the time of this note: `feature/single-highlighter-bac6`. The chained decoder is commit `2572437`. This handoff is the commit on top of it.
