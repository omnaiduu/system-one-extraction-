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
3. **Student.** Gemma 4 E2B, later, frozen, with small heads. It sees the clean text and the questions. It never sees a bracket. Training has not been run.

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

The situation is chosen by the writer, not rolled by a fact sheet. Each of the twelve kept documents is a different life: a blouse, a goat vet, a dental crown, a damp machiya, a cello. That is the randomization. There is no seed that swaps names inside one skeleton.

The label is not random. After the words exist, the program either measures them or drops them.

The **shift** is the thing the student has to survive: the same kind of answer sitting in a new sentence.

- In Hetal's WhatsApp, the job code starts at character 134 and the money is a sum of three payments.
- In the held-out test letter, `support-10`, Helena in Belo Horizonte writes in Portuguese. The cello contract `BH-5529` sits in a different place. The money is a subtraction, `730.60 - 185.40 = 545.20`. The loan is 72 days, from 20 January 2026 to 2 April 2026. An unmarked decoy, `BH-17` and `318` reais for the case, is in the letter and is not the label.

Train, val, and test are whole documents. Fields from one story are never split across the sets, because the model would then be tested on a paragraph it already read. After sorting ids, index mod 10 chooses the bucket: 8 is val, 9 is test, everything else is train. Today that is `support-09` val, `support-10` test, and the other ten in train.

This shift is real and still narrow. All twelve are support messages from one writer. A head can memorize the repeated shape: one code, one list, one money calculation, one day count, one date, one missing id. Passing the cello letter does not prove the heads work on a hospital bill or a forecast.

## Calculated answers the search cannot see

These are specified in `ask` and implemented in the auditor. The writer marks only the inputs.

- `sum` of every mark with one shared name, at least two parts.
- `product` of a left mark and a right mark, such as hours times a rate.
- `subtract` of left minus right, such as a deposit minus a fee.
- `days_between` two date marks. The day count must not be printed.
- `add_days` on a calendar field. "Issued 2 Sep 2026, net 30" becomes `2026-10-02`, and that ISO date must not appear in the text. This is the Clairic "net 30" case. No such row has been kept yet. The bills file that was meant to hold them never landed on disk.
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
5. Spread head: loss on the bin of the median, or on each withheld forecast step. No rows for this head exist yet, so it cannot be trained from this folder.
6. Score the heads on `support-10` only after training. That letter was not in the training file.

Confidence comes later. A sharp softmax is not a probability until it is checked on documents the generator never saw, including real ones. Clairic says the same thing in their preview notes: intervals can run narrow on unfamiliar data.

## Quality of the data on disk today

`data/audited/report.json` after the only successful writer:

- 12 kept, 0 dropped, from `data/raw/support.jsonl`.
- 10 train, 1 val, 1 test.
- Field counts: 15 dates, 12 durations, 12 lists, 12 numbers, 12 text spans, 12 missing text fields.
- No time, datetime, add-days, estimate, or forecast rows.
- Three other writers (bills, ops, series) reported files that are not on disk. They are not part of the dataset. Do not describe them as kept.

The labels that exist match the text. The sums and the day counts were produced by the auditor. The prose is specific.

The shape is too regular, and the set is too small to train the four heads. Twelve checked messages prove the checker. They do not prove a model. More volume is justified only after new document types pass the same auditor. Copying Hetal's skeleton with new numbers would inflate the count and teach the skeleton.

Real customer mail is still required before anyone calls the heads production-ready. This pipeline cannot invent that mail. A few hundred real documents, run through the same measurement rules where the evidence can be marked, are the actual shift that matters.

## What the blog must not say

- That Cloudflare published Clairic's head code. They published the output types. The four heads are our reading of those types.
- That Mark and Measure is how NVIDIA or Google builds data. They do SDG with validators. The bracket-and-measure step is this project's method.
- That the writer "just knows the character offsets." It knows the words. The program counts.
- That 12 rows, or a future 30,000 cloned rows, are enough.
- That estimate and forecast labels are real-world truth. They are rules applied to numbers the writer placed in the document.
- That training has happened.
- That a passed auditor means the English is diverse. The auditor has no opinion about style. The repeated field shape is a quality bug it cannot see.

## File map

- `extractor/audit.py` — the measuring program.
- `tests/test_audit.py` — spans, leaked sums, date shift, even-comp median, forecast tail removal, and the keep/drop split.
- `data/raw/support.jsonl` — writer output, brackets still present.
- `data/audited/train.jsonl`, `val.jsonl`, `test.jsonl` — clean text and measured labels.
- `data/audited/report.json` — counts and drop reasons.
- `blogs/mark-and-measure.md` — this note.

## Open threads for the essay

- Show the bracketed Hetal message and the clean training row side by side. That pair is the whole method.
- Then show the cello letter as the shift: new language, new position, subtraction instead of a sum, same heads.
- Name the prior art in a short paragraph so the coinage stays honest.
- Spend time on the leak rule and the decoy. Those two are the difference between a labeled story and a labeled exam.
- Say plainly that the spread head is designed and unfed.
- End on the limit: the program can certify a label inside a document it can parse. It cannot certify that the document looks like the user's real inbox.
