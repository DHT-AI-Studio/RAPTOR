# Temporal primitives (TMP-5 / TMP-6), extraction (P11), durable temporal jobs (P12), capture / publish anchors (P13), source lifecycle (P14) and TKG contract v2 (P08 / P09)

`app.temporal.normalize.normalize(text, timezone="UTC", source_published=None)`
returns start/end endpoint objects `{state,lo_ms,hi_ms}`, extent, precision,
anchor, raw evidence, timezone and normalizer_version. Calendar periods have
exact closed UTC-millisecond boundaries; a structured timestamp is a point.
`since YEAR` retains the uncertain start domain for that year; end.state is
ongoing with no invented end, and confirmed_active_at is the explicit source
publication instant. The start's upper bound cannot exceed that confirmation.
Unresolved endpoints have null bounds, state=unresolved,
extent=UNRESOLVED and a reason. No server clock is used as an implicit anchor.

Supported grammar: ISO year/month/date/timestamp, explicit offsets, EN month
names, decades, quarters, northern meteorological seasons, year/day ranges;
ZH Gregorian/ROC years, digit and place-value numerals, 年代, quarters and
上下半年; source-relative years/months/days/weeks and civil clock times.
A civil date uses the supplied IANA timezone; defaults use UTC. Clock gaps and
folds are unresolved; an explicit timestamp offset disambiguates the instant.
A bare short decade (九〇年代), unanchored clock/relative phrase or vague modifier
is not guessed. Structured ISO spans are recognized in other language text;
multiple spans require a range grammar. The ticket's inverted expression
`two years after Google bought it in 2014` maps to purchase year 2012 and keeps
an inverted_same_sentence anchor. Normalizer grammar changes must increment
NORMALIZER_VERSION and update golden fixtures.

These primitives do not migrate existing stored facts themselves. TMP-3's storage
contract (`app.services.temporal_normalize`, PR #221) parses every stored and queried
date through this grammar: its `parse_bound()` maps one value's normalize() result to
a stored endpoint range (start of its start .. end of its end; an ongoing end gives
the start's range; a timestamp stays a point). Its NORMALIZER_VERSION embeds this one, so a grammar version bump makes the
TMP-3 backfill re-normalise stored facts. Original §5.2, Appendix B and
`temporal_patch_verify.py` have not been located; their 19-check compatibility
remains **pending**. The
included EN/ZH golden files are derived from the supplied backlog, not a copy
of the unavailable original set.

Run from module 25:

```bash
python -m pytest tests/unit/test_temporal_grammar.py -q
```

Temporal tests need no ArcadeDB, Redis, GPU, or model downloads. The current main separates database integration
fixtures under tests/integration; those fixtures are retained unchanged. Real
integration tests still require their PD_ARCADEDB_* credentials and services.
The grammar test is named test_temporal_grammar.py to avoid colliding with
TMP-3 PR #221's test_temporal_normalize.py.

Allen convention: shared start/end and equality precede endpoint touching.
A point at an interval's start `starts` it; at its end it `finishes` it.
`IntervalDomain` holds closed integer endpoint ranges; `extent="point"` adds
start=end, so uncertain points do not expand into spurious nonzero intervals.
None bounds are unbounded; unresolved evidence returns no feasible relation.
For an ongoing fact, confirmed_active_at additionally constrains start <=
confirmation <= end in both solver paths; it does not imply activity after
the confirmation or invent a finite end.
The fixed-window path intersects relation-specific bounds; the general solver
uses integer difference constraints and negative-cycle detection. Both use
bounded immutable-result caches. Infeasible domains return the empty set;
certainty(empty) raises ValueError.

The composition table enumerates all s<=e intervals on a six-point line, which
covers every weak ordering of the six endpoints in a triple. It contains all
169 immutable entries. For A=[0,1], B=[1,1], C=[1,2], the result is **meets**.
The full `finished-by` composed with `starts` entry is {meets, overlaps} because
proper (non-point) B can also yield overlaps; it cannot be reduced to meets.

Local verification (2026-10-07): 441 exact pairs, 19,208 feasible fact/window
cases (including correlated points), 9,025 feasible domain pairs, and additional
infeasible/unbounded cases agree with independent predicate enumeration.
Uncached p95: window 9.7 us, general 56.2 us on this host; these are observed
CPU measurements, not a portable guarantee. The unavailable Appendix B still
needs comparison against this documented convention.

Final regression on 2026-10-07: 67 normalizer tests and 7 Allen tests pass,
including all exhaustive case counts above. The latest uncached p95 observed
9.1 us/window and 49.6 us/general pair. Version 1.0.1 also rejects malformed
ISO prefixes and handles source leap days/month ends for relative periods.
Version 1.0.2 pads 1-5 digit second fractions before `datetime.fromisoformat`, so
`.25` parses on Python 3.10 (CI) exactly as on 3.11 (the service image).

## Grammar 1.1.0: the rest of §5.2

Calendar forms the plan lists, now read (precision in brackets):

| Form | Days | |
|---|---|---|
| `early / mid / late YYYY`, `mid-YYYY` | Jan–Apr / May–Aug / Sep–Dec | [part_year] — the plan's §5.1 example: "late 2021" = [2021-09-01, 2021-12-31] |
| `YYYY年初 / 年中 / 年底` | the same thirds | [part_year] |
| `YYYY年M月初 / 中 / 底`, `上旬 / 中旬 / 下旬` | days 1–10 / 11–20 / 21–end (旬) | [part_month] |
| `(the) summer of YYYY` | as `summer YYYY`: June–August | [season] |
| `the third quarter of YYYY`, `the 2nd quarter YYYY` | as `Q3 YYYY` | [quarter] |
| `the early / mid / late YYY0s` | years 0–3 / 4–6 / 7–9 | [decade] |
| `the 20th century`, `20世紀`, `二十世紀` | 1901–2000, as Module 18 reads it | [century] |
| `between X and Y` | as `from X to Y` | |
| `2010–11` (en/em dash) | 2010–2011 | [year]. `2010-11` with a hyphen stays ISO year-month (November 2010); `2011-13` is a calendar error, not a range |

**Anaphoric phrases need an anchor** (§5.3, P05). `normalize(text, anchor=(lo_ms, hi_ms))` — the
interval of the event the phrase refers back to — dates:

| Phrase | From the anchor |
|---|---|
| "two years later", "a year earlier", `3年後`, `三個月後` | shifted by calendar arithmetic |
| "the same day", `同一天`, `當天` | the anchor itself |
| "within a year" | [anchor start, anchor end + 1 year] |
| "less than six months earlier" / "… later" | [anchor start − 6 months, anchor start] / [anchor end, anchor end + …] |
| "July of that year", "in September" | that month of the anchor's year |
| "two years after <clause>" | anchor − 2 years: the anchor is the sentence's own explicit date ("sold in 2014, two years after Google bought it" → 2012) |

Without an anchor they are `UNRESOLVED` (`missing_anchor`) — **never** dated from `source_published`,
which stays the anchor of deictic phrases only: "two years ago", "yesterday", "last year", `去年`,
`三年前`, `昨天`. (Before 1.1.0, "two years later" and `3年後` were dated from the publish date.)
Month-end policy: a day the target month lacks is clamped to its last day (31 January + 1 month =
28/29 February). Vague or relation-only phrases are never dated: "about two years later" →
`approximate_relative`; "later", "afterwards", `後來` → `relation_only:after`; "at the time", `當時`
→ `relation_only:during` — a relation for reasoning (§5.5), not a date.

`NORMALIZER_VERSION` is 1.1.0, so the storage normaliser's version changes and the backfill
re-normalises rows written under 1.0.2 from their raw values.

**Grammar 1.1.1: spaced Chinese dates.** A space between a number and 年 / 月 / 日 / 號 / 季, or after
民國 / 第, is read as if absent: `2010 年 6 月` = `2010年6月` [month], `民國 113 年`, `2013 年第 3 季`. A space
between two numbers is never removed (`2010 2011` stays unresolved), and `raw` keeps the text as written.
`NORMALIZER_VERSION` is 1.1.1, so the backfill re-normalises stored rows from their raw values.

## Extraction (`app.temporal.extract`, temporal plan §5.6 / P11)

`extract(segments, llm=None)` turns a source's segments (chunk text, in document order) into
`(accepted, unresolved)` candidates. Pure: no I/O; the model is passed in as
`llm(messages, schema) -> dict`, so storage and the Module 07 client stay in `app.services`.

**Rules** run on every segment. The date-mention scanner and the clause-level
subject/verb/object builder are ported from the Temporal Demo (`find_dates`,
`extract_rules`): on the Android history document they give the same 39 facts as the demo.
The scanner only finds the date phrases and what a relative one refers back to — the
previous mention, which carries across segments, or, for "sold in 2014, two years after
Google bought it", the explicit date of its own sentence. **The days come from
`app.temporal.normalize`** (grammar 1.1.0), with that mention as its `anchor` — the
normaliser is the single source of truth for dates (§5.2). A phrase it does not date stays a
mention without days and its fact is unresolved with the normaliser's reason ("About two
years later" → `approximate_relative`); a relative phrase after it has no anchor
(`missing_anchor`). English only; Chinese dates come through the model and `normalize`.

**Dates are phrases at their own precision.** Each range is written as the
coarsest phrase that is exact — `2005`, `2008 to 2013`, `2005-07`, `2019-06 to 2019-08`,
`2015-08-08 to 2015-09-07` — and must normalise back to the same days, or the candidate
is unresolved. A range gives each endpoint its own period (`From 2008 to 2013` →
`2008`, `2013`); `since X` has no end. These are the `time_start` / `time_end` values the
writer stores, so a month is never padded to its 1st (and TMP-8's padded-period check,
which only looks at written days, never fires on them).

**The model** gets the segment and 300 characters of context as delimited data in the
user message only — never in the instruction part — and the schema has no date fields: only a
verbatim `date_text`, plus `extent` (`point` | `span`), its own `confidence` (0–1) and `links`.
Its output passes the evidence gate:

| Model output | Result |
|---|---|
| entity not verbatim in the segment, nor a name the document uses | dropped |
| value not verbatim in the segment (a paraphrase) | dropped |
| `date_text` verbatim in the segment | dated by the scanner's mention there, or by `normalize` of the text itself when that is narrower (`2020年5月`, where the scanner only saw `2020`) |
| `date_text` not verbatim (a quoted year with an invented month/day) | unresolved, `date_text_not_verbatim`; its date is never used |
| `date_text` that neither reads | unresolved, `date_text_not_normalised` |
| `date_text` that is a heading on its own line (`2000s` over a section) | unresolved, `date_text_is_a_heading`; a heading asserts nothing |
| no `date_text`, but `links` whose other event is in the segment | unresolved, `undated`, links kept (input for reasoning, §5.5) |

Every kept candidate carries the offsets of its evidence in the segment: `date_offsets`,
`entity_offsets` (None for a resolved coreference) and `value_offsets`. A rule value is the clause
with its date mention taken out ("in Palo Alto, California, [in October 2003] by Andy Rubin"), so its
offsets run from its first word to its last. `extent` comes from the model (anything but `point` /
`span` is dropped) or, for rules, `span` for a range or `since` mention and unknown otherwise — a
date alone does not say whether the event lasted. The model's `confidence` is kept as
`model_confidence` and only scales the heuristic confidence (× 0.5–1); it never replaces it.

**Merge** is conservative: a rule and a model candidate are one assertion only on the
same date evidence (overlapping offsets in the same segment) and the same entity. Same
days: one candidate, `origin=rules+llm`, confidence raised, the model's `extent` filling a rule's
unknown one. Different days: both kept,
both `conflict=True`. Shared words or the same entity on other evidence never merge.

A failed model call loses only that segment's model candidates (`on_error` is told); its
rule candidates stay. Confidence is the demo's heuristic (precision × how the subject was
found), not yet calibrated (§5.6), and separate from logical certainty (§5.4).

**References.** The extractor is scored against references written independently of it, from the
Temporal Demo's examples: `tests/unit/fixtures/android_history_reference.json`, the demo author's
hand-structured graph of the Android history text, and Project Aurora's notes with their graph. Only the
reference's `event` entries are scored — each is one dated statement; its `period` / `role` entries are
spans inferred across sentences. An event agrees when a candidate starts on its start day (and ends on
its end day, where one is given) and shares a word with its name.

Rules: 30 of the 31 Android events agree. The one the rules miss is "the arrival of Apple's 2007 iPhone
meant that Android had to go back to the drawing board"; the model adds it. (Three reference entries were
revised to the plan's conventions before scoring — "late 2021" = September–December (§5.1), summer =
June–August (the normaliser's), and no guessed month for "less than six months earlier" (§5.1, §5.2) —
see the reference's `note`.)

Project Aurora: all three dated statements agree (kickoff, pilot range, start of general availability).

Live, rules + `gemma4:26b` through Module 07 (`/inference/infer`, temperature 0), entity and value
required verbatim (2026-10-16), two runs: 31 of the 31 Android events agree in both — the model adds
the 2007 iPhone event (scored against the revised reference); 0 conflicts between rule and
model candidates; 36 confirmed by both, 16–17 model-only, 23–27 unresolved (mostly "later", "at the
time", "eventually" — kept with their links); every non-empty value has offsets. Model output varies
between runs even at temperature 0, so a fact only the model finds may be missed in one run.


## Durable temporal jobs (`app.services.temporal_jobs`, temporal plan §5.7 / P12)

Every indexed source gets its temporal facts from one job, whatever the medium. Nothing depends on
video moments or on a summary any more. The old video-only path in `graph_extractor` (its
per-moment temporal-fact LLM calls) is gone; video entity and relation extraction stays as it was.

**Flow.**

1. The Kafka consumer indexes the chunks.
2. It then records the job with `enqueue()`, before the offset commits. The job is a
   `TemporalJob` document plus one `TemporalJobSegment` per chunk, in document order. Each segment
   holds the evidence part of the chunk's text and where it starts, so the stored evidence offsets
   point into `Chunk.text`. What a model or the pipeline wrote around the source's own words is left
   out: for a Module 12 document chunk, everything but its `[CHUNK] ... [/CHUNK]` text (the file name
   and the LLM's `[CONTEXT]` sentence); for a video moment, the leading `contextual:{...}` (Module 11's
   LLM-written context), keeping what the recording says (ASR, OCR, visual description).
3. A background worker (`TemporalJobRunner`, `PD_TEMPORAL_JOB_WORKERS` of them, default 2) runs the job.
   Jobs run side by side while every model call takes its turn, so a short document is not stuck
   behind a long video; one job never runs twice at once. For each segment it does three
   things:
   - runs the language check;
   - makes one Module 07 call: serialised, retried with backoff on connection errors,
     502/503/504 and unreadable JSON. The reply is asked for as `CANDIDATE_SCHEMA` (`options.format`,
     P10): Module 07 constrains the model with it, validates the output and returns it as
     `result.parsed`. An output that misses the schema (`structured_output_error`) is not retried.
     A model that cannot take a schema, or an older Module 07, is read the old way: tolerantly, with
     `strict=False`. Either way the reply then goes through the evidence gate, because a valid shape
     says nothing about whether the content is true;
   - stores the model's reply on the segment row.
4. `extract(segments, replies=...)` then rebuilds the candidates deterministically from the stored
   replies.
5. The accepted facts are upserted with their evidence:
   - `date_text` and `source_sentence`;
   - `evidence_offsets` (JSON offsets of date, entity and value);
   - `extent`;
   - `anchor`, which records how the date was read;
   - `extractor_version`;
   - `media_start_ms` / `media_end_ms` for recordings.
6. Every named subject gets an `Entity`, with `MENTIONS` from its chunk and `HAS_TEMPORAL_FACT` to its
   facts. Pronouns, resolved coreferences and descriptions ("the company") get none, and an Entity that
   another source already made keeps its name and type. Without this, documents would have no entity
   graph at all, and the TKG search could not find "What was Hugo Barra doing in 2010?" by entity.
7. The candidates it could not date ("later", "at the time", "after the acquisition") are kept as
   `TemporalCandidate` rows: evidence, `reason`, and the qualitative `links` reasoning (P15) will use.
   They are a document type of their own, so no TKG or temporal query returns an undated row as a fact.
8. Every other fact of the source is removed. The job owns the source's temporal facts.

**Identity.**
- **Job key:** sha256 of tenant, source version, `EXTRACTOR_VERSION` and `NORMALIZER_VERSION`.
  A new extractor or grammar gives a new job.
- **`fact_id`:** follows §5.6. It is a hash of:
  - source version and chunk;
  - evidence offsets;
  - entity, relation and value;
  - timeline and extent;
  - both endpoint states and ranges;
  - precision and normaliser version.

**Durability.** The job key is independent of the embedding dedup, so a redelivered message
whose chunks were already indexed still finds its job. After a restart, `start()` re-queues
pending and running jobs. Only segments that are not `done` call the model again, and fact writes
are idempotent upserts. A finished job with failed segments is reopened when its source is
enqueued again, until a segment has failed `PD_TEMPORAL_SEGMENT_MAX_ATTEMPTS` runs.

**Generation checks.** The source must still be live, meaning an active Chunk or Source for the
version. This is checked twice: before any model call, and again before any fact is written.
Archiving or deleting the source in between ends the job as `cancelled`, with no facts.
Delete-by-version removes the job and its segments.

**Status** (`GET /personal/temporal/status/{version_id}`, header `X-Branch-ID`) returns:
- the job state: `pending`, `running`, `complete`, `partial`, `failed` or `cancelled`;
- `reason`;
- the segment counts: total, done, failed and unsupported;
- the facts written and the unresolved candidates;
- the extractor and normaliser versions;
- the model;
- the segments that failed or could not be read.

The same status is public as gateway `GET /api/0.4/temporal/status/{version_id}` (scoped to the
caller's JWT; a Keycloak resource `/api/0.4/temporal/*` alongside `identity/*`) and as the MCP tool
`raptor_temporal_status(source_version_id)`.

A segment whose letters are mostly neither Latin nor Han is `unsupported` (`unsupported_language`)
and is never extracted. A job made only of such segments is `failed`, never "complete with zero
facts". A job with some of them is `partial`.

**Not yet.**
- **Reasoning (P15).** Ingestion-time reasoning is not wired in yet, so `complete` means extraction
  is done.
- **Per-span language coverage (P21).** The language check is per segment, not per span.

## Capture / publish time as the anchor of deictic phrases (`app.temporal.anchor`, P13)

"Two years ago", "yesterday", "last month" and 去年 mean nothing without the moment the source was written or
recorded. That moment comes from the file's own metadata, read by the worker that opened the file. It arrives
on every chunk as `capture_time` / `capture_time_source`, and is kept on `Chunk` and `Source` as read:

| `capture_time_source` | Read by | From | Meaning |
|---|---|---|---|
| `pdf_creation_date` | Module 12 analysis | PDF `/CreationDate` | publish / authoring date |
| `docx_core_created` | Module 12 analysis | DOCX / PPTX `docProps/core.xml` `dcterms:created` | publish / authoring date |
| `exif_datetime_original` | Module 10 orchestrator | EXIF `DateTimeOriginal` (+ `OffsetTimeOriginal`) | capture time |
| `container_creation_time` | Modules 11 / 09 orchestrators (ffprobe) | QuickTime `com.apple.quicktime.creationdate`, else container `creation_time`, else WAV BWF date + time | capture time |

An ID3 `date` (MP3) is never used: it is a release date, not when the recording was made. The readers
never raise; a file without a readable date is processed exactly as before.

**Validation** (`anchor.validate`). The value must parse (PDF `D:…`, EXIF `YYYY:MM:DD HH:MM:SS`, ISO 8601;
naive values are read in `PD_TEMPORAL_DEFAULT_TIMEZONE`). It must not be a known placeholder: the Unix,
QuickTime, DOS/ZIP or Windows epoch. It must lie between 1900 and tomorrow. Otherwise the source has no
anchor, and the job records why (`anchor_reason`). The upload time is never used instead (§9).

**Use.** The temporal job records the validated instant (`anchor_time`, `anchor_source`) and passes its
calendar day to the extractor:

- **Deictic date texts.** A model candidate's date text such as "two years ago" is normalised with it,
  and the fact is stored with `anchor = deictic`.
- **Confidence.** Deictic facts get confidence × 0.8, because file metadata can be an export date rather
  than the writing date.
- **No anchor.** Without an anchor, the same phrases stay unresolved, as before.
- **Unaffected.** Explicit dates and anaphoric phrases ("two years later", dated from the previous event)
  do not use the anchor.
- **Version.** `EXTRACTOR_VERSION` is 1.1.1. 1.1.1: a blank line ends a sentence, so a heading
  (the title, `2000s`) is never read as part of the sentence below it.

**A photo is itself an observation event.** When an image carries a validated EXIF time, the consumer
writes one TemporalFact for it, even when it has no OCR text and so no chunk and no job:
- entity: the file name; relation: "was taken"; value: the image's caption;
- dated to the instant (precision `second`), with `anchor = capture` and the raw EXIF value as `date_text`.

The temporal job never deletes it as one of its own stale facts; delete-by-version removes it with the photo.

**Recordings.** Video moments and audio chunks carry the recording's capture time, which anchors deictic
phrases in what is said or shown ("yesterday", "last week"). Historical claims in a recording keep their
own dates, and `media_start_ms` / `media_end_ms` say where in the recording each was said.

Not yet: mapping a recording's own segments to world time (`capture_time + offset`). This is valid only
for a recording declared continuous, and nothing declares that yet. The same goes for per-recording
`timeline_id`s on a media axis, which only those observation events would use.

**Caveats.**

- **Quoted text.** A deictic phrase inside quoted text ("Earlier this week, …" quoting a 2015 article)
  refers to the quoted source's time, not the document's. "This week" / "now" / "currently" are not in
  the grammar, so they stay unresolved rather than being dated wrongly.
- **Export dates.** A PDF exported from another format carries its export date, e.g.
  `Android_History.pdf`: 2026-09-21.

## Source lifecycle: generations, archive, clone, re-extraction (P14)

- **Generation.** `Source.generation` advances on every archive and reactivation. A temporal job records
  the generation it was started for. Before any model call, and again before any write, it checks that
  the source is active *and* still at that generation. So work started before an archive never writes
  after a reactivation.
- **Stale runs.** If a job was reopened at a newer generation while an old run of it was busy, the old
  run ends as `superseded`: it writes nothing and leaves the job to the rerun.
- **Archive / reactivate** (`set_status_by_version`). The source's undated candidates take the same
  status as its facts. Archiving cancels the source's pending or running jobs (`source_archived`).
  Reactivating reopens them at the new generation and hands them to the worker.
- **Delete** removes the source's jobs, segments, candidates and facts, plus any entity left without
  content. Another source's assertions stay.
- **Clone** (`clone_version`) copies the job with its segments (model replies included) and the undated
  candidates onto the target version, with chunk ids remapped onto the cloned chunks. The target's status
  answers at once, and the model is never called again for it. A job still pending or running is cloned
  as pending and handed to the worker.
- **Re-extraction clears obsolete fields.** A fact written by an extractor (`extractor_version` set)
  writes every evidence field, NULL included. A re-extraction that no longer finds an extent, offset or
  media position does not leave the old value behind. Other writers (Kafka entries, HTTP) still set only
  what they send.
- **Assertion identity** (§5.6) now includes the `axis` too.

Not yet: invalidating derived results when their supporting assertions are retired. There are no
derived results before reasoning (P15).

## TKG contract v2: ranking, pagination, capabilities (`app.temporal.contract`, P08 / P09)

A `/personal/search/tkg` request with `"version": 2` follows the plan's contract (§5.9). Without `version`, a
request is v1 and behaves exactly as before: Module 21 and older clients are unaffected.

- **Strict.** Fields the request model does not know are refused with 400 `UNKNOWN_FIELDS`. Fields the
  contract defines but this deployment does not serve are refused with `UNSUPPORTED_FEATURES`: `as_of`
  and a `timeline_id` other than `world` (`reference` is served since P16, `source_filter` since P21). Other refusals
  are `CONFLICTING_FIELDS` (`window` together with `time_start` / `time_end`) and `INVALID_TIMEZONE`. Each
  is a structured `detail` that names the fields; nothing is silently ignored.
- **Window and zone.** `window: {start, end}`, where either side may be omitted, is the v2 form of
  `time_start` / `time_end`. `timezone` sets the zone calendar dates are read in, for this request only.
- **Whole-result ranking.** Every match is collected first: the entity's own facts (no longer uncapped),
  the facts that mention it, or the keyword / temporal candidates. All of them are then sorted:
  - `sort: time` (the default): by start, undated last, then `fact_id`;
  - `sort: relevance`: every match is scored. Each query word counts 3 when it is in the subject, 2 in the
    relation or value, and 1 in the source sentence or evidence; the scores are averaged over the words.
    Ties are broken by time. Chinese is matched by two-character pieces.

  The first `limit` temporally matching facts are never taken as "the most relevant".
- **Keyset pagination.** `cursor` encodes the last returned fact's sort key and a fingerprint of the
  request (excluding `cursor` and `limit`). The next page is everything after that key, so facts written
  between pages are neither skipped nor repeated. A cursor sent with a different query is refused
  (`CURSOR_FOR_ANOTHER_QUERY`). The response carries `version: 2`, `sort`, `cursor` (null on the last
  page), `total` and `complete`; `complete` is true only on the last page.
- **Evidence travels.** Every TKG fact now carries `date_text`, `source_sentence`, `extent` and `anchor`
  (v1 too; the fields are additive).

`GET /personal/temporal/capabilities` (gateway `GET /api/{v}/temporal/capabilities`, MCP
`raptor_temporal_capabilities`) returns:
- the contract versions;
- the TKG match modes, Allen relations, sorts, pagination, timelines and zone;
- the unsupported fields;
- the temporal operations, served or null;
- the extraction pipeline's media types, model and versions, and the file dates deictic phrases are
  anchored on (P13).

A client checks it before relying on a feature. The MCP `raptor_tkg_query` asks for v2 when it is given
`sort` or `cursor`, and reports `complete`, `total` and `next_cursor`.

Not yet:
- an execution budget: every candidate in the window is still read before ranking;
- reranker-based relevance;
- the `as_of` feature itself.

## Reasoning at ingestion (`app.temporal.stp`, `app.temporal.reasoning`, P15)

The last step of a source's temporal job (§5.5). The job's status goes `running` → `reasoning` →
`complete`, so `complete` now means extraction *and* reasoning.

**Network.**
- **Nodes.** The source's facts (endpoint ranges) and its undated candidates.
- **Links.** The qualitative `links` the extractor found, such as "joined later, after Android Inc. was
  founded". Facts now keep theirs too (`TemporalFact.links`).
- **Resolving a link.** A link's `other` phrase is resolved to one assertion of the same source, by the
  words it shares with the subject, relation and value. Nodes whose sentence contains the phrase are
  preferred. A sibling from the same sentence ("Nick Sears joined") is not taken for the event the phrase
  names ("Android Inc. was founded").
- **Unresolved links.** No match, a weak match, or several matches with different dates leave the link
  unresolved. It is reported (`links_unresolved`), never guessed.

**Solver (STP).** Each link is exactly a conjunction of endpoint inequalities, in integer UTC ms:

| Link | Constraint |
|---|---|
| `before` | `eA < sB` |
| `after` | `sA > eB` |
| `during` | `sB ≤ sA`, `eA ≤ eB` |
| `same_time` | `sA ≤ eB`, `sB ≤ eA` |

Strict order is 1 ms, as in the plan's worked example.
- **Bounds.** Floyd–Warshall gives the tightest bounds the network entails.
- **Inconsistency.** A negative cycle proves it. It is returned as a witness, which is a conflict but not
  necessarily a minimal one: the constraints along the cycle, each with what states it.
- **Consistency.** The network is fully represented by the STP, so a feasible STP is a consistent network.
  Path consistency over the Allen algebra would prove nothing more here, so it is not run.

**Stored.**
- **Reasoning record.** A `TemporalReasoning` record per source holds:
  - `status`: `consistent`, `inconsistent`, `no_links` or `failed`;
  - `network_version`;
  - `witness`, `links_used` and `links_unresolved`.
- **Derived facts.** For every undated candidate that the links tie to dated facts, a derived
  `TemporalFact` is written:
  - flags: `derived = true`, `anchor = derived`, `precision = derived`;
  - `support_ids`: the dated facts the links reach, and the undated ones on the way;
  - `network_version`;
  - `date_text`: e.g. "after 2003-10-01 (derived)";
  - confidence: the candidate's × 0.8.
- **Half-bounded sides.** A side bounded on only one end is stored as `unknown`. This is sound, as a
  superset, and never a sentinel. TKG search returns derived facts with `derived` and `support_ids`, and
  their `when` is that labelled text. They are never presented as a stated date.
- **Backfill.** Derived facts are not re-normalised by the backfill, because they have no date string;
  a job rerun recomputes them.

**Operations.**
- `GET /personal/temporal/reason/{version_id}`: the stored result. It is `stale` while the job runs.
- `POST /personal/temporal/reason/{version_id}/recompute`: reopens the job and returns at once. Its segments
  stay done, so there is no model call.
- Both are also on the gateway (`/api/0.4/temporal/reason/...`) and in MCP (`raptor_temporal_reason`).

**Lifecycle (P14).**
- **Archive, reactivate, delete.** These treat derived facts and the reasoning record like the rest of
  the source's data.
- **Re-extraction.** It replaces the derived facts.
- **Clone.** It copies the derived facts as facts of the target.

**Not yet.**
- **Cross-source scope.** The same entity in other sources is not reasoned over.
- **Query-time correlated classification** (option A, Appendix D F3), with P16.
- **Tightening dated facts.** Facts that already have dates are not tightened by the links.
- **The `undetermined` status.** It would come from a bounded search over disjunctive links, and every
  link type here is conjunctive.

## Reference-event queries (`app.temporal.reference`, P16)

"What happened after Google acquired Android?" A TKG v2 request names the event the answer is relative
to, and the Allen relations a fact must have to it:

```json
{"version": 2, "query": "what happened", "reference": {"text": "Google acquired Android"},
 "reference_relations": ["after"], "certainty": "possible"}
```

`reference` is `{"fact_id": ...}` or `{"text": ...}`, and must come with a non-empty
`reference_relations`. Otherwise the request is a 400: `INCOMPLETE_REFERENCE`, `INVALID_REFERENCE` or
`INVALID_REFERENCE_RELATIONS`. There is no default meaning of "relative to X".

**Resolving the event.**
- **By `fact_id`:** that fact, if it is active.
- **By `text`:** every active fact is scored by the reference's content words:
  - a word in the fact's subject counts 3, in its relation or value 2, and only in its sentence 1;
  - inflections match on a shared stem, so "acquisition" finds "acquired"; Chinese is matched in
    two-character pieces.
  - each word is weighted by how rare it is in the tenant's facts: in a history of Android, "acquired"
    names the acquisition, while "Google" and "Android" are in almost every fact;
  - facts derived by reasoning (P15) are candidates only when no stated fact is: an inference is not the
    event the question names.
- **Supported facts:** a fact must have at least 60% of the words to be a candidate.
- **Outcome**, taken from the best-scoring candidates:
  - `resolved`: one event. Several rows with the same dates are one event stated more than once.
  - `ambiguous`: events with different dates score equally well ("the acquisition" with both Google /
    Android and Facebook / Instagram in the data). The `candidates` are returned, and the top one is
    never silently taken.
  - `not_found`: nothing supports the reference.
  - `undated`: the event is in the data but has no known date, so nothing can be placed relative to it.

Unless the status is `resolved`, the response has no facts, and its `reference` block says why.

**Relating facts to it.**
- **Which facts.** Every retrieved fact other than the event's own rows is checked. Retrieval is by
  entity, keywords or temporal-only, as before.
- **How.** `allen.possible_relations` is applied over both facts' endpoint ranges, so a date known only
  to the month or year is honoured on both sides. A fact in July 2005 *may* be after an acquisition in
  July 2005, so it is `possible`, not certain. One in August is certainly after it.
- **What is kept.** `certainty: possible` keeps a fact that can have one of the requested relations;
  `certain` keeps one whose every possible relation is requested.
- **Strict `after`.** It is the Allen `after`: an event that meets or shares the reference's end is not
  after it. "Immediately after" is `met-by`.
- **With a window.** A window still applies as well.
- **Labels.** Each fact carries `allen.reference: {relations, certainty}`.

**Across modules.**
- **Module 18** reads the reference from the question. A relation word followed by an event, not a date,
  gives `reference` / `reference_relations`:
  - "after Google acquired Android" → `["after"]`; "before" → `["before"]`;
  - "during" / "while" → starts, during, finishes, equals;
  - Chinese 之後 / 以後, 之前 / 以前 and 期間 likewise.

  "after 2011" stays a window. The reference phrase's words are not added to `keywords`, because the
  answer's facts need not mention them.
- **Module 21** sends the reference as contract v2. A resolved event leads the triples. The summary marks
  `possible` facts as "[possibly]". An ambiguous, missing or undated reference is explained instead of
  summarised.
- **Gateway / MCP.** The gateway passes `reference` through. MCP `raptor_tkg_query` takes `reference` and
  `reference_relations`, and returns the `reference` block.

**Not yet.**
- **Reranker scoring.** Reference text is not scored with the reranker: resolution is lexical, and
  deterministic.
- **Scan cost.** Resolving a text reference reads every active fact of the tenant.
- **Choosing a candidate.** An ambiguous reference cannot yet be followed up by picking one of its
  candidates in the conversation. A client can resend it with that candidate's `fact_id`.

## Grounded answers: verifying temporal claims (`app.temporal.claims`, P17)

An answer's temporal statements are verified before anyone reads them (§5.12). The answer stage
(Module 21's TKG agent, `agents/grounded_claims.py`) works in four steps.

1. **Evidence.** The facts TKG returned are ranked against the question by the cross-encoder
   (`/personal/search/rerank`). The top `TKG_EVIDENCE_BUDGET` (default 20) are kept, not the first N in
   time order.
2. **Claims.** Module 07 returns claims, not prose, through P10 structured output (`CLAIMS_SCHEMA`):
   - `{"kind": "date", "fact", "date", "endpoint": "whole" | "start" | "end"}`;
   - `{"kind": "ordering", "subject", "object", "relations": [...], "wording": "certain" | "possible"}`;
   - `{"kind": "absence"}`.
3. **Verification.** `POST /personal/temporal/verify {claims, evidence, complete, timezone?}` checks
   every claim deterministically:
   - **Scope.** Every cited ID is in `evidence` and is an active fact of the tenant. A passage, an
     archived fact or an invented ID establishes nothing.
   - **Date.** The cited fact's range (whole span, or one endpoint) must lie inside the claimed period,
     which is read by the grammar (`2005`, `July 2005`, `2005年7月` …). A coarser claim is entailed: "2005"
     by "2005-07". A finer one is not: "12 July 2005". A 2008–2013 role is not "in 2010", though its
     start is "2008".
   - **Ordering.** `allen.possible_relations` of the two facts must be non-empty.
     - `certain` wording needs all of those relations to be among the claimed ones, so
       {before, equals, after} never becomes "X happened before Y".
     - `possible` wording needs one of them. Each relation in the set is individually feasible, so the
       set is a witness.
   - **Absence.** Rejected unless the retrieval was `complete`.

   The response holds the `verdicts` (each with `verified`, `reason` and what was found), `all_verified`,
   and the cited facts with their `when`.
4. **Rendering.** If every claim passes, a template writes them out from the server's facts and
   relations, never from the model's wording. If any claim fails, the answer is the cited facts as
   recorded, with each claim's verification status. If the model or the verifier is unavailable, the
   answer is the evidence facts as recorded. It is never unverified prose.

The answer's other statements (from passages) are not claims here. The final answer prompt takes dates
and orderings only from these verified statements. `TKG_GROUNDED_ANSWERS=false` brings back the old
free-text TKG summary.

Not yet:
- claims about passages: a passage citation establishes no date or ordering, by design;
- answering in the question's language by template: the rendering is English.

## Ingest: inline text and uploaded assets, with 7-day expiry (`app.services.source_expiry`, P21)

Source material reaches the temporal pipeline without the client converting anything or supplying
facts (§5.7). The gateway's `POST /api/{v}/temporal/ingest` (MCP `raptor_temporal_ingest`) accepts
exactly one of two forms.

- **Inline content.** `content` with `content_type` `text/plain` or `text/markdown`: notes, prose, a
  pasted article, up to 1 MB.
  - It becomes a `.txt` asset of the caller (Module 04) and goes through the document worker's text
    path, so evidence, status, retry and deletion work exactly as for an upload.
  - The asset's archive and destroy dates are 7 days out, and the gateway registers the same
    `expires_at` here (`PUT /personal/temporal/expiry/{version_id}`) before the source is even indexed.
- **An uploaded asset.** `asset: {asset_path, version_id}` for PDF, DOCX, PPTX, TXT, images, audio or
  video. It is sent for analysis if it has not been (Module 04's register guard). Uploads keep their own
  expiry.
- **Refusals.**
  - Other types are refused with `415 UNSUPPORTED_CONTENT_TYPE` and never processed as plain text: HTML,
    email, chat exports and CSV / JSON / logs come after v1.0.
  - An `idempotency_key` replays the first result; the same key with a different body is `409
    IDEMPOTENCY_KEY_REUSED`.
  - Identical text whose source has expired but is not yet deleted is `409 SOURCE_EXPIRED`.
- **Response.** `source_version_id` (also the `job_id` for `temporal/status`), `status`, `expires_at`,
  `duplicate`. Capabilities list the live inline types and asset extensions under `ingest`.

**Expiry.** Each inline source has a `SourceExpiry` record: `{version_id, branch_id, expires_at,
state}`, with `state` going pending → expired → deleted.
- **Logical expiry.** Every `PD_TEMPORAL_EXPIRY_SWEEP_SECONDS` (default 60), a sweeper archives the
  sources whose `expires_at` has passed. It uses the archive path Module 04 uses (P14): Chunks, Source and
  facts become archived, so every query leaves them out, and pending jobs are cancelled.
- **Late content.** Content indexed after the expiry, from a late worker, is archived by the next pass.
- **No resurrection.** Reactivating an expired source is refused, and registering it again does not lift
  the expiry.
- **Physical deletion.** Module 04's destroy deletes the source (`DELETE /personal/index/{version_id}`)
  and marks the record `deleted`.

**Operations and scope.** `operation` dispatches the request, as in the plan's P21 handler:
- **`ingest`** is the default, as above.
- **`query`** asks a `question` over the caller's existing sources, optionally limited to
  `source_version_ids`. Nothing is stored; the question is never a source.
- **`ingest_and_query`** ingests, waits up to `wait_seconds` (default 60, max 300) for the temporal job,
  then answers `question` from that source only.
  - Job still running at the deadline: `answer_status: pending` and no answer. The client polls the
    status, then asks with `query` and the source. An answer made without the source's facts would be a
    guess.
  - Job failed or cancelled: `not_answered`.
  - The idempotency key covers the ingest only, so a retry waits and answers again.

**How a scoped answer is built.** It uses Module 21 `POST /query` (mode direct) with
`source_version_ids`:
- vector and keyword search filter Chunks by `version_id IN (...)`;
- TKG gets contract v2 `source_filter`, which limits both the facts and the resolution of a `reference`;
- GraphRAG (a cross-source entity graph) and the user's memory are left out.

`source_filter` is a list of 1 to 100 version ids; anything else is `INVALID_SOURCE_FILTER`.

Not yet:
- **After-v1.0 adapters.** HTML, email, chat exports and CSV / JSON / logs.
- **Exactness of logical expiry.** It is exact to one sweep interval (default 60 s), not to the
  millisecond.

## Parse: the time a question asks about (`app.temporal.query`, TMP-7)

`POST /personal/temporal/parse {text, timezone?, anchor?}` reads a question's time condition. Every date phrase
in it is read by `app.temporal.normalize`, the grammar extraction uses, so the normaliser lives in one place.
What `parse` adds is the query side: which words are a date phrase, the operator around it, the event a
question is relative to, and the keywords a TKG search filters the evidence by. No database is read.

| `status` | Meaning |
|---|---|
| `RESOLVED` | a `window` and its `match`, and/or a `reference` with its `reference_relations` |
| `NO_TEMPORAL` | nothing in the text constrains time |
| `UNRESOLVED` | a date phrase the grammar cannot read, with normalize's `reason` |

| Question | `window` | `match` |
|---|---|---|
| "after 2011", 「2011年以後」 | [2012-01-01T00:00:00.000, unbounded) | `inside` |
| "before 2013", 「2013年之前」 | (unbounded, 2012-12-31T23:59:59.999] | `inside` |
| "since 2010", 「2010年以來」 / 「2010年起」 | [2010-01-01T00:00:00.000, unbounded) | `inside` |
| "until 2012", 「到2012年為止」 | (unbounded, 2012-12-31T23:59:59.999] | `inside` |
| "in 2010", "between 2010 and 2016", 「2010年到2016年」, "June 2010", 「2010 年 6 月」 | the whole period | `overlaps` |
| "within 2010" | the whole period | `inside` |
| "2011-13", "after 2011-02-30" | — | `UNRESOLVED` (`invalid_calendar`) |
| "yesterday", 「去年」 without `anchor` | — | `UNRESOLVED` (`missing_anchor`) |

- **Bounds.** `window.start` / `window.end` are instants with their offset in `timezone` (default: the
  deployment's); `start_ms` / `end_ms` are UTC milliseconds. A null side is unbounded.
- **Strict operators.** "after" excludes the period (`{after}`, §5.9), "since" includes it; "before" and
  "until" alike.
- **A year is a year.** A year inside a longer number or a name ("iPhone 20110", "Galaxy S2010") is not read.
  An ISO date is read whole, so a calendar error is never reduced to its year.
- **Deictic phrases** ("last year", 「去年」, "two years ago") are read from `anchor` (ISO 8601 with an offset:
  the question's own time), never guessed without it.
- `phrase`, `operator`, `precision`, `keywords`, `timezone` and `grammar_version` come back too.

**Module 18** reads every query through `parse` once Module 25's capabilities show `temporal_v2`
(`PD_TEMPORAL_V2`) on and a `parse` route; until then its interim regex (TMP-2) stays. Once in use, a parse
that times out (`PARSE_TIMEOUT_MS`), fails or returns `UNRESOLVED` makes the plan
`{temporal_status: UNRESOLVED, temporal_reason, temporal_constraint_applied: false, answer: null}` with no
window, reference or keywords. Module 21 then retrieves nothing and answers nothing, so unfiltered facts are
never presented as the answer to a temporal question (Appendix D F2). The gateway serves it as
`POST /api/{version}/temporal/parse`, the MCP server as `raptor_temporal_parse`.

**Latency.** Measured on the tmp34 test deployment (300 calls, mixed EN/ZH questions): p50 0.7 ms, p95 3.5 ms,
p99 4.1 ms. `PARSE_TIMEOUT_MS` defaults to 1000 ms -- far above that, because the host Module 25 shares is near
CPU saturation and a timeout makes the question unanswerable, not slower.

Not yet:
- **LLM fallback** for a temporal cue the grammar cannot read ("the year after the pandemic started", §5.11).
- **Media timelines** ("12 minutes into the video").
