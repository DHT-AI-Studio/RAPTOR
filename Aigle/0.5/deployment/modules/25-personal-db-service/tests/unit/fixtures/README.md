# Test fixtures

| File | Used by | Source / license |
|---|---|---|
| `temporal_en.json`, `temporal_zh.json` | `test_temporal_grammar.py` (TMP-5 golden sets) | written for these tests |
| `android_history_segments.json` | `test_temporal_extract.py` | Wikipedia, "Android (operating system)", history section — CC BY-SA 4.0. Read from the Temporal Demo's `Android_History.pdf` and cut into ~700-character chunks, as a document worker hands them to Module 25 |
| `android_history_reference.json` | `test_temporal_extract.py` | The Temporal Demo's hand-structured knowledge graph of the same text (`examples/android_history.json`): the reference the extractor is scored against. Three entries were revised to the temporal plan's conventions (its `note` says which): "late 2021", "the summer of 2019", and a month the text never states. Its `event` entries are single dated statements; its `period` / `role` entries are spans the author inferred across sentences and are not scored |
| `project_aurora_notes.txt`, `project_aurora_reference.json` | `test_temporal_extract.py` | The Temporal Demo's `examples/project_aurora*`: a short text and the knowledge graph written for it — a second document the extractor was not developed on |
