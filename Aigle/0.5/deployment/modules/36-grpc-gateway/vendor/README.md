# vendor/

`app/services/video_search_service.py` is a copy from Raptor 0.5's module 13
(logic unchanged; only mechanical `ruff --fix` rewrites such as `Optional[X]` -> `X | None`). Module 13 on `main` no longer ships it (the REST search path was
cut over to module 25 directly), but SearchByPerson / StreamSearchByPerson
need it. The Dockerfile copies it into `app/services/` next to module 13's
own `personal_db_service.py` / `search_service.py`.

Remove this directory once module 13 owns an equivalent shared service.

Local change to the 0.5 logic: `VideoSearchService.search` re-raises when all four
retrievers (BM25, vector, GraphRAG, TKG) fail, instead of returning an empty result;
`services/search_service.py` maps that to `UNAVAILABLE`. One to three failing
retrievers are still tolerated, as in 0.5.
