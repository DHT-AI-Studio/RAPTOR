from __future__ import annotations

from mcp import types
from mcp.server.fastmcp import FastMCP

PROMPT_DEFINITIONS = [
    types.Prompt(
        name="raptor_search_and_summarise",
        description="Search the Raptor knowledge base and return a concise summary with sources.",
        arguments=[
            types.PromptArgument(name="topic", description="Topic or question to search for", required=True),
            types.PromptArgument(name="top_k", description="Number of results to retrieve (default 10)", required=False),
        ],
    ),
    types.Prompt(
        name="raptor_video_analysis",
        description="Find and analyse video clips on a topic, listing relevant timestamps and key points.",
        arguments=[
            types.PromptArgument(name="topic", description="Topic to search in videos", required=True),
            types.PromptArgument(name="top_k", description="Number of videos to retrieve (default 5)", required=False),
        ],
    ),
    types.Prompt(
        name="raptor_document_qa",
        description="Document-grounded Q&A: search documents and answer using only retrieved content.",
        arguments=[
            types.PromptArgument(name="question", description="Question to answer from documents", required=True),
        ],
    ),
    types.Prompt(
        name="raptor_temporal_query",
        description="Time-range knowledge graph query: find events and relationships within a date window.",
        arguments=[
            types.PromptArgument(name="entity", description="Event or entity to query", required=True),
            types.PromptArgument(name="start_date", description="Start date (ISO 8601, e.g. 2025-01-01)", required=False),
            types.PromptArgument(name="end_date", description="End date (ISO 8601, e.g. 2026-12-31)", required=False),
        ],
    ),
    types.Prompt(
        name="raptor_quick_answer",
        description="Ask any question and get an answer from the knowledge base.",
        arguments=[
            types.PromptArgument(name="question", description="What do you want to know?", required=True),
        ],
    ),
    types.Prompt(
        name="raptor_explore_topic",
        description="Get a comprehensive overview of a topic from all available content (videos, documents, audio, images).",
        arguments=[
            types.PromptArgument(name="topic", description="Topic you want to explore", required=True),
        ],
    ),
    types.Prompt(
        name="raptor_find_in_video",
        description="Find specific moments in videos — returns timestamps you can jump to directly.",
        arguments=[
            types.PromptArgument(name="what", description="What are you looking for in the videos?", required=True),
        ],
    ),
    types.Prompt(
        name="raptor_upload_workflow",
        description="[Dev] Full upload workflow: upload a file, poll processing status until complete, then confirm it is searchable.",
        arguments=[
            types.PromptArgument(name="filename", description="Original filename with extension, e.g. 'interview.mp4'", required=True),
            types.PromptArgument(name="content_base64", description="Base64-encoded file content", required=True),
            types.PromptArgument(name="content_type", description="MIME type, e.g. 'video/mp4', 'application/pdf'", required=True),
        ],
    ),
    types.Prompt(
        name="raptor_search_strategy",
        description="[Dev] Choose and compare search modes (hybrid / BM25 / vector / video) for a given query.",
        arguments=[
            types.PromptArgument(name="query", description="Search query text", required=True),
            types.PromptArgument(name="media_type", description="Target media type: 'videos', 'documents', 'audios', 'images', or omit for all", required=False),
        ],
    ),
    types.Prompt(
        name="raptor_rag_pipeline",
        description="[Dev] Run a RAG query using raptor_a2a_direct (fast, no agent loop) or raptor_a2a_agent (multi-step reasoning).",
        arguments=[
            types.PromptArgument(name="question", description="Question to answer via RAG", required=True),
            types.PromptArgument(name="mode", description="'direct' for single-pass retrieval+generation, 'agent' for multi-step reasoning (default: direct)", required=False),
            types.PromptArgument(name="top_k", description="Number of chunks to retrieve (default 5)", required=False),
        ],
    ),
]


def get_prompt_messages(name: str, arguments: dict) -> list[types.PromptMessage]:
    if name == "raptor_search_and_summarise":
        topic = arguments.get("topic", "")
        top_k = arguments.get("top_k", "10")
        return [types.PromptMessage(
            role="user",
            content=types.TextContent(type="text", text=(
                f"Please search the Raptor knowledge base for: **{topic}**\n\n"
                f"1. Use `raptor_search` with query=\"{topic}\" and top_k={top_k}.\n"
                f"2. Review the results and provide a concise summary.\n"
                f"3. List the source filenames at the end.\n"
                f"Keep the summary factual and grounded in the retrieved content only."
            )),
        )]

    if name == "raptor_video_analysis":
        topic = arguments.get("topic", "")
        top_k = arguments.get("top_k", "5")
        return [types.PromptMessage(
            role="user",
            content=types.TextContent(type="text", text=(
                f"Find and analyse videos about: **{topic}**\n\n"
                f"1. Use `raptor_video_search` with query=\"{topic}\" and top_k={top_k}.\n"
                f"2. For each video, list the most relevant timestamp segments.\n"
                f"3. Summarise the key points covered in each video.\n"
                f"4. Highlight any differences or complementary information between videos."
            )),
        )]

    if name == "raptor_document_qa":
        question = arguments.get("question", "")
        return [types.PromptMessage(
            role="user",
            content=types.TextContent(type="text", text=(
                f"Answer the following question using only content from the Raptor document knowledge base:\n\n"
                f"**Question:** {question}\n\n"
                f"Steps:\n"
                f"1. Use `raptor_search` with type=\"documents\" to find relevant document chunks.\n"
                f"2. Answer the question strictly based on retrieved content.\n"
                f"3. If the answer is not found in the documents, say so explicitly.\n"
                f"4. Cite the source filename for each claim."
            )),
        )]

    if name == "raptor_temporal_query":
        entity = arguments.get("entity", "")
        start_date = arguments.get("start_date", "")
        end_date = arguments.get("end_date", "")
        time_range = f" between {start_date} and {end_date}" if start_date or end_date else ""
        return [types.PromptMessage(
            role="user",
            content=types.TextContent(type="text", text=(
                f"Query the knowledge graph for events related to: **{entity}**{time_range}\n\n"
                f"1. Use `raptor_tkg_query` with query=\"{entity}\""
                + (f", time_start=\"{start_date}\"" if start_date else "")
                + (f", time_end=\"{end_date}\"" if end_date else "") + ".\n"
                f"2. List the key events in chronological order.\n"
                f"3. Describe the relationships between entities involved.\n"
                f"4. Note any temporal gaps or uncertainties in the data."
            )),
        )]

    if name == "raptor_quick_answer":
        question = arguments.get("question", "")
        return [types.PromptMessage(
            role="user",
            content=types.TextContent(type="text", text=(
                f"{question}\n\n"
                f"Please answer using the Raptor knowledge base:\n"
                f"1. Use `raptor_chat` with message=\"{question}\" to get a grounded answer.\n"
                f"2. Present the answer in plain language — no jargon.\n"
                f"3. If the knowledge base doesn't have enough information, say so clearly."
            )),
        )]

    if name == "raptor_explore_topic":
        topic = arguments.get("topic", "")
        return [types.PromptMessage(
            role="user",
            content=types.TextContent(type="text", text=(
                f"I want to learn everything available about: **{topic}**\n\n"
                f"Please give me a comprehensive overview:\n"
                f"1. Use `raptor_search` with query=\"{topic}\" and top_k=10 to find all relevant content.\n"
                f"2. Group the results by type (videos, documents, audio, images).\n"
                f"3. For each group, summarise the key information found.\n"
                f"4. If videos are found, also run `raptor_video_search` with query=\"{topic}\" "
                f"to get specific timestamps.\n"
                f"5. End with a short overall summary of what the knowledge base contains on this topic."
            )),
        )]

    if name == "raptor_find_in_video":
        what = arguments.get("what", "")
        return [types.PromptMessage(
            role="user",
            content=types.TextContent(type="text", text=(
                f"Find moments in videos where: **{what}**\n\n"
                f"1. Use `raptor_video_search` with query=\"{what}\" and top_k=5.\n"
                f"2. For each result, show:\n"
                f"   - Video filename\n"
                f"   - Timestamp (start → end)\n"
                f"   - What is happening at that moment (from the segment text)\n"
                f"3. Sort by relevance — most relevant moment first.\n"
                f"4. If no results are found, suggest a rephrased search term."
            )),
        )]

    if name == "raptor_upload_workflow":
        filename = arguments.get("filename", "")
        content_base64 = arguments.get("content_base64", "")
        content_type = arguments.get("content_type", "application/octet-stream")
        return [types.PromptMessage(
            role="user",
            content=types.TextContent(type="text", text=(
                f"Upload the file **{filename}** and wait until processing is complete.\n\n"
                f"Step 1 — Upload:\n"
                f"  Call `raptor_upload_asset` with:\n"
                f"    filename=\"{filename}\"\n"
                f"    content_base64=\"{content_base64[:40]}...\" (truncated)\n"
                f"    content_type=\"{content_type}\"\n"
                f"  Save the returned `correlation_id` and `asset_path`.\n\n"
                f"Step 2 — Poll status:\n"
                f"  Repeatedly call `raptor_check_status` with the `correlation_id`.\n"
                f"  The `m_type` maps from content_type: video/* → 'video', audio/* → 'audio',\n"
                f"  application/pdf or text/* → 'document', image/* → 'image'.\n"
                f"  Poll every 10–15 seconds until status is 'complete' or 'failed'.\n"
                f"  Expected progression: queued → transcribing → extracting → indexing → complete\n\n"
                f"Step 3 — Verify:\n"
                f"  Once status is 'complete', call `raptor_search` with a keyword from the filename\n"
                f"  to confirm the asset is now indexed and searchable.\n\n"
                f"Report the final asset_path, version_id, and processing duration."
            )),
        )]

    if name == "raptor_search_strategy":
        query = arguments.get("query", "")
        media_type = arguments.get("media_type", "")
        type_clause = f" (media_type filter: \"{media_type}\")" if media_type else " (all media types)"
        return [types.PromptMessage(
            role="user",
            content=types.TextContent(type="text", text=(
                f"Compare search strategies for query: **{query}**{type_clause}\n\n"
                f"Run the following searches and compare results:\n\n"
                f"Option A — Hybrid (recommended default):\n"
                f"  `raptor_search` query=\"{query}\" top_k=5"
                + (f" type=\"{media_type}\"" if media_type else "") + "\n"
                f"  Combines BM25 + vector via RRF, then reranks. Best general-purpose.\n\n"
                f"Option B — Keyword only (BM25):\n"
                f"  `raptor_search_bm25` query=\"{query}\" top_k=5"
                + (f" type=\"{media_type}\"" if media_type else "") + "\n"
                f"  Fast, exact-match friendly. Best for proper nouns and IDs.\n\n"
                f"Option C — Semantic only (vector):\n"
                f"  `raptor_search_vector` query=\"{query}\" top_k=5"
                + (f" type=\"{media_type}\"" if media_type else "") + "\n"
                f"  Best for concept-level queries where exact wording doesn't matter.\n\n"
                + (f"Option D — Video-specific (if media_type is videos):\n"
                   f"  `raptor_video_search` query=\"{query}\" top_k=5\n"
                   f"  4-way RRF (BM25+vector+GraphRAG+TKG) with timestamp segments.\n\n"
                   if not media_type or media_type == "videos" else "") +
                f"After running all applicable options, summarise:\n"
                f"  - Which returned the most relevant results?\n"
                f"  - Were there unique results in any single mode?\n"
                f"  - Recommended strategy for this type of query."
            )),
        )]

    if name == "raptor_rag_pipeline":
        question = arguments.get("question", "")
        mode = arguments.get("mode", "direct")
        top_k = arguments.get("top_k", "5")
        if mode == "agent":
            tool_name = "raptor_a2a_agent"
            mode_desc = "multi-step agent loop — the LLM plans and executes multiple retrieval steps before answering"
        else:
            tool_name = "raptor_a2a_direct"
            mode_desc = "single-pass retrieval+generation — fast, deterministic, lower latency"
        return [types.PromptMessage(
            role="user",
            content=types.TextContent(type="text", text=(
                f"Answer the following question using the Raptor RAG pipeline ({mode} mode):\n\n"
                f"**Question:** {question}\n\n"
                f"Mode: `{mode}` — {mode_desc}\n\n"
                f"Call `{tool_name}` with:\n"
                f"  question=\"{question}\"\n"
                f"  top_k={top_k}\n\n"
                f"Return format:\n"
                f"  - answer: LLM-generated response grounded in retrieved chunks\n"
                f"  - sources: list of asset_path + chunk text that supported the answer\n"
                f"  - retrieval_count: number of chunks retrieved\n\n"
                f"If the answer is empty or confidence is low, retry with:\n"
                f"  1. A rephrased query\n"
                f"  2. Higher top_k (e.g. 10)\n"
                f"  3. Switch to '{('agent' if mode == 'direct' else 'direct')}' mode for comparison"
            )),
        )]

    raise ValueError(f"Unknown prompt: {name}")


def register(mcp: FastMCP) -> None:
    """Register all Raptor prompts with the FastMCP server."""

    @mcp.prompt(
        name="raptor_search_and_summarise",
        description="Search the Raptor knowledge base and return a concise summary with sources.",
    )
    async def raptor_search_and_summarise(topic: str, top_k: str = "10") -> list[types.PromptMessage]:
        return get_prompt_messages("raptor_search_and_summarise", {"topic": topic, "top_k": top_k})

    @mcp.prompt(
        name="raptor_video_analysis",
        description="Find and analyse video clips on a topic, listing relevant timestamps and key points.",
    )
    async def raptor_video_analysis(topic: str, top_k: str = "5") -> list[types.PromptMessage]:
        return get_prompt_messages("raptor_video_analysis", {"topic": topic, "top_k": top_k})

    @mcp.prompt(
        name="raptor_document_qa",
        description="Document-grounded Q&A: answer using only retrieved document content.",
    )
    async def raptor_document_qa(question: str) -> list[types.PromptMessage]:
        return get_prompt_messages("raptor_document_qa", {"question": question})

    @mcp.prompt(
        name="raptor_temporal_query",
        description="Time-range knowledge graph query: find events within a date window.",
    )
    async def raptor_temporal_query(
        entity: str,
        start_date: str = "",
        end_date: str = "",
    ) -> list[types.PromptMessage]:
        return get_prompt_messages("raptor_temporal_query", {
            "entity": entity, "start_date": start_date, "end_date": end_date,
        })

    @mcp.prompt(
        name="raptor_quick_answer",
        description="Ask any question and get an answer from the knowledge base.",
    )
    async def raptor_quick_answer(question: str) -> list[types.PromptMessage]:
        return get_prompt_messages("raptor_quick_answer", {"question": question})

    @mcp.prompt(
        name="raptor_explore_topic",
        description="Get a comprehensive overview of a topic from all available content.",
    )
    async def raptor_explore_topic(topic: str) -> list[types.PromptMessage]:
        return get_prompt_messages("raptor_explore_topic", {"topic": topic})

    @mcp.prompt(
        name="raptor_find_in_video",
        description="Find specific moments in videos — returns timestamps you can jump to directly.",
    )
    async def raptor_find_in_video(what: str) -> list[types.PromptMessage]:
        return get_prompt_messages("raptor_find_in_video", {"what": what})

    @mcp.prompt(
        name="raptor_upload_workflow",
        description="[Dev] Full upload workflow: upload a file, poll processing status until complete, then confirm it is searchable.",
    )
    async def raptor_upload_workflow(
        filename: str,
        content_base64: str,
        content_type: str,
    ) -> list[types.PromptMessage]:
        return get_prompt_messages("raptor_upload_workflow", {
            "filename": filename, "content_base64": content_base64, "content_type": content_type,
        })

    @mcp.prompt(
        name="raptor_search_strategy",
        description="[Dev] Choose and compare search modes (hybrid / BM25 / vector / video) for a given query.",
    )
    async def raptor_search_strategy(
        query: str,
        media_type: str = "",
    ) -> list[types.PromptMessage]:
        return get_prompt_messages("raptor_search_strategy", {
            "query": query, "media_type": media_type,
        })

    @mcp.prompt(
        name="raptor_rag_pipeline",
        description="[Dev] Run a RAG query using raptor_a2a_direct (fast) or raptor_a2a_agent (multi-step reasoning).",
    )
    async def raptor_rag_pipeline(
        question: str,
        mode: str = "direct",
        top_k: str = "5",
    ) -> list[types.PromptMessage]:
        return get_prompt_messages("raptor_rag_pipeline", {
            "question": question, "mode": mode, "top_k": top_k,
        })
