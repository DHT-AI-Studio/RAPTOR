# services/video_indexer_service/kafka_handler.py
# Enriches each entry with moment metadata before hybrid-search upload.

import json
import logging
import os
import tempfile
import uuid
import aiohttp
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from datetime import datetime
from typing import Any, Dict, List, Optional
import config
from message_utils import create_response_message
from dotenv import load_dotenv
import opencc

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(parent_dir, ".env"))
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
HYBRID_SEARCH_INGEST_URL = os.getenv("HYBRID_SEARCH_INGEST_URL")

logger = logging.getLogger(__name__)

_cc = opencc.OpenCC("s2tw")


class VideoIndexerKafkaHandler:
    def __init__(self):
        self.consumer = None
        self.producer = None

    async def start_consumer(self):
        self.consumer = AIOKafkaConsumer(
            config.KAFKA_TOPIC_SAVE_REQUEST,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=config.KAFKA_GROUP_ID,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )

        self.producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        await self.consumer.start()
        await self.producer.start()
        logger.info(f"Kafka consumer started, listening to {config.KAFKA_TOPIC_SAVE_REQUEST}")

        try:
            async for message in self.consumer:
                await self.process_message(message.value)
        finally:
            await self.consumer.stop()
            await self.producer.stop()

    async def process_message(self, message: Dict[str, Any]):
        payload = message.get("payload", {})
        parameters = payload.get("parameters", {})
        summary_result_path = parameters.get("summary_result_path")
        video_analysis_path: Optional[str] = None
        audio_analysis_path: Optional[str] = None
        try:
            logger.info(f"Received message: {message.get('message_id')}")

            if not summary_result_path:
                raise ValueError("Missing summary_result_path in parameters")

            chunking_moments: List[Dict[str, Any]] = parameters.get("chunking_moments", [])
            video_analysis_path = parameters.get("video_analysis_path")
            audio_analysis_path = parameters.get("audio_analysis_path")
            logger.info(f"[DEBUG] video_analysis_path={video_analysis_path}, exists={video_analysis_path and os.path.exists(video_analysis_path)}")
            contextualized_moments: List[Dict[str, Any]] = parameters.get("contextualized_moments", [])
            filename: str = parameters.get("filename", "")
            asset_path: str = parameters.get("asset_path", "")
            version_id: str = parameters.get("version_id", "")
            status: str = parameters.get("status", "active")
            branch_id: str = payload.get("user_id", "")

            results = await self.ingest_to_search(
                summary_result_path, chunking_moments,
                video_analysis_path, audio_analysis_path,
                filename, asset_path, version_id, status,
                contextualized_moments=contextualized_moments,
                branch_id=branch_id,
            )

            response = create_response_message(message, status="success", results=results)
            await self.send_response(response)
            logger.info(f"Successfully processed message: {message.get('message_id')}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            response = create_response_message(message, status="failed", error=str(e))
            await self.send_response(response)
        finally:
            for cleanup_path in [summary_result_path, video_analysis_path, audio_analysis_path]:
                if cleanup_path:
                    try:
                        if os.path.exists(cleanup_path):
                            os.remove(cleanup_path)
                    except Exception as cleanup_err:
                        logger.warning(f"Failed to clean up file {cleanup_path}: {cleanup_err}")

    async def ingest_to_search(
        self,
        file_path: str,
        chunking_moments: List[Dict[str, Any]] = None,
        video_analysis_path: Optional[str] = None,
        audio_analysis_path: Optional[str] = None,
        filename: str = "",
        asset_path: str = "",
        version_id: str = "",
        status: str = "active",
        contextualized_moments: Optional[List[Dict[str, Any]]] = None,
        branch_id: str = "",
    ) -> Dict[str, Any]:
        """
        Read summary JSON, keep only the summary entry,
        then build one text entry per 10-second moment (OCR+ASR+LVLM).
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            summary_entries = self._enrich_summary(data, video_analysis_path, branch_id=branch_id)
            moment_text_entries = self._build_moment_text_entries(
                chunking_moments or [],
                video_analysis_path,
                audio_analysis_path,
                filename, asset_path, version_id, status,
                contextualized_moments=contextualized_moments or [],
                branch_id=branch_id,
            )

            enriched = summary_entries + moment_text_entries
            logger.info(f"Total entries: {len(enriched)} (summary + {len(moment_text_entries)} moments)")

            if os.getenv("DEBUG_DUMP_PAYLOAD", "").lower() in ("1", "true", "yes"):
                debug_dir = "/tmp/media_processing/video_processing/qdrant_debug"
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, f"{version_id}_qdrant_payload.json")
                with open(debug_path, "w", encoding="utf-8") as dbf:
                    json.dump(enriched, dbf, indent=2, ensure_ascii=False)
                logger.info(f"[DEBUG] Qdrant payload saved to: {debug_path}")

            if not HYBRID_SEARCH_INGEST_URL:
                logger.info("[DEBUG] HYBRID_SEARCH_INGEST_URL not set — skipping upload (dry-run)")
                return {"dry_run": True, "total_entries": len(enriched), "debug_file": debug_path, "records": enriched, "text_entries": moment_text_entries}

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json")
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp:
                    json.dump(enriched, tmp, ensure_ascii=False)

                async with aiohttp.ClientSession() as session:
                    with open(tmp_path, "rb") as f:
                        form = aiohttp.FormData()
                        form.add_field(
                            "file", f,
                            filename=os.path.basename(file_path),
                            content_type="application/json",
                        )
                        async with session.post(HYBRID_SEARCH_INGEST_URL, data=form) as resp:
                            if resp.status != 200:
                                raise Exception(f"Search API error: {resp.status}")
                            result = await resp.json()
                            logger.info(f"Search API response: {result}")
                            return {**result.get("results", {}), "records": enriched, "text_entries": moment_text_entries}
            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Error calling search ingest API: {e}")
            raise

    def _enrich_summary(
        self,
        data: Any,
        video_analysis_path: Optional[str] = None,
        branch_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Keep only the summary entry; summary was built by video_summary_service from OCR+ASR+LVLM."""
        if not isinstance(data, list):
            return []
        entries = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            if entry.get("payload", entry).get("embedding_type") != "summary":
                continue
            if branch_id:
                entry = dict(entry)
                entry["payload"] = {**entry.get("payload", {}), "branch_id": branch_id}
            entries.append(entry)
        return entries

    def _assign_asr_to_moment(
        self,
        audio_chunks: List[Dict[str, Any]],
        start: float,
        end: float,
    ):
        """
        Use word-level timestamps to precisely extract ASR text within a moment window.
        Falls back to segment center-point if no word timestamps available.
        Returns (asr_text, speaker).
        """
        words_in_moment = []
        speakers_in_moment = []

        for segment in audio_chunks:
            if not isinstance(segment, dict):
                continue
            speaker = segment.get("speaker")
            words = segment.get("words", [])

            if words:
                for w in words:
                    word_start = float(w.get("start", -1))
                    if start <= word_start < end:
                        words_in_moment.append(w.get("word", ""))
                        if speaker:
                            speakers_in_moment.append(speaker)
            else:
                seg_start = float(segment.get("start_time", 0))
                seg_end = float(segment.get("end_time", 0))
                center = (seg_start + seg_end) / 2
                if start <= center < end and segment.get("text"):
                    words_in_moment.append(segment["text"])
                    if speaker:
                        speakers_in_moment.append(speaker)

        asr_text = _cc.convert("".join(words_in_moment))
        speakers = list(dict.fromkeys(speakers_in_moment))
        speaker = speakers[0] if len(speakers) == 1 else ", ".join(speakers) if speakers else "Unknown"
        return asr_text, speaker

    def _build_moment_text_entries(
        self,
        chunking_moments: List[Dict[str, Any]],
        video_analysis_path: Optional[str],
        audio_analysis_path: Optional[str],
        filename: str,
        asset_path: str,
        version_id: str,
        status: str,
        contextualized_moments: Optional[List[Dict[str, Any]]] = None,
        branch_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Build one text entry per 10-second moment with combined OCR + ASR + LVLM text.
        When contextualized_moments is provided, prepends [上下文補充] prefix (Contextual RAG).
        """
        if not chunking_moments:
            return []

        # Build contextual_prefix lookup: moment_index → prefix
        prefix_map: Dict[int, str] = {}
        for cm in (contextualized_moments or []):
            idx = cm.get("moment_index")
            if idx is not None:
                prefix_map[idx] = cm.get("contextual_prefix", "")
        if prefix_map:
            logger.info(f"Contextual prefixes available for {len(prefix_map)} moments")

        # Load video analysis (OCR frames + per-moment LVLM descriptions)
        video_frames = []
        moment_desc_map: dict = {}
        if video_analysis_path and os.path.exists(video_analysis_path):
            try:
                with open(video_analysis_path, "r", encoding="utf-8") as f:
                    va = json.load(f)
                video_frames = va.get("frames", [])
                for md in va.get("moment_descriptions", []):
                    moment_desc_map[md.get("moment_index")] = md.get("lvlm_description", "")
                logger.info(f"Loaded {len(moment_desc_map)} per-moment VLM descriptions")
            except Exception as e:
                logger.warning(f"Failed to load video analysis: {e}")

        # Load audio analysis
        audio_chunks = []
        if audio_analysis_path and os.path.exists(audio_analysis_path):
            try:
                with open(audio_analysis_path, "r", encoding="utf-8") as f:
                    audio_chunks = json.load(f)
                if not isinstance(audio_chunks, list):
                    audio_chunks = []
            except Exception as e:
                logger.warning(f"Failed to load audio analysis: {e}")

        upload_time = datetime.utcnow().isoformat()
        entries = []

        for moment in chunking_moments:
            start = moment.get("start_time", 0.0)
            end = moment.get("end_time", 0.0)
            moment_index = moment.get("moment_index", 0)

            # OCR: frames whose timestamp falls within [start, end)
            moment_frames = sorted(
                [f for f in video_frames
                 if isinstance(f, dict) and start <= f.get("timestamp", -1) < end and f.get("text")],
                key=lambda f: f.get("timestamp", 0)
            )
            ocr_parts = [f"frame{i+1}:{{{f['text']}}}" for i, f in enumerate(moment_frames)]
            ocr_text = ", ".join(ocr_parts)

            # ASR: word-level timestamp slicing
            asr_text, speaker = self._assign_asr_to_moment(audio_chunks, start, end)

            # Per-moment LVLM description
            lvlm_description = moment_desc_map.get(moment_index, "")

            parts = []
            if ocr_text:
                parts.append(f"ocr:{{{ocr_text}}}")
            if asr_text:
                parts.append(f"asr:{{{asr_text}}}")
            if lvlm_description:
                parts.append(f"lvlm:{{{lvlm_description}}}")
            combined = " / ".join(parts)
            if not combined:
                continue

            # Contextual RAG: prepend contextual_prefix when available
            contextual_prefix = prefix_map.get(moment_index, "")
            if contextual_prefix:
                combined = f"contextual:{{{contextual_prefix}}} / {combined}"

            entries.append({
                "id": str(uuid.uuid4()),
                "payload": {
                    "video_id": f"{filename}_moment_{moment_index}",
                    "text": combined,
                    "filename": filename,
                    "chunk_index": moment_index,
                    "start_time": str(start),
                    "end_time": str(end),
                    "speaker": speaker,
                    "type": "videos",
                    "source": filename.rsplit('.', 1)[-1].lower(),
                    "upload_time": upload_time,
                    "embedding_type": "text",
                    "asset_path": asset_path,
                    "version_id": version_id,
                    "branch_id": branch_id,
                    "status": status,
                },
            })

        logger.info(f"Built {len(entries)} moment text entries")
        return entries

    async def send_response(self, response: Dict[str, Any]):
        try:
            await self.producer.send_and_wait(config.KAFKA_TOPIC_SAVE_RESULT, response)
            logger.info(f"Response sent: {response.get('message_id')}")
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            raise
