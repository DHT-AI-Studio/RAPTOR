"""CM-3 (VIE01-406): video keyframe moderation -- per-moment aggregation and the
video rollup (video_chunking_service/video_moderation.py), the chunking service
calling it, the orchestrator writing the rollup to Module 04 and passing it on,
and the indexer putting the flags on what goes to Module 25
(video_indexer_service/moderation_fields.py).

Module 23 is faked (an aiohttp test server answering like CM-1's
/infer/google-shieldgemma-2 with modality=video_frame). The real-Module-23 run is
tests/unit/test_video_moderation_live.py.

Run from the 11-video-processing/ directory:
    pytest tests/unit/ -v
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer
from conftest import make_request_message

_SERVICES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "worker", "services")
_CHUNKING_DIR = os.path.join(_SERVICES, "video_chunking_service")


def _load(name: str, path: str):
    """Import one service file under its own name, without putting its service
    directory on sys.path (every service has its own config.py/kafka_handler.py)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vm = _load("cm3_video_moderation", os.path.join(_CHUNKING_DIR, "video_moderation.py"))
fields = _load("cm3_moderation_fields", os.path.join(_SERVICES, "video_indexer_service", "moderation_fields.py"))


def frame(*flagged: str, decision: str | None = None, **scores: float) -> dict:
    """A Module 23 `moderation` block for one frame."""
    names = ("nsfw", "violence", "hate", "self_harm")
    return {
        "modality": "video_frame",
        "decision": decision or ("block" if flagged else "allow"),
        "flagged": list(flagged),
        "not_covered": ["hate"],
        "categories": {n: {"score": scores.get(n), "flagged": n in flagged} for n in names},
        "source": "active_policy", "policy_name": "moderation-policy", "policy_version": "1.0",
        "moderation_version": "1.0",
    }


def moments(n: int, frames_each: int = 2) -> list[dict]:
    return [{"moment_index": i, "start_time": float(i * 10), "end_time": float(i * 10 + 10),
             "keyframe_paths": [f"/kf/m{i}_f{j}.jpg" for j in range(frames_each)]} for i in range(n)]


# ── parse_frame ───────────────────────────────────────────────────────────────


def test_parse_frame_returns_the_moderation_block():
    assert vm.parse_frame({"overall_verdict": "safe", "moderation": frame()})["decision"] == "allow"


@pytest.mark.parametrize("body", [
    {"overall_verdict": "unsafe", "policies": {}},          # Module 23 without CM-1
    {"moderation": {"decision": "deny", "flagged": []}},
    {"moderation": {"decision": "allow"}},
    ["not", "an", "object"],
])
def test_parse_frame_rejects_answers_without_a_verdict(body):
    with pytest.raises(ValueError):
        vm.parse_frame(body)


def test_parse_frame_names_cm1_when_moderation_is_missing():
    with pytest.raises(ValueError, match="CM-1"):
        vm.parse_frame({"overall_verdict": "safe", "policies": {}})


# ── per-moment aggregation ────────────────────────────────────────────────────


def test_clean_moment():
    f = vm.aggregate_moment([frame(nsfw=0.01), frame(nsfw=0.03)], "shieldgemma-2")
    assert (f["verdict"], f["decision"], f["flagged"]) == ("safe", "allow", [])
    assert (f["frames_checked"], f["frames_failed"], f["scores"]["nsfw"]) == (2, 0, 0.03)
    assert f["not_covered"] == ["hate"]


def test_one_flagged_frame_flags_the_moment_with_the_worst_decision_and_max_scores():
    f = vm.aggregate_moment([frame(nsfw=0.1, violence=0.2),
                             frame("nsfw", decision="warn", nsfw=0.9, violence=0.1)], "shieldgemma-2")
    assert (f["verdict"], f["decision"], f["flagged"]) == ("unsafe", "warn", ["nsfw"])
    assert f["scores"] == {"nsfw": 0.9, "violence": 0.2}


def test_flagged_categories_are_merged_across_frames():
    f = vm.aggregate_moment([frame("violence", "self_harm"), frame("nsfw", "violence")], "m")
    assert f["flagged"] == ["violence", "self_harm", "nsfw"]


def test_partly_failed_moment_keeps_the_frames_that_were_checked():
    f = vm.aggregate_moment([frame("nsfw"), ValueError("HTTP 503: busy")], "m")
    assert (f["verdict"], f["frames_checked"], f["frames_failed"]) == ("unsafe", 1, 1)
    assert f["error"] == "ValueError: HTTP 503: busy"


def test_moment_with_no_verdict_is_an_error():
    f = vm.aggregate_moment([TimeoutError(), TimeoutError()], "m")
    assert (f["verdict"], f["decision"], f["frames_failed"], f["error"]) == ("error", None, 2, "TimeoutError")


def test_moment_without_frames_is_an_error():
    f = vm.aggregate_moment([], "m")
    assert (f["verdict"], f["error"]) == ("error", "no frames")


def test_llama_guard_frames_have_no_scores():
    lg = frame("hate")
    lg["categories"] = {"hate": {"flagged": True, "matched": ["S10"]}}
    f = vm.aggregate_moment([lg], "llama-guard-3-vision")
    assert (f["flagged"], f["scores"]) == (["hate"], {})


# ── video rollup ──────────────────────────────────────────────────────────────


def _flagged(ms: list[dict], **per_index: dict) -> list[dict]:
    for m in ms:
        m["moderation_flags"] = per_index.get(f"m{m['moment_index']}", vm.aggregate_moment([frame()], "m"))
    return ms


def test_rollup_of_a_clean_video():
    r = vm.rollup(_flagged(moments(3)), "shieldgemma-2", {"source": "default"}, 0.0)
    assert (r["verdict"], r["decision"], r["flagged"], r["flagged_moments"]) == ("safe", "allow", [], [])
    assert (r["moments_checked"], r["moments_failed"], r["frames_checked"]) == (3, 0, 3)
    assert (r["model"], r["modality"], r["policy"]) == ("shieldgemma-2", "video_frame", {"source": "default"})
    assert "latency_ms" in r and "checked_at" in r


def test_rollup_lists_the_flagged_moments_with_their_times():
    ms = _flagged(moments(3), m1=vm.aggregate_moment([frame("nsfw")], "m"),
                  m2=vm.aggregate_moment([frame("violence", decision="warn")], "m"))
    r = vm.rollup(ms, "m", None, 0.0)
    assert (r["verdict"], r["decision"], r["flagged"]) == ("unsafe", "block", ["nsfw", "violence"])
    assert r["flagged_moments"] == [
        {"moment_index": 1, "start_time": 10.0, "end_time": 20.0, "decision": "block", "flagged": ["nsfw"]},
        {"moment_index": 2, "start_time": 20.0, "end_time": 30.0, "decision": "warn", "flagged": ["violence"]},
    ]


def test_rollup_never_calls_an_incompletely_checked_video_safe():
    ms = _flagged(moments(2), m1=vm.aggregate_moment([TimeoutError()], "m"))
    r = vm.rollup(ms, "m", None, 0.0)
    assert (r["verdict"], r["moments_checked"], r["moments_failed"], r["error"]) == ("error", 1, 1, "TimeoutError")


def test_rollup_flagged_wins_over_failed_moments():
    ms = _flagged(moments(2), m0=vm.aggregate_moment([frame("nsfw")], "m"),
                  m1=vm.aggregate_moment([TimeoutError()], "m"))
    assert vm.rollup(ms, "m", None, 0.0)["verdict"] == "unsafe"


# ── moderate_moments against a fake Module 23 ─────────────────────────────────


@pytest_asyncio.fixture
async def module23(monkeypatch):
    """Fake Module 23: answers per keyframe file name ("flag" in the name ->
    nsfw flagged, "fail" -> HTTP 503) and records each request's form."""
    seen: list[dict] = []

    async def infer(request: web.Request) -> web.Response:
        form = await request.post()
        name = form["image"].filename
        seen.append({"path": request.path, "name": name, "modality": form.get("modality")})
        if "fail" in name:
            return web.Response(status=503, text="GPU busy")
        body = {"overall_verdict": "safe", "policies": {},
                "moderation": frame("nsfw", nsfw=0.95) if "flag" in name else frame(nsfw=0.02)}
        return web.json_response(body)

    app = web.Application()
    app.router.add_post("/infer/google-shieldgemma-2", infer)
    app.router.add_post("/infer/Llama-Guard-3-11B-Vision", infer)
    server = TestServer(app)
    await server.start_server()
    monkeypatch.setenv("GUARDRAIL_URL", str(server.make_url("")))
    for var in ("VIDEO_MODERATION_MODEL", "VIDEO_MODERATION_FAIL_MODE", "VIDEO_MODERATION_SAMPLE_EVERY_SEC",
                "VIDEO_MODERATION_SKIP_DUPLICATES", "VIDEO_MODERATION_DUPLICATE_MAX_CHANGE",
                "VIDEO_MODERATION_SCENE_CUTS", "VIDEO_MODERATION_SCENE_THRESHOLD",
                "VIDEO_MODERATION_MAX_CUTS_PER_MOMENT"):
        monkeypatch.delenv(var, raising=False)
    yield seen
    await server.close()


@pytest.fixture
def keyframes(tmp_path):
    def make(*names: str) -> list[str]:
        paths = []
        for name in names:
            p = tmp_path / name
            p.write_bytes(b"\xff\xd8 fake jpeg")
            paths.append(str(p))
        return paths
    return make


@pytest.mark.asyncio
async def test_moderate_moments_sends_every_keyframe_as_video_frame(module23, keyframes):
    ms = [{"moment_index": 0, "start_time": 0.0, "end_time": 10.0, "keyframe_paths": keyframes("a.jpg", "b.jpg")},
          {"moment_index": 1, "start_time": 10.0, "end_time": 20.0, "keyframe_paths": keyframes("c.jpg", "flag.jpg")}]
    r = await vm.moderate_moments(ms)
    assert sorted(s["name"] for s in module23) == ["a.jpg", "b.jpg", "c.jpg", "flag.jpg"]
    assert {(s["path"], s["modality"]) for s in module23} == {("/infer/google-shieldgemma-2", "video_frame")}
    assert (ms[0]["moderation_flags"]["verdict"], ms[1]["moderation_flags"]["verdict"]) == ("safe", "unsafe")
    assert (r["verdict"], r["flagged"], r["frames_checked"]) == ("unsafe", ["nsfw"], 4)
    assert [m["moment_index"] for m in r["flagged_moments"]] == [1]
    assert r["policy"] == {"source": "active_policy", "policy_name": "moderation-policy",
                           "policy_version": "1.0", "moderation_version": "1.0"}


@pytest.mark.asyncio
async def test_duplicate_frames_reuse_the_verdict_of_the_frame_sent(module23, images):
    """A frame identical to the last one sent isn't sent again -- even when it is
    a copy of a flagged frame, so its moment is still flagged."""
    grey, flag = images("grey.jpg"), images("flag.jpg", square=True)
    ms = [{"moment_index": 0, "start_time": 0.0, "end_time": 10.0,
           "keyframe_paths": [grey, images("grey_again.jpg")]},
          {"moment_index": 1, "start_time": 10.0, "end_time": 20.0,
           "keyframe_paths": [flag, images("same_as_before.jpg", square=True)]},
          {"moment_index": 2, "start_time": 20.0, "end_time": 30.0,
           "keyframe_paths": [images("still_the_same.jpg", square=True)]}]
    r = await vm.moderate_moments(ms)
    assert sorted(s["name"] for s in module23) == ["flag.jpg", "grey.jpg"]       # sent 2 at a time: any order
    assert [m["moderation_flags"]["verdict"] for m in ms] == ["safe", "unsafe", "unsafe"]
    assert [m["moderation_flags"]["frames_reused"] for m in ms] == [1, 1, 1]
    assert (r["frames_checked"], r["frames_sent"], r["frames_reused"]) == (5, 2, 3)
    assert [m["moment_index"] for m in r["flagged_moments"]] == [1, 2]


@pytest.mark.asyncio
async def test_skip_duplicates_off_sends_every_frame(module23, images, monkeypatch):
    monkeypatch.setenv("VIDEO_MODERATION_SKIP_DUPLICATES", "false")
    ms = [{"moment_index": 0, "start_time": 0.0, "end_time": 10.0,
           "keyframe_paths": [images("a.jpg"), images("b.jpg"), images("c.jpg")]}]
    r = await vm.moderate_moments(ms)
    assert len(module23) == 3 and (r["frames_sent"], r["frames_reused"]) == (3, 0)


@pytest.mark.asyncio
async def test_sampled_frames_are_sent_and_their_folder_removed(module23, keyframes, monkeypatch, tmp_path):
    folder = tmp_path / "moderation_frames"
    folder.mkdir()
    paths = keyframes("s0.jpg", "s1_flag.jpg")
    seen_every = []

    def fake_sample(ms, every, threshold, max_cuts):
        seen_every.append((every, threshold, max_cuts))
        return [[paths[0]], [paths[1]]], str(folder), [0, 1], set()

    monkeypatch.setattr(vm, "sample_frames", fake_sample)
    monkeypatch.setenv("VIDEO_MODERATION_SAMPLE_EVERY_SEC", "1")
    ms = [{"moment_index": 0, "start_time": 0.0, "end_time": 10.0, "keyframe_paths": keyframes("kf_flag.jpg")},
          {"moment_index": 1, "start_time": 10.0, "end_time": 20.0, "keyframe_paths": []}]
    r = await vm.moderate_moments(ms)
    assert sorted(s["name"] for s in module23) == ["s0.jpg", "s1_flag.jpg"]      # not the keyframes
    assert [m["moment_index"] for m in r["flagged_moments"]] == [1]
    assert (seen_every, r["sample_every_sec"]) == ([(1.0, 0.3, 5)], 1.0)
    assert (r["scene_threshold"], r["frames_scene_cut"], ms[1]["moderation_flags"]["frames_scene_cut"]) == (0.3, 1, 1)
    assert not folder.exists()


@pytest.mark.asyncio
async def test_llama_guard_model_uses_its_endpoint(module23, keyframes, monkeypatch):
    monkeypatch.setenv("VIDEO_MODERATION_MODEL", "llama-guard-3-vision")
    await vm.moderate_moments([{"moment_index": 0, "keyframe_paths": keyframes("a.jpg")}])
    assert module23[0]["path"] == "/infer/Llama-Guard-3-11B-Vision"


@pytest.mark.asyncio
async def test_fail_open_records_the_failed_frames(module23, keyframes):
    ms = [{"moment_index": 0, "keyframe_paths": keyframes("fail1.jpg", "fail2.jpg")},
          {"moment_index": 1, "keyframe_paths": keyframes("a.jpg")}]
    r = await vm.moderate_moments(ms)
    assert ms[0]["moderation_flags"]["verdict"] == "error"
    assert "HTTP 503" in ms[0]["moderation_flags"]["error"]
    assert (r["verdict"], r["moments_failed"], r["frames_failed"]) == ("error", 1, 2)


@pytest.mark.asyncio
async def test_unreadable_keyframe_is_a_failed_frame(module23):
    r = await vm.moderate_moments([{"moment_index": 0, "keyframe_paths": ["/nonexistent/kf.jpg"]}])
    assert (r["verdict"], r["frames_failed"]) == ("error", 1)
    assert module23 == []


@pytest.mark.asyncio
async def test_fail_closed_raises(module23, keyframes, monkeypatch):
    monkeypatch.setenv("VIDEO_MODERATION_FAIL_MODE", "closed")
    with pytest.raises(vm.ModerationUnavailable, match="1 frame"):
        await vm.moderate_moments([{"moment_index": 0, "keyframe_paths": keyframes("a.jpg", "fail.jpg")}])


@pytest.mark.asyncio
async def test_fail_closed_also_counts_moments_without_keyframes(module23, keyframes, monkeypatch):
    monkeypatch.setenv("VIDEO_MODERATION_FAIL_MODE", "closed")
    with pytest.raises(vm.ModerationUnavailable):
        await vm.moderate_moments([{"moment_index": 0, "keyframe_paths": []}])


@pytest.fixture
def images(tmp_path):
    """Real JPEGs: a grey frame, or (square=True) the grey frame with a dark square
    in the middle. Files made with the same arguments are pixel-identical."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    def make(name: str, square: bool = False, corner: bool = False) -> str:
        img = np.full((360, 640, 3), 128, dtype=np.uint8)
        if square:
            img[130:230, 270:370] = 20
        if corner:
            img[0:40, 0:60] = 250            # ~1 % of the frame
        path = str(tmp_path / name)
        cv2.imwrite(path, img)
        return path
    return make


def test_plan_calls_reuses_only_near_identical_frames(images):
    grey, same, corner = images("g.jpg"), images("g2.jpg"), images("c.jpg", corner=True)
    sent, plan = vm.plan_calls([[grey, same], [corner]], True, 0.005)
    assert (sent, plan) == ([grey, corner], [[0, 0], [1]])      # a small object in a corner is a change


def test_plan_calls_reuses_any_frame_already_sent_for_the_moment(images):
    """A keyframe showing the same second as an earlier sample of its moment (not
    the last one sent) still reuses that sample's verdict."""
    grey, square, kf = images("g.jpg"), images("sq.jpg", square=True), images("kf.jpg")
    sent, plan = vm.plan_calls([[grey, square, kf]], True, 0.005)
    assert (sent, plan) == ([grey, square], [[0, 1, 0]])


def test_plan_calls_only_looks_back_one_frame_across_moments(images):
    grey, square, grey_again = images("g.jpg"), images("sq.jpg", square=True), images("g_again.jpg")
    sent, plan = vm.plan_calls([[grey, square], [grey_again]], True, 0.005)
    assert (sent, plan) == ([grey, square, grey_again], [[0, 1], [2]])  # not matched to moment 0's grey


def test_plan_calls_without_skipping_sends_everything(images):
    grey, same = images("g.jpg"), images("g2.jpg")
    assert vm.plan_calls([[grey, same]], False, 0.005) == ([grey, same], [[0, 1]])


def test_plan_calls_always_sends_an_unreadable_frame(images):
    grey = images("g.jpg")
    sent, plan = vm.plan_calls([[grey, "/nonexistent.jpg"]], True, 0.005)
    assert (sent, plan) == ([grey, "/nonexistent.jpg"], [[0, 1]])


def _timed_moments(tmp_path, ends):
    out, start = [], 0.0
    for i, end in enumerate(ends):
        out.append({"moment_index": i, "start_time": start, "end_time": end, "video_path": "/v.mp4",
                    "output_dir": str(tmp_path / "req" / f"moment_{i}"), "keyframe_paths": [f"/kf/{i}.jpg"]})
        start = end
    return out


def _grid(*times: float, cuts: tuple = ()) -> list[tuple[float, str, float]]:
    """What _extract_frames returns: the samples at `times`, plus frames at `cuts`
    with a scene score of 1, in time order."""
    frames = [(t, f"/s/{t:g}.jpg", 0.0) for t in times] + [(t, f"/c/{t:g}.jpg", 1.0) for t in cuts]
    return sorted(frames)


@pytest.fixture
def no_first_frame(monkeypatch):
    """_extract_at as if the video had no frame there; returns the times asked for."""
    asked: list[float] = []

    def fake(video, t, out):
        asked.append(t)
        return None, False     # no frame, and not for lack of a picture

    monkeypatch.setattr(vm, "_extract_at", fake)
    return asked


# ── split_frames: samples, and the brief shots between them ─────────────────

def test_split_frames_the_first_frame_of_each_window_is_the_sample():
    samples, brief = vm.split_frames(_grid(0, 2, 4, 6, 8), 2.0, 0.3, 10.0)
    assert [t for t, _ in samples] == [0, 2, 4, 6, 8] and brief == []


def test_a_shot_between_two_samples_gets_its_own_frame():
    """A 0.8 s shot from 4.6 to 5.4 s: the samples at 4 and 6 s show the shots on
    either side of it, so its first frame is kept."""
    samples, brief = vm.split_frames(_grid(0, 2, 4, 6, 8, cuts=(4.6, 5.4)), 2.0, 0.3, 10.0)
    assert [t for t, _ in samples] == [0, 2, 4, 6, 8]
    assert brief == [(4.6, "/c/4.6.jpg")]          # 5.4-10 s holds the samples at 6 and 8 s


def test_a_shot_that_holds_a_sample_costs_nothing():
    _, brief = vm.split_frames(_grid(0, 2, 4, 6, 8, cuts=(3.0, 5.5)), 2.0, 0.3, 10.0)
    assert brief == []                              # 3.0-5.5 holds 4 s, 5.5-10 holds 6 and 8 s


def test_a_cut_on_a_sample_is_just_that_sample():
    """The cut at 2 s is also the sample of its window: not counted twice. The
    2.5-3.0 s shot is a brief one; 3.0-5.0 s holds the sample at 4 s."""
    frames = [(0, "/s/0.jpg", 0.0), (2, "/s/2.jpg", 1.0), (2.5, "/c/2.5.jpg", 1.0), (3.0, "/c/3.jpg", 1.0),
              (4, "/s/4.jpg", 0.0)]
    samples, brief = vm.split_frames(frames, 2.0, 0.3, 5.0)
    assert [t for t, _ in samples] == [0, 2, 4] and brief == [(2.5, "/c/2.5.jpg")]


def test_the_last_shot_runs_to_the_end_of_the_video():
    assert vm.split_frames(_grid(0, 2, cuts=(3.9,)), 2.0, 0.3, 4.0)[1] == [(3.9, "/c/3.9.jpg")]
    assert vm.split_frames(_grid(0, 2, 4, cuts=(3.9,)), 2.0, 0.3, 6.0)[1] == []      # holds 4 s


def test_scores_at_or_under_the_threshold_are_no_cut():
    frames = [(0, "/s/0.jpg", 0.0), (1, "/c/1.jpg", 0.3), (2, "/s/2.jpg", 0.0)]
    assert vm.split_frames(frames, 2.0, 0.3, 4.0)[1] == []
    assert vm.split_frames(frames, 2.0, None, 4.0)[1] == []                          # scene cuts off


# ── sample_frames: frames per moment ─────────────────────────────────────────

def test_sample_frames_puts_each_frame_in_the_moment_holding_its_time(tmp_path, monkeypatch, no_first_frame):
    calls = []

    def fake_extract(video, every, threshold, out_dir):
        calls.append((video, every, threshold, out_dir))
        return _grid(*range(0, 24, 2), cuts=(13.2, 13.8))         # samples t = 0, 2, ..., 22

    monkeypatch.setattr(vm, "_extract_frames", fake_extract)
    lists, folder, cuts, _ = vm.sample_frames(_timed_moments(tmp_path, [10.0, 20.0, 25.0]), 2.0, 0.3)
    assert lists == [[f"/s/{t}.jpg" for t in (0, 2, 4, 6, 8)] + ["/kf/0.jpg"],      # samples, then keyframes
                     ["/s/10.jpg", "/s/12.jpg", "/c/13.2.jpg", "/s/14.jpg", "/s/16.jpg", "/s/18.jpg",
                      "/kf/1.jpg"],                                                # brief shot in time order
                     ["/s/20.jpg", "/s/22.jpg", "/kf/2.jpg"]]
    assert cuts == [0, 1, 0]
    assert folder == str(tmp_path / "req" / "moderation_frames") and calls[0][:3] == ("/v.mp4", 2.0, 0.3)
    assert no_first_frame == []                                    # every moment had frames


def test_brief_shots_are_capped_per_moment(tmp_path, monkeypatch, no_first_frame):
    flashes = tuple(round(4.1 + 0.2 * i, 1) for i in range(8))     # 8 cuts between the samples at 4 and 6 s
    monkeypatch.setattr(vm, "_extract_frames", lambda v, e, th, d: _grid(0, 2, 4, 6, 8, cuts=flashes))
    lists, _, cuts, _ = vm.sample_frames(_timed_moments(tmp_path, [10.0]), 2.0, 0.3, max_cuts=3)
    assert cuts == [3]
    assert [p for p in lists[0] if p.startswith("/c/")] == ["/c/4.1.jpg", "/c/4.3.jpg", "/c/4.5.jpg"]


def test_a_moment_without_sampled_frames_still_has_its_keyframes(tmp_path, monkeypatch, no_first_frame):
    monkeypatch.setattr(vm, "_extract_frames", lambda v, e, th, d: _grid(0, 2, 4, 6, 8))
    lists, _, _, _ = vm.sample_frames(_timed_moments(tmp_path, [10.0, 10.4]), 2.0)
    assert lists == [[f"/s/{t}.jpg" for t in (0, 2, 4, 6, 8)] + ["/kf/0.jpg"], ["/kf/1.jpg"]]
    assert no_first_frame == []


def test_a_moment_with_no_frame_at_all_gets_its_first_frame(tmp_path, monkeypatch):
    """The 90.0-90.05 s last moment of a 90.05 s video: no sample, no cut, no
    keyframe -- it used to be reported as unchecked (`no frames`)."""
    monkeypatch.setattr(vm, "_extract_frames", lambda v, e, th, d: _grid(*range(0, 90, 2)))
    asked = []

    def fake_at(video, t, out):
        asked.append((video, t))
        return out, False

    monkeypatch.setattr(vm, "_extract_at", fake_at)
    ms = _timed_moments(tmp_path, [float(e) for e in range(10, 100, 10)] + [90.05])
    ms[-1]["keyframe_paths"] = []
    lists, folder, _, no_picture = vm.sample_frames(ms, 2.0, 0.3)
    assert asked == [("/v.mp4", 90.0)] and no_picture == set()
    assert lists[-1] == [os.path.join(folder, "start_0009.jpg")]


def test_a_moment_with_no_frame_stays_empty_when_ffmpeg_has_none(tmp_path, monkeypatch, no_first_frame):
    monkeypatch.setattr(vm, "_extract_frames", lambda v, e, th, d: _grid(0, 2, 4, 6, 8))
    ms = _timed_moments(tmp_path, [10.0, 10.05])
    ms[1]["keyframe_paths"] = []
    lists, _, _, no_picture = vm.sample_frames(ms, 2.0)
    assert (lists[1], no_first_frame, no_picture) == ([], [10.0], set())


def test_a_moment_past_the_last_picture_is_no_picture(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "_extract_frames", lambda v, e, th, d: _grid(0, 2, 4, 6, 8))
    monkeypatch.setattr(vm, "_extract_at", lambda v, t, out: (None, True))
    ms = _timed_moments(tmp_path, [10.0, 10.047])
    ms[1]["keyframe_paths"] = []
    lists, _, _, no_picture = vm.sample_frames(ms, 2.0, 0.3)
    assert (lists[1], no_picture) == ([], {1})


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_mode", ["open", "closed"])
async def test_a_no_picture_moment_is_neither_checked_nor_failed(module23, images, tmp_path, monkeypatch, fail_mode):
    """A clean video whose audio outlasts its picture stays `safe` (it used to be
    `error`, and in closed mode fail the job)."""
    monkeypatch.setenv("VIDEO_MODERATION_FAIL_MODE", fail_mode)
    monkeypatch.setattr(vm, "_extract_frames", lambda v, e, th, d: [(0.0, images("s0.jpg"), 0.0)])
    monkeypatch.setattr(vm, "_extract_at", lambda v, t, out: (None, True))
    ms = _timed_moments(tmp_path, [10.0, 10.047])
    ms[1]["keyframe_paths"] = []
    ms[0]["keyframe_paths"] = []
    r = await vm.moderate_moments(ms)
    assert ms[1]["moderation_flags"]["verdict"] == "no_picture" and ms[1]["moderation_flags"]["flagged"] == []
    assert (r["verdict"], r["moments_checked"], r["moments_failed"], r["moments_no_picture"]) == ("safe", 1, 0, 1)
    assert "error" not in r


def test_rollup_does_not_count_no_picture_moments():
    ms = [{"moment_index": 0, "moderation_flags": vm.aggregate_moment([frame()], "m")},
          {"moment_index": 1, "moderation_flags": vm._no_picture_flags("m")}]
    r = vm.rollup(ms, "m", None, 0.0)
    assert (r["verdict"], r["moments_checked"], r["moments_failed"], r["moments_no_picture"]) == ("safe", 1, 0, 1)


def test_failed_sampling_falls_back_to_the_keyframes(tmp_path, monkeypatch, no_first_frame):
    monkeypatch.setattr(vm, "_extract_frames", lambda v, e, th, d: [])
    lists, _, cuts, _ = vm.sample_frames(_timed_moments(tmp_path, [10.0, 20.0]), 2.0, 0.3)
    assert (lists, cuts) == ([["/kf/0.jpg"], ["/kf/1.jpg"]], [0, 0])


@pytest.mark.asyncio
async def test_a_brief_sharp_scene_on_a_keyframe_is_caught(module23, images, tmp_path, monkeypatch):
    """The even samples all show the clean scene; only the description keyframe
    landed on the 1 s flagged one -- the moment is still flagged."""
    samples = [(float(t), images(f"s{t}.jpg"), 0.0) for t in (0, 2, 4, 6, 8)]
    monkeypatch.setattr(vm, "_extract_frames", lambda v, e, th, d: samples)
    ms = _timed_moments(tmp_path, [10.0])
    ms[0]["keyframe_paths"] = [images("kf_flag.jpg", square=True)]
    r = await vm.moderate_moments(ms)
    assert sorted(s["name"] for s in module23) == ["kf_flag.jpg", "s0.jpg"]  # 4 identical samples reused
    assert (ms[0]["moderation_flags"]["verdict"], r["frames_checked"], r["frames_reused"]) == ("unsafe", 6, 4)


@pytest.mark.asyncio
async def test_a_brief_blurred_shot_between_samples_is_caught(module23, images, tmp_path, monkeypatch):
    """A 0.6 s flagged shot between the samples at 4 and 6 s that no keyframe
    shows: only its scene-cut frame reaches Module 23, and the moment is flagged."""
    frames = [(float(t), images(f"s{t}.jpg"), 0.0) for t in (0, 2, 4, 6, 8)]
    frames.insert(3, (4.7, images("cut_flag.jpg", square=True), 1.0))
    frames.insert(4, (5.3, images("cut_back.jpg"), 1.0))
    monkeypatch.setattr(vm, "_extract_frames", lambda v, e, th, d: frames)
    ms = _timed_moments(tmp_path, [10.0])
    ms[0]["keyframe_paths"] = [images("kf.jpg")]
    r = await vm.moderate_moments(ms)
    assert "cut_flag.jpg" in [s["name"] for s in module23]
    assert (ms[0]["moderation_flags"]["verdict"], ms[0]["moderation_flags"]["frames_scene_cut"]) == ("unsafe", 1)
    assert (r["verdict"], r["frames_scene_cut"], r["scene_threshold"]) == ("unsafe", 1, 0.3)


@pytest.mark.asyncio
async def test_scene_cuts_off_asks_ffmpeg_for_samples_only(module23, images, tmp_path, monkeypatch):
    seen = []

    def fake_extract(video, every, threshold, out_dir):
        seen.append(threshold)
        return [(0.0, images("s0.jpg"), 0.0)]

    monkeypatch.setattr(vm, "_extract_frames", fake_extract)
    monkeypatch.setenv("VIDEO_MODERATION_SCENE_CUTS", "false")
    r = await vm.moderate_moments(_timed_moments(tmp_path, [10.0]))
    assert (seen, r["scene_threshold"], r["frames_scene_cut"]) == ([None], None, 0)


# ── ffmpeg ─────────────────────────────────────────────────────────────────────

def test_select_filter():
    assert vm._select_filter(2.0, None) == \
        r"select=isnan(prev_t)+gt(floor(t/2.0)\,floor(prev_t/2.0)),metadata=print:key=lavfi.scene_score,showinfo"
    assert vm._select_filter(2.0, 0.3).startswith(r"select=isnan(prev_t)+gt(floor(t/2.0)\,floor(prev_t/2.0))"
                                                  r"+gt(scene\,0.3),")


def test_parse_select_log():
    log = ("[Parsed_metadata_1 @ 0x1] frame:0    pts:0       pts_time:0\n"
           "[Parsed_metadata_1 @ 0x1] lavfi.scene_score=0.000000\n"
           "[Parsed_showinfo_2 @ 0x2] n:   0 pts:      0 pts_time:0       pos: 48 fmt:yuv420p\n"
           "[Parsed_showinfo_2 @ 0x2] color_range:unknown color_space:unknown\n"
           "Input #0, mov,mp4 -- not a frame line, pts_time:99\n"
           "[Parsed_metadata_1 @ 0x1] frame:1    pts:65536   pts_time:5.12\n"
           "[Parsed_metadata_1 @ 0x1] lavfi.scene_score=1.000000\n"
           "[Parsed_showinfo_2 @ 0x2] n:   1 pts:  65536 pts_time:5.12    pos: 99 fmt:yuv420p\n"
           "[Parsed_showinfo_2 @ 0x2] n:   2 pts:  76800 pts_time:6       pos: 120 fmt:yuv420p\n")
    assert vm._parse_select_log(log) == [(0.0, 0.0), (5.12, 1.0), (6.0, 0.0)]   # 6 s: no score logged


def _make_video(path: str, segments: list[tuple[str, float]]) -> str:
    """H.264 at 25 fps from (colour, seconds) segments."""
    inputs, labels = [], []
    for i, (colour, seconds) in enumerate(segments):
        inputs += ["-f", "lavfi", "-i", f"color=c={colour}:s=320x180:d={seconds}:r=25"]
        labels.append(f"[{i}:v]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex",
                    f"{''.join(labels)}concat=n={len(segments)}:v=1:a=0,format=yuv420p", "-c:v", "libx264", path],
                   check=True, timeout=120)
    return path


@pytest.mark.skipif(not (shutil.which("ffmpeg")), reason="needs ffmpeg")
def test_extract_frames_samples_at_exact_times_and_finds_the_cuts(tmp_path):
    """A real 10 s video: red, a 0.8 s blue shot at 4.6-5.4 s, red again. Samples at
    exactly 0, 2, 4, 6 and 8 s (all red), plus the two cuts; only the blue shot,
    which no sample shows, is a brief shot."""
    cv2 = pytest.importorskip("cv2")
    video = _make_video(str(tmp_path / "flash.mp4"), [("red", 4.6), ("blue", 0.8), ("red", 4.6)])
    frames = vm._extract_frames(video, 2.0, 0.3, str(tmp_path / "out"))
    assert [round(t, 2) for t, _, _ in frames] == [0, 2, 4, 4.6, 5.4, 6, 8]
    samples, brief = vm.split_frames(frames, 2.0, 0.3, 10.0)
    assert [round(t, 2) for t, _ in samples] == [0, 2, 4, 6, 8]
    assert [round(t, 2) for t, _ in brief] == [4.6]
    assert cv2.imread(brief[0][1])[90, 160, 0] > 200                 # BGR: index 0 = blue
    assert all(cv2.imread(p)[90, 160, 2] > 200 for _, p in samples)  # red


@pytest.mark.skipif(not (shutil.which("ffmpeg")), reason="needs ffmpeg")
def test_extract_frames_without_scene_cuts_gives_the_samples_only(tmp_path):
    video = _make_video(str(tmp_path / "flash.mp4"), [("red", 4.6), ("blue", 0.8), ("red", 4.6)])
    assert [round(t, 2) for t, _, _ in vm._extract_frames(video, 2.0, None, str(tmp_path / "out"))] == [0, 2, 4, 6, 8]


@pytest.mark.skipif(not (shutil.which("ffmpeg")), reason="needs ffmpeg")
def test_extract_at_gets_a_frame_from_a_last_sliver(tmp_path):
    """The last 0.05 s of a 10.05 s video still has a frame to check."""
    video = _make_video(str(tmp_path / "v.mp4"), [("green", 10.05)])
    out = str(tmp_path / "start.jpg")
    assert vm._extract_at(video, 10.0, out) == (out, False) and os.path.getsize(out) > 0


@pytest.mark.skipif(not (shutil.which("ffmpeg")), reason="needs ffmpeg")
def test_extract_at_past_the_last_picture_is_no_picture_not_an_error(tmp_path):
    """Audio longer than the picture: picture to 90.0 s, audio to 90.047 s, so the last
    moment (90.0-90.047 s) has no picture. ffmpeg runs fine and gives no frame."""
    video = str(tmp_path / "av.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=green:s=320x180:d=10:r=30",
                    "-f", "lavfi", "-i", "sine=d=10.05", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    video], check=True, timeout=120)
    assert vm._extract_at(video, 10.0, str(tmp_path / "x.jpg")) == (None, True)
    assert vm._extract_at(str(tmp_path / "missing.mp4"), 1.0, str(tmp_path / "y.jpg")) == (None, False)


def _log_for(n: int) -> bytes:
    return "".join(f"[Parsed_showinfo_2 @ 0x2] n:{i} pts:{i} pts_time:{2 * i}\n" for i in range(n)).encode()


def test_extract_frames_recovers_from_a_timed_out_gpu_attempt(tmp_path, monkeypatch):
    """A GPU attempt that raises (timeout, or any other subprocess failure) is
    treated like a nonzero exit code: fall back to the CPU attempt instead of
    letting the exception reach the caller."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append("gpu" if "-hwaccel" in cmd else "cpu")
        if "-hwaccel" in cmd:
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        open(cmd[-1].replace("%06d", "000001"), "wb").close()
        return subprocess.CompletedProcess(cmd, 0, b"", _log_for(1))

    monkeypatch.setattr(vm.subprocess, "run", fake_run)
    frames = vm._extract_frames("/v.mp4", 2.0, 0.3, str(tmp_path / "out"))
    assert calls == ["gpu", "cpu"]
    assert frames == [(0.0, str(tmp_path / "out" / "f_000001.jpg"), 0.0)]


def test_extract_frames_rejects_frames_without_times(tmp_path, monkeypatch):
    """Files but no matching times in the log (an ffmpeg that prints no metadata):
    a failed attempt -- the frames would land in the wrong moments otherwise."""
    def fake_run(cmd, **kwargs):
        for i in (1, 2):
            open(cmd[-1].replace("%06d", f"{i:06d}"), "wb").close()
        return subprocess.CompletedProcess(cmd, 0, b"", _log_for(1))

    monkeypatch.setattr(vm.subprocess, "run", fake_run)
    out = tmp_path / "out"
    assert vm._extract_frames("/v.mp4", 2.0, 0.3, str(out)) == [] and os.listdir(out) == []


def test_extract_frames_returns_empty_when_every_attempt_raises(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(vm.subprocess, "run", fake_run)
    assert vm._extract_frames("/v.mp4", 2.0, 0.3, str(tmp_path / "out")) == []


@pytest.mark.parametrize("value, expected", [(None, True), ("true", True), ("false", False), ("FALSE", False)])
def test_scene_cuts_switch(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("VIDEO_MODERATION_SCENE_CUTS", raising=False)
    else:
        monkeypatch.setenv("VIDEO_MODERATION_SCENE_CUTS", value)
    assert vm._scene_cuts() is expected


@pytest.mark.parametrize("value, expected", [(None, 0.3), ("0.2", 0.2), ("0", 0.3), ("1.5", 0.3), ("x", 0.3)])
def test_scene_threshold_setting(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("VIDEO_MODERATION_SCENE_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("VIDEO_MODERATION_SCENE_THRESHOLD", value)
    assert vm._float_env("VIDEO_MODERATION_SCENE_THRESHOLD", 0.3, 0.01, 1.0) == expected


@pytest.mark.parametrize("value, expected", [(None, 2.0), ("1", 1.0), ("0.5", 0.5), ("0", 2.0), ("-3", 2.0),
                                             ("abc", 2.0), ("600", 2.0)])
def test_sample_every_setting(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("VIDEO_MODERATION_SAMPLE_EVERY_SEC", raising=False)
    else:
        monkeypatch.setenv("VIDEO_MODERATION_SAMPLE_EVERY_SEC", value)
    assert vm._float_env("VIDEO_MODERATION_SAMPLE_EVERY_SEC", 2.0, 0.1, 60.0) == expected


@pytest.mark.asyncio
async def test_unknown_model_raises_whatever_the_fail_mode(monkeypatch):
    monkeypatch.setenv("VIDEO_MODERATION_MODEL", "gpt-vision")
    with pytest.raises(vm.ModerationUnavailable, match="VIDEO_MODERATION_MODEL"):
        await vm.moderate_moments(moments(1))


@pytest.mark.parametrize("value, expected", [(None, False), ("false", False), ("true", True), (" TRUE ", True)])
def test_enabled_switch(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("VIDEO_MODERATION_ENABLED", raising=False)
    else:
        monkeypatch.setenv("VIDEO_MODERATION_ENABLED", value)
    assert vm.enabled() is expected


# ── video_chunking_service calls it ───────────────────────────────────────────


@pytest.fixture
def chunking(monkeypatch):
    """video_chunking_service's kafka_handler, imported with its own config.py
    (the orchestrator's same-named modules are set aside and restored)."""
    names = ("config", "kafka_handler", "chunker", "frame_selector", "video_moderation")
    saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
    monkeypatch.syspath_prepend(_CHUNKING_DIR)
    import kafka_handler as kh
    yield kh
    for n in names:
        sys.modules.pop(n, None)
    sys.modules.update(saved)


class SentProducer:
    def __init__(self):
        self.sent = []

    async def send_and_wait(self, topic, value):
        self.sent.append((topic, value))


def _chunking_request() -> dict:
    return {"correlation_id": "corr-1", "source_service": "video_orchestrator_service",
            "payload": {"request_id": "req-1", "parameters": {"video_file_path": "/v.mp4"}}}


@pytest.fixture
def chunking_stubs(chunking, monkeypatch, tmp_path):
    monkeypatch.setattr(chunking, "normalize_video", lambda path, rid: str(tmp_path / "n.mp4"))
    monkeypatch.setattr(chunking, "split_into_moments",
                        lambda path, rid: [{"moment_index": 0, "start_time": 0.0, "end_time": 10.0}])
    monkeypatch.setattr(chunking, "select_keyframes", lambda m: ["/kf/0.jpg"])
    return chunking


@pytest.mark.asyncio
async def test_chunking_adds_the_rollup_when_enabled(chunking_stubs, monkeypatch):
    kh = chunking_stubs
    monkeypatch.setenv("VIDEO_MODERATION_ENABLED", "true")

    async def fake_moderate(ms):
        ms[0]["moderation_flags"] = {"verdict": "safe"}
        return {"verdict": "safe", "model": "shieldgemma-2"}

    monkeypatch.setattr(kh.video_moderation, "moderate_moments", fake_moderate)
    producer = SentProducer()
    await kh.VideoChunkingKafkaHandler().process_message(_chunking_request(), producer)
    [(topic, msg)] = producer.sent
    results = msg["payload"]["results"]
    assert topic == "video-chunking-results"
    assert results["moderation_flags"] == {"verdict": "safe", "model": "shieldgemma-2"}
    assert results["moments"][0]["moderation_flags"] == {"verdict": "safe"}


@pytest.mark.asyncio
async def test_chunking_skips_moderation_when_disabled(chunking_stubs, monkeypatch):
    kh = chunking_stubs
    monkeypatch.delenv("VIDEO_MODERATION_ENABLED", raising=False)

    async def must_not_run(ms):
        raise AssertionError("moderation ran while disabled")

    monkeypatch.setattr(kh.video_moderation, "moderate_moments", must_not_run)
    producer = SentProducer()
    await kh.VideoChunkingKafkaHandler().process_message(_chunking_request(), producer)
    [(_, msg)] = producer.sent
    assert msg["payload"]["results"]["moderation_flags"] is None
    assert "moderation_flags" not in msg["payload"]["results"]["moments"][0]


@pytest.mark.asyncio
async def test_chunking_fails_the_job_in_closed_mode(chunking_stubs, monkeypatch):
    kh = chunking_stubs
    monkeypatch.setenv("VIDEO_MODERATION_ENABLED", "true")

    async def unavailable(ms):
        raise kh.video_moderation.ModerationUnavailable("2 frame(s) got no verdict")

    monkeypatch.setattr(kh.video_moderation, "moderate_moments", unavailable)
    producer = SentProducer()
    await kh.VideoChunkingKafkaHandler().process_message(_chunking_request(), producer)
    topics = sorted(t for t, _ in producer.sent)
    assert topics == ["video-chunking-dlq", "video-chunking-results"]
    assert all(m["message_type"] == "ERROR" and "no verdict" in m["payload"]["error"] for _, m in producer.sent)


def test_chunking_consumer_allows_long_moderation(chunking):
    assert chunking.KAFKA_MAX_POLL_INTERVAL_MS == 3600000


# ── the orchestrator writes the rollup and passes it on ───────────────────────


class FakeRedisState:
    def __init__(self):
        self.states = {}

    def set_state(self, cid, state):
        self.states[cid] = dict(state)
        return True

    def get_state(self, cid):
        s = self.states.get(cid)
        return dict(s) if s is not None else None

    def update_state(self, cid, updates):
        if cid not in self.states:
            return False
        self.states[cid].update(updates)
        return True


class FakeProducer:
    def __init__(self):
        self.sent = []

    async def send(self, topic, value):
        self.sent.append((topic, value))


@pytest.fixture
def orch(handler):
    handler.redis_manager = FakeRedisState()
    handler.redis_manager.set_state("corr-1", {
        "original_message": make_request_message(asset_path="u1/videos/clip"),
        "temp_file_path": "/tmp/x.mp4", "step": "chunking", "branch_id": "b1",
    })
    return handler


def _chunking_result(rollup: dict | None) -> dict:
    ms = [{"moment_index": 0, "start_time": 0.0, "end_time": 10.0, "keyframe_paths": [],
           "moderation_flags": {"verdict": "unsafe"}}]
    return {"correlation_id": "corr-1",
            "payload": {"results": {"moments": ms, "moment_count": 1, "moderation_flags": rollup}}}


@pytest.mark.asyncio
async def test_orchestrator_stores_the_rollup_and_writes_it_to_module_04(orch):
    rollup = {"verdict": "unsafe", "model": "shieldgemma-2"}
    state = orch.redis_manager.get_state("corr-1")
    await orch.handle_chunking_result(_chunking_result(rollup), FakeProducer(), state, "corr-1")
    saved = orch.redis_manager.get_state("corr-1")
    assert saved["moderation_flags"] == rollup
    assert saved["chunking_moments"][0]["moderation_flags"] == {"verdict": "unsafe"}
    assert orch.seaweedfs_client.moderation_calls == [("u1", "b1", "u1/videos/clip", "v1", rollup)]


@pytest.mark.asyncio
async def test_orchestrator_writes_nothing_when_moderation_is_off(orch):
    state = orch.redis_manager.get_state("corr-1")
    await orch.handle_chunking_result(_chunking_result(None), FakeProducer(), state, "corr-1")
    assert orch.redis_manager.get_state("corr-1")["moderation_flags"] is None
    assert orch.seaweedfs_client.moderation_calls == []


@pytest.mark.asyncio
async def test_orchestrator_passes_the_rollup_to_the_indexer(orch):
    rollup = {"verdict": "safe"}
    orch.redis_manager.update_state("corr-1", {"moderation_flags": rollup, "chunking_moments": []})
    producer = FakeProducer()
    state = orch.redis_manager.get_state("corr-1")
    message = {"correlation_id": "corr-1", "payload": {"results": {"parameters": {"moments": []}}}}
    await orch.handle_contextualize_result(message, producer, state, "corr-1")
    [(topic, request)] = producer.sent
    assert topic == "video-indexer-requests"
    assert request["payload"]["parameters"]["moderation_flags"] == rollup


# ── the indexer puts the flags on what goes to Module 25 ──────────────────────


def test_moment_flags_are_copied_onto_the_contextualized_moments():
    chunking_moments = [{"moment_index": 0, "moderation_flags": {"verdict": "safe"}},
                        {"moment_index": 1, "moderation_flags": {"verdict": "unsafe"}},
                        {"moment_index": 2}]
    contextualized = [{"moment_index": 1, "contextual_prefix": "p"}, {"moment_index": 0}, {"moment_index": 2}]
    out = fields.with_moment_moderation(contextualized, chunking_moments)
    assert [m.get("moderation_flags") for m in out] == [{"verdict": "unsafe"}, {"verdict": "safe"}, None]
    assert out[0]["contextual_prefix"] == "p"
    assert "moderation_flags" not in contextualized[0]               # inputs untouched


def test_moments_unchanged_when_moderation_is_off():
    ms = [{"moment_index": 0}]
    assert fields.with_moment_moderation(ms, [{"moment_index": 0}]) is ms


def test_rollup_goes_on_the_summary_entry_only():
    entries = [{"id": "s", "payload": {"embedding_type": "summary", "summary": "x"}},
               {"id": "t", "payload": {"embedding_type": "text"}}]
    out = fields.with_summary_moderation(entries, {"verdict": "unsafe"})
    assert out[0]["payload"]["moderation_flags"] == {"verdict": "unsafe"}
    assert "moderation_flags" not in out[1]["payload"]
    assert "moderation_flags" not in entries[0]["payload"]             # inputs untouched


def test_summary_unchanged_without_a_rollup():
    entries = [{"payload": {"embedding_type": "summary"}}]
    assert fields.with_summary_moderation(entries, None) is entries
