"""Check build.py's GPU registry (Module.gpu) and the --mem-usage helpers,
without invoking Docker or a real nvidia-smi."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def builder(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[2] / "build.py"
    path = tmp_path / "build.py"
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / ".env").write_text("RAPTOR_VERSION=unit-test\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("raptor_test_builder_gpu", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_module_23_is_registered_as_a_gpu_module(builder):
    """Regression test: 23-guardrail-service's docker-compose.yml
    unconditionally reserves an nvidia GPU device, but the registry's own
    `gpu` flag had drifted out of sync (defaulted to False) -- the same
    class of bug already known for modules 04/18/25/26. --cpu-only is
    supposed to skip every GPU module; it silently didn't skip this one."""
    assert builder.MODULE_BY_ID["23"].gpu is True


def test_gpu_devices_parses_nvidia_smi_csv_output(builder, monkeypatch):
    csv_output = (
        b"0, GPU-aaaa, NVIDIA GeForce RTX 4060 Ti, 16380, 610, 15338\n"
        b"1, GPU-bbbb, NVIDIA A100, 81920, 2048, 79872\n"
    )
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *a, **k: csv_output)
    devices = builder._gpu_devices()
    assert devices == [
        {"index": "0", "uuid": "GPU-aaaa", "name": "NVIDIA GeForce RTX 4060 Ti",
         "total_mib": 16380, "used_mib": 610, "free_mib": 15338},
        {"index": "1", "uuid": "GPU-bbbb", "name": "NVIDIA A100",
         "total_mib": 81920, "used_mib": 2048, "free_mib": 79872},
    ]


def test_gpu_devices_returns_empty_list_when_nvidia_smi_is_unavailable(builder, monkeypatch):
    def raise_missing(*_args, **_kwargs):
        raise FileNotFoundError("no nvidia-smi")

    monkeypatch.setattr(builder.subprocess, "check_output", raise_missing)
    assert builder._gpu_devices() == []


def test_gpu_devices_falls_back_to_a_query_without_uuid_when_the_combined_query_is_empty(builder, monkeypatch):
    """Confirmed needed on a real GB10/DGX Spark host: the combined
    index,uuid,name,memory.* query came back with nothing parseable, even
    though nvidia-smi itself works fine there (--query-compute-apps
    succeeds on the same host) -- --gpu-status was wrongly claiming
    "nvidia-smi not available" while nvitop showed real, heavy usage."""
    def fake_check_output(cmd, **_kwargs):
        if "uuid" in cmd[1]:
            return b""  # the real failure mode: nothing parseable back
        return b"0, GB10, 124928, 69172, 55756\n"

    monkeypatch.setattr(builder.subprocess, "check_output", fake_check_output)
    devices = builder._gpu_devices()
    assert devices == [
        {"index": "0", "uuid": None, "name": "GB10", "total_mib": 124928, "used_mib": 69172, "free_mib": 55756},
    ]


def test_gpu_devices_tolerates_a_non_numeric_memory_field_instead_of_dropping_the_device(builder, monkeypatch):
    """Another unified-memory quirk this codebase already works around
    elsewhere (gpu_free_bytes()): don't let one field nvidia-smi can't
    report cleanly (e.g. "[N/A]") silently drop the whole device. Two
    devices here on purpose -- the single-device /proc/meminfo fallback
    (tested separately below) only applies when there's exactly one."""
    csv_output = (
        b"0, GPU-aaaa, GB10, [N/A], 69172, 55756\n"
        b"1, GPU-bbbb, GB10, 49140, 1024, 48116\n"
    )
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *a, **k: csv_output)
    devices = builder._gpu_devices()
    assert devices == [
        {"index": "0", "uuid": "GPU-aaaa", "name": "GB10", "total_mib": None, "used_mib": 69172, "free_mib": 55756},
        {"index": "1", "uuid": "GPU-bbbb", "name": "GB10", "total_mib": 49140, "used_mib": 1024, "free_mib": 48116},
    ]


def test_gpu_devices_falls_back_to_proc_meminfo_on_a_single_unified_memory_device(builder, monkeypatch):
    """The real GB10 bug, continued: even the uuid-less fallback query's
    memory.total/used/free came back non-numeric there -- nvidia-smi
    genuinely can't report a VRAM figure on this chip. Confirmed live that
    /proc/meminfo and nvitop's own GPU-memory gauge agree exactly there
    (no separate VRAM pool exists to query), so fall back to it instead of
    showing "N/A" for a number that's actually knowable -- but only when
    there's exactly one device; with several GPUs, host RAM isn't
    attributable to any one of them."""
    def fake_check_output(cmd, **_kwargs):
        if "uuid" in cmd[1]:
            return b""
        return b"0, GB10, [N/A], [N/A], [N/A]\n"

    monkeypatch.setattr(builder.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(builder, "_host_memory_mib", lambda: (124518, 69172, 55346))

    devices = builder._gpu_devices()
    assert devices == [
        {"index": "0", "uuid": None, "name": "GB10",
         "total_mib": 124518, "used_mib": 69172, "free_mib": 55346, "mem_is_host_approx": True},
    ]


def test_gpu_devices_leaves_na_alone_when_proc_meminfo_is_also_unreadable(builder, monkeypatch):
    def fake_check_output(cmd, **_kwargs):
        if "uuid" in cmd[1]:
            return b""
        return b"0, GB10, [N/A], [N/A], [N/A]\n"

    monkeypatch.setattr(builder.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(builder, "_host_memory_mib", lambda: None)

    devices = builder._gpu_devices()
    assert devices == [
        {"index": "0", "uuid": None, "name": "GB10", "total_mib": None, "used_mib": None, "free_mib": None},
    ]


def test_host_memory_mib_parses_proc_meminfo(builder, monkeypatch):
    meminfo = (
        "MemTotal:       127512576 kB\n"
        "MemFree:          2048000 kB\n"
        "MemAvailable:    56688640 kB\n"
    )
    monkeypatch.setattr(builder.Path, "read_text", lambda self: meminfo)
    assert builder._host_memory_mib() == (124524, 124524 - 55360, 55360)


def test_docker_stats_mem_mib_parses_various_units(builder, monkeypatch):
    stats_output = (
        b"abc123def456\t512MiB / 62.5GiB\n"
        b"c1\t1.5GiB / 62.5GiB\n"
        b"c2\t2048KiB / 62.5GiB\n"
    )
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *a, **k: stats_output)
    assert builder._docker_stats_mem_mib() == {"abc123def456": 512, "c1": 1536, "c2": 2}


def test_docker_stats_mem_mib_returns_empty_dict_when_docker_is_unavailable(builder, monkeypatch):
    def raise_missing(*_args, **_kwargs):
        raise FileNotFoundError("no docker")
    monkeypatch.setattr(builder.subprocess, "check_output", raise_missing)
    assert builder._docker_stats_mem_mib() == {}


def test_cgroup_mem_mib_reads_memory_current_per_container(builder, monkeypatch):
    """The fast path: confirmed live this is ~0.01s for a whole-host walk
    vs. `docker stats --no-stream`'s own ~2.45s for ~100 containers."""
    cid_a = "a1" + "0" * 62
    cid_b = "b2" + "0" * 62
    scope_a = builder.Path(f"/sys/fs/cgroup/system.slice/docker-{cid_a}.scope")
    scope_b = builder.Path(f"/sys/fs/cgroup/system.slice/docker-{cid_b}.scope")
    monkeypatch.setattr(builder.Path, "rglob", lambda self, pattern: iter([scope_a, scope_b]))
    bytes_by_path = {
        scope_a / "memory.current": "41943040",   # 40 MiB
        scope_b / "memory.current": "104857600",  # 100 MiB
    }
    monkeypatch.setattr(builder.Path, "read_text", lambda self, *a, **k: bytes_by_path[self])

    assert builder._cgroup_mem_mib() == {cid_a: 40, cid_b: 100}


def test_cgroup_mem_mib_returns_empty_dict_when_cgroup_is_unreadable(builder, monkeypatch):
    def raise_oserror(self, pattern):
        raise OSError("no such path")
    monkeypatch.setattr(builder.Path, "rglob", raise_oserror)
    assert builder._cgroup_mem_mib() == {}


def test_container_mem_mib_prefers_the_fast_cgroup_path_over_docker_stats(builder, monkeypatch):
    monkeypatch.setattr(builder, "_cgroup_mem_mib", lambda: {"a1": 40})

    def fail_if_called():
        raise AssertionError("docker stats must not run when cgroup data is already available")
    monkeypatch.setattr(builder, "_docker_stats_mem_mib", fail_if_called)

    assert builder._container_mem_mib() == {"a1": 40}


def test_container_mem_mib_falls_back_to_docker_stats_when_cgroup_finds_nothing(builder, monkeypatch):
    monkeypatch.setattr(builder, "_cgroup_mem_mib", dict)
    monkeypatch.setattr(builder, "_docker_stats_mem_mib", lambda: {"a1": 40})
    assert builder._container_mem_mib() == {"a1": 40}


def test_raptor_host_mem_mib_sums_across_every_module_not_just_gpu_ones(builder, monkeypatch):
    """The whole point of this figure: on a unified-memory host, VRAM and
    host RAM are the same pool, so a CPU-only module's containers (e.g.
    03's Postgres) count toward Raptor's real memory footprint just as
    much as a GPU module's do."""
    def fake_container_rows(self):
        if self.id == "23":
            return [("guardrail_service", "running", "Up 1 hour", "c1")]
        if self.id == "03":
            return [("postgres", "running", "Up 1 hour", "c2")]
        return []

    monkeypatch.setattr(builder.Module, "container_rows", fake_container_rows)

    gpu_module = builder.MODULE_BY_ID["23"]
    cpu_module = builder.MODULE_BY_ID["03"]
    assert gpu_module.gpu is True
    assert cpu_module.gpu is False

    total = builder._raptor_host_mem_mib([gpu_module, cpu_module], {"c1": 2048, "c2": 4096})
    assert total == 2048 + 4096


def test_raptor_host_mem_mib_skips_image_only_modules(builder, monkeypatch):
    def fail_if_called(self):
        raise AssertionError("container_rows() must not be called for an image-only module")
    monkeypatch.setattr(builder.Module, "container_rows", fail_if_called)

    image_module = builder.Module(
        id="08", name="08-media-worker", description="", steps=[], deps=[],
        health_images=["raptor/media-worker:unit-test"],
    )
    assert builder._raptor_host_mem_mib([image_module], {"c1": 2048}) == 0


def test_gpu_mem_by_container_and_device_groups_by_gpu_uuid(builder, monkeypatch, tmp_path):
    csv_output = b"GPU-aaaa, 111, 4096\nGPU-bbbb, 111, 512\nGPU-aaaa, 222, 1024\n"
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *a, **k: csv_output)

    cgroup_by_pid = {
        "111": f"0::/docker-{'a1' + '0' * 62}.scope\n",
        "222": f"0::/docker-{'b2' + '0' * 62}.scope\n",
    }

    def fake_read_text(self, *_a, **_k):
        pid = self.parts[2]  # /proc/<pid>/cgroup
        return cgroup_by_pid[pid]

    monkeypatch.setattr(builder.Path, "read_text", fake_read_text)

    result = builder._gpu_mem_by_container_and_device()
    assert result == {
        "a1" + "0" * 62: {"GPU-aaaa": 4096, "GPU-bbbb": 512},
        "b2" + "0" * 62: {"GPU-aaaa": 1024},
    }


def test_gpu_module_usage_only_marks_containers_that_use_gpu_and_labels_the_device(builder, monkeypatch):
    devices = [
        {"index": "0", "uuid": "GPU-aaaa", "name": "A6000", "total_mib": 49140, "used_mib": 4096, "free_mib": 45044},
        {"index": "1", "uuid": "GPU-bbbb", "name": "A6000", "total_mib": 49140, "used_mib": 1024, "free_mib": 48116},
    ]
    monkeypatch.setattr(builder, "_gpu_mem_by_container_and_device", lambda: {
        "a1" + "0" * 62: {"GPU-aaaa": 4096},
    })

    def fake_container_rows(self):
        if self.id == "23":
            return [
                ("guardrail_service", "running", "Up 2 hours", "a1"),
                ("guardrail-init", "exited", "Exited (0) 2 hours ago", "b2"),
            ]
        return []

    monkeypatch.setattr(builder.Module, "container_rows", fake_container_rows)

    cpu_module = builder.Module(id="99", name="99-cpu-only", description="", steps=[], deps=[])
    gpu_module = builder.Module(id="23", name="23-guardrail-service", description="", steps=[], deps=[])
    gpu_module.gpu = True

    usage = builder._gpu_module_usage([cpu_module, gpu_module], devices)

    assert len(usage) == 1  # the CPU-only module is skipped entirely
    row = usage[0]
    assert row["module"] is gpu_module
    assert row["module_mib"] == 4096
    assert row["containers"] == [
        ("guardrail_service", "running", "Up 2 hours", 4096, {"0": 4096}),
        ("guardrail-init", "exited", "Exited (0) 2 hours ago", None, {}),
    ]


def test_gpu_module_usage_attributes_correctly_on_a_single_gpu_host_even_if_the_uuid_does_not_match(builder, monkeypatch):
    """The real GB10 bug: --query-compute-apps's gpu_uuid for a process did
    not match any uuid _gpu_devices() returned (which had fallen back to
    the uuid-less query above) -- previously this made the per-container
    label leak the raw uuid doubled up as "GPUGPU-...". On a single-GPU
    host there's no actual ambiguity about which device it's on --
    attribute to that one device regardless of the uuid mismatch."""
    devices = [{"index": "0", "uuid": None, "name": "GB10", "total_mib": 124928, "used_mib": 69172, "free_mib": 55756}]
    monkeypatch.setattr(builder, "_gpu_mem_by_container_and_device",
                         lambda: {"a1" + "0" * 62: {"GPU-e02d1e13-1356-fb97-2e77-fe3ddf8745e8": 4522}})
    monkeypatch.setattr(builder.Module, "container_rows",
                         lambda self: [("personal-db-service", "running", "Up 2 hours (healthy)", "a1")]
                         if self.id == "25" else [])

    usage = builder._gpu_module_usage([builder.MODULE_BY_ID["25"]], devices)

    assert usage[0]["containers"] == [
        ("personal-db-service", "running", "Up 2 hours (healthy)", 4522, {"0": 4522}),
    ]


def test_print_mem_usage_does_not_double_prefix_a_raw_uuid_fallback_label(builder, monkeypatch, capsys):
    """Multi-GPU host where the uuid join genuinely fails for one
    container (the single-GPU shortcut above doesn't apply) -- the label
    must stay a plain uuid, never "GPUGPU-..."."""
    monkeypatch.setattr(builder, "_gpu_devices", lambda: [
        {"index": "0", "uuid": "GPU-aaaa", "name": "A6000", "total_mib": 49140, "used_mib": 4096, "free_mib": 45044},
        {"index": "1", "uuid": "GPU-bbbb", "name": "A6000", "total_mib": 49140, "used_mib": 1024, "free_mib": 48116},
    ])
    monkeypatch.setattr(builder, "_gpu_mem_by_container_and_device",
                         lambda: {"c1" + "0" * 62: {"GPU-unknown-device": 2048}})
    monkeypatch.setattr(builder, "_bulk_container_rows",
                         lambda modules: {"23": [("guardrail_service", "running", "Up 2 hours", "c1")]})

    builder.print_mem_usage([builder.MODULE_BY_ID["23"]])

    out = builder.strip_ansi(capsys.readouterr().out)
    assert "GPUGPU" not in out
    assert "GPU-unknown-device" in out


def test_gpu_module_usage_treats_an_image_only_module_as_having_no_containers(builder, monkeypatch):
    """08-media-worker has no docker-compose.yml of its own (health_images,
    not health_containers) -- it should be listed, but never call
    container_rows() or claim "(not running)", which would misleadingly
    imply it could be running standalone."""
    monkeypatch.setattr(builder, "_gpu_mem_by_container_and_device", dict)

    def fail_if_called(self):
        raise AssertionError("container_rows() must not be called for an image-only module")

    monkeypatch.setattr(builder.Module, "container_rows", fail_if_called)

    image_module = builder.Module(
        id="08", name="08-media-worker", description="", steps=[], deps=[],
        health_images=["raptor/media-worker:unit-test"],
    )
    image_module.gpu = True

    usage = builder._gpu_module_usage([image_module], [])
    assert usage == [{"module": image_module, "containers": None, "module_mib": 0}]


def test_print_mem_usage_runs_without_a_terminal_or_nvidia_smi(builder, monkeypatch, capsys):
    """Smoke test: must not crash when nvidia-smi and docker are both
    unavailable (e.g. a CI runner with no GPU)."""
    monkeypatch.setattr(builder, "_gpu_devices", list)
    monkeypatch.setattr(builder, "_gpu_mem_by_container_and_device", dict)
    monkeypatch.setattr(builder, "_bulk_container_rows", lambda modules: {})
    monkeypatch.setattr(builder, "_container_mem_mib", dict)

    builder.print_mem_usage()

    out = builder.strip_ansi(capsys.readouterr().out)
    assert "nvidia-smi not available" in out
    assert "23" in out and "guardrail-service" in out
    assert "Total VRAM used by Raptor containers: 0 MiB" in out


def test_print_mem_usage_prints_the_grand_total_before_the_device_and_module_tables(builder, monkeypatch, capsys):
    """The total is the one number most people reach for this flag to get
    -- it should be the first thing printed, not buried after a wall of
    per-device/per-container detail."""
    monkeypatch.setattr(builder, "_gpu_devices", lambda: [
        {"index": "0", "uuid": "GPU-aaaa", "name": "NVIDIA RTX A6000",
         "total_mib": 49140, "used_mib": 18658, "free_mib": 30010},
    ])
    monkeypatch.setattr(builder, "_gpu_mem_by_container_and_device",
                         lambda: {"a1" + "0" * 62: {"GPU-aaaa": 4096}})
    monkeypatch.setattr(builder, "_bulk_container_rows",
                         lambda modules: {"23": [("guardrail_service", "running", "Up 2 hours", "a1")]})

    builder.print_mem_usage()

    out = builder.strip_ansi(capsys.readouterr().out)
    total_pos = out.index("Total VRAM used by Raptor containers")
    device_table_pos = out.index("NVIDIA RTX A6000")
    module_table_pos = out.index("23-guardrail-service")
    assert total_pos < device_table_pos < module_table_pos


def test_print_mem_usage_hides_containers_not_using_gpu_and_labels_the_device(builder, monkeypatch, capsys):
    """The actual complaint this fixed: a module with 10 containers where
    only 1-2 touch the GPU used to print all 10, burying the ones that
    matter. Only the active ones should show, tagged with which physical
    GPU they're on, plus a one-line count of the rest."""
    monkeypatch.setattr(builder, "_gpu_devices", lambda: [
        {"index": "0", "uuid": "GPU-aaaa", "name": "A6000", "total_mib": 49140, "used_mib": 4096, "free_mib": 45044},
        {"index": "3", "uuid": "GPU-dddd", "name": "A6000", "total_mib": 49140, "used_mib": 2048, "free_mib": 47092},
    ])
    monkeypatch.setattr(builder, "_gpu_mem_by_container_and_device",
                         lambda: {"c1" + "0" * 62: {"GPU-dddd": 2048}})

    monkeypatch.setattr(builder, "_bulk_container_rows", lambda modules: {"09": [
        ("audio-orchestrator", "running", "Up 23 hours", "00" + "0" * 62),
        ("audio-classifier", "running", "Up 5 hours (healthy)", "c1" + "0" * 62),
        ("audio-indexer", "running", "Up 23 hours", "00" + "0" * 62),
    ]})

    builder.print_mem_usage([builder.MODULE_BY_ID["09"]])

    out = builder.strip_ansi(capsys.readouterr().out)
    assert "audio-classifier" in out
    assert "GPU3" in out
    assert "audio-orchestrator" not in out
    assert "audio-indexer" not in out
    assert "+2 idle" in out


def test_print_mem_usage_adds_a_per_gpu_raptor_column_to_the_device_table(builder, monkeypatch, capsys):
    """Asked for directly: the per-device table should also show how much
    of each GPU's usage is Raptor's own, not just Used/Free/Total."""
    monkeypatch.setattr(builder, "_gpu_devices", lambda: [
        {"index": "0", "uuid": "GPU-aaaa", "name": "A6000", "total_mib": 49140, "used_mib": 4902, "free_mib": 44238},
        {"index": "1", "uuid": "GPU-bbbb", "name": "A6000", "total_mib": 49140, "used_mib": 21516, "free_mib": 27155},
    ])
    monkeypatch.setattr(builder, "_gpu_mem_by_container_and_device", lambda: {
        "c1" + "0" * 62: {"GPU-aaaa": 1202},
        "c2" + "0" * 62: {"GPU-aaaa": 3342, "GPU-bbbb": 358},  # one container spanning two GPUs
    })

    monkeypatch.setattr(builder, "_bulk_container_rows", lambda modules: {"09": [
        ("audio-classifier", "running", "Up 5 hours (healthy)", "c1" + "0" * 62),
        ("audio-recognizer", "running", "Up 5 hours (healthy)", "c2" + "0" * 62),
    ]})

    builder.print_mem_usage([builder.MODULE_BY_ID["09"]])

    out = builder.strip_ansi(capsys.readouterr().out)
    assert "Raptor" in out
    lines = {line.split()[0]: line for line in out.splitlines() if line.strip() and line.strip()[0] in "01"}
    assert "4544 MiB" in lines["0"]   # 1202 + 3342 on GPU0
    assert "358 MiB" in lines["1"]


def test_print_mem_usage_reports_no_gpu_modules_in_a_cpu_only_selection(builder, monkeypatch, capsys):
    cpu_module = builder.MODULE_BY_ID["03"]
    assert cpu_module.gpu is False
    monkeypatch.setattr(builder, "_container_mem_mib", lambda: {"c3": 8192})
    monkeypatch.setattr(builder, "_bulk_container_rows",
                         lambda modules: {"03": [("postgres", "running", "Up 1 hour", "c3")]})

    builder.print_mem_usage([cpu_module])

    out = builder.strip_ansi(capsys.readouterr().out)
    assert "No GPU modules in the given selection." in out
    # A CPU-only selection still gets a host-RAM total -- that's the whole
    # point of widening this past GPU modules.
    assert "Total host RAM (CPU memory) used by Raptor containers: 8192 MiB" in out


def test_print_mem_usage_always_shows_both_vram_and_host_ram_totals_on_a_discrete_gpu_host(builder, monkeypatch, capsys):
    """VRAM and host RAM are two separate pools on a normal (non-unified-
    memory) host -- both totals should appear, independently, with no
    unified-memory caveat note."""
    monkeypatch.setattr(builder, "_gpu_devices", lambda: [
        {"index": "0", "uuid": "GPU-aaaa", "name": "A6000", "total_mib": 49140, "used_mib": 4096, "free_mib": 45044},
    ])
    monkeypatch.setattr(builder, "_gpu_mem_by_container_and_device",
                         lambda: {"c1" + "0" * 62: {"GPU-aaaa": 2048}})
    monkeypatch.setattr(builder, "_container_mem_mib", lambda: {"c1": 6000, "c2": 1000})

    monkeypatch.setattr(builder, "_bulk_container_rows", lambda modules: {
        "23": [("guardrail_service", "running", "Up 1 hour", "c1" + "0" * 62)],
        "03": [("postgres", "running", "Up 1 hour", "c2")],
    })

    builder.print_mem_usage([builder.MODULE_BY_ID["23"], builder.MODULE_BY_ID["03"]])

    out = builder.strip_ansi(capsys.readouterr().out)
    assert "Total VRAM used by Raptor containers: 2048 MiB" in out
    assert "Total host RAM (CPU memory) used by Raptor containers: 7000 MiB" in out
    assert "unified-memory host" not in out
