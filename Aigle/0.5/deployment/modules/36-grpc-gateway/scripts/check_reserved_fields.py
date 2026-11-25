#!/usr/bin/env python3
"""CI contract check: any proto field number/name present in a baseline
version but absent from the current version must be `reserved` in the
current version (never silently deleted and left open for reuse, which
would corrupt binary compatibility with any peer still on the old
contract).

Not the same check as `buf breaking` (which flags ANY field deletion,
reserved or not) -- this implements the narrower policy this repo actually
wants: deleting a field is fine as long as its number and name are
reserved so nothing new can ever claim them.

Usage:
    python scripts/check_reserved_fields.py <baseline.proto> <current.proto>

Also fails when a reservation present in the baseline is gone from the current
version (reusing a reserved number or name).

Exit code 0 if every removed field number/name is properly reserved (or
nothing was removed); 1 and a description of each violation otherwise.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from google.protobuf import descriptor_pb2


def _compile_to_descriptor_set(proto_path: Path) -> descriptor_pb2.FileDescriptorSet:
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "out.bin"
        subprocess.run(
            [
                sys.executable, "-m", "grpc_tools.protoc",
                f"-I{proto_path.parent.parent.parent}",  # .../proto/ -- raptor/v1/raptor.proto's package root
                f"--descriptor_set_out={out_path}",
                str(proto_path),
            ],
            check=True, capture_output=True, text=True,
        )
        fds = descriptor_pb2.FileDescriptorSet()
        fds.ParseFromString(out_path.read_bytes())
        return fds


def _messages_by_name(fds: descriptor_pb2.FileDescriptorSet) -> dict[str, descriptor_pb2.DescriptorProto]:
    messages = {}
    for file in fds.file:
        for msg in file.message_type:
            messages[msg.name] = msg
    return messages


def _reserved_numbers(msg: descriptor_pb2.DescriptorProto) -> set:
    numbers = set()
    for r in msg.reserved_range:
        # DescriptorProto.ReservedRange.end is exclusive, per protobuf's own convention.
        numbers.update(range(r.start, r.end))
    return numbers


def check(baseline_path: Path, current_path: Path) -> list[str]:
    baseline_messages = _messages_by_name(_compile_to_descriptor_set(baseline_path))
    current_messages = _messages_by_name(_compile_to_descriptor_set(current_path))

    violations: list[str] = []
    for msg_name, baseline_msg in baseline_messages.items():
        current_msg = current_messages.get(msg_name)
        if current_msg is None:
            # Whole message removed -- a much bigger break than a field
            # number reuse risk; out of scope for this check specifically.
            continue

        baseline_fields: dict[int, str] = {f.number: f.name for f in baseline_msg.field}
        current_field_numbers = {f.number for f in current_msg.field}
        current_field_names = {f.name for f in current_msg.field}
        reserved_numbers = _reserved_numbers(current_msg)
        reserved_names = set(current_msg.reserved_name)

        for number, name in baseline_fields.items():
            if number in current_field_numbers:
                continue  # field still present (possibly renamed -- a separate concern)
            if number not in reserved_numbers:
                violations.append(
                    f"{msg_name}: field {number} ({name!r}) was removed but its number is not "
                    f"in a `reserved` range -- add `reserved {number};`"
                )
            if name not in current_field_names and name not in reserved_names:
                violations.append(
                    f"{msg_name}: field {number} ({name!r}) was removed but its name is not "
                    f"reserved -- add `reserved \"{name}\";`"
                )

        # A reservation is permanent. Dropping one (and, worse, handing the number or name to a
        # new field) lets a peer still on the old contract decode the new field as the old one.
        current_fields_by_number = {f.number: f.name for f in current_msg.field}
        for number in sorted(_reserved_numbers(baseline_msg) - reserved_numbers):
            reused = f" and field {current_fields_by_number[number]!r} now uses it" if number in current_fields_by_number else ""
            violations.append(f"{msg_name}: the reservation of number {number} was removed{reused}")
        for name in sorted(set(baseline_msg.reserved_name) - reserved_names):
            reused = " and a field now uses it" if name in current_field_names else ""
            violations.append(f"{msg_name}: the reservation of name {name!r} was removed{reused}")
    return violations


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    baseline_path, current_path = Path(sys.argv[1]), Path(sys.argv[2])
    violations = check(baseline_path, current_path)
    if violations:
        print("Reserved-field check FAILED:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("Reserved-field check passed: no removed field numbers/names left unreserved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
