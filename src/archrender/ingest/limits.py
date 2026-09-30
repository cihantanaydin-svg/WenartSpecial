"""Exec a command under hard resource limits.

    python -m archrender.ingest.limits --mem-mb 6144 --cpu-s 900 --fsize-mb 4096 -- CMD [ARGS…]

Used to run every parser of untrusted input (and external tools such as LibreDWG and libheif) in a
fresh process. The limits are applied in this launcher and then ``exec``'d into the target, which
is safe in a multi-threaded parent (unlike ``subprocess``'s ``preexec_fn``).
"""

from __future__ import annotations

import argparse
import os
import resource
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="archrender.ingest.limits")
    ap.add_argument("--mem-mb", type=int, required=True)
    ap.add_argument("--cpu-s", type=int, required=True)
    ap.add_argument("--fsize-mb", type=int, required=True)
    ap.add_argument("--nofile", type=int, default=1024)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("missing command after --")
    mem = args.mem_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    # soft limit → SIGXCPU (reported as a limit), hard limit 5 s later → SIGKILL
    resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_s, args.cpu_s + 5))
    fsize = args.fsize_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (args.nofile, args.nofile))
    os.execvp(cmd[0], cmd)  # noqa: S606 - command list built by archrender.ingest.sandbox
    return 127  # unreachable: execvp only returns by raising


if __name__ == "__main__":
    sys.exit(main())
