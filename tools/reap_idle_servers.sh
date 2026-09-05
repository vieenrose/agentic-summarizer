#!/usr/bin/env bash
# Kill MY idle llama-server processes — ones holding GPU memory at 0% utilisation.
#
# Written after leaving a finished 24 GB teacher parked on GPU0 twice in one session while
# the other card did all the work. A served model does not exit when the job that needed it
# finishes, and nothing notices: utilisation reads 0% and the memory stays reserved.
#
# NEVER touches another user's processes. This workstation is shared, and the other tenant's
# llama-server is always present; `stat -c %U` on /proc/<pid> is the ownership check, and
# `pgrep -x llama-server` is an exact-name match so it cannot match this script's own shell
# (CLAUDE.md trap 11: `pkill -f <pattern>` killed the agent's session three times).
set -u
me=$(id -un)
for pid in $(pgrep -x llama-server); do
    owner=$(stat -c %U "/proc/$pid" 2>/dev/null) || continue
    [ "$owner" = "$me" ] || continue
    util=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -cx "$pid")
    [ "$util" -gt 0 ] || continue
    model=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -oP '(?<=-m )\S+' | head -1)
    if ss -tnp 2>/dev/null | grep -q "pid=$pid,"; then
        echo "keep  pid=$pid (active connection) $(basename "${model:-?}")"
    else
        echo "reap  pid=$pid $(basename "${model:-?}")"
        kill -9 "$pid" 2>/dev/null
    fi
done
