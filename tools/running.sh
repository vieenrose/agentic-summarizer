#!/usr/bin/env bash
# Is a job of mine actually running? Scans /proc for processes owned by me whose cmdline
# matches, EXCLUDING this script's own pid and its parent shell.
#
# Two failure modes this exists to avoid, both hit repeatedly in this project:
#   * `pgrep -f <pattern>` matches the invoking shell's own command line — it has killed this
#     agent's session three times (CLAUDE.md trap 11) and, more quietly, reported already-dead
#     jobs as still running, which sent a diagnostic down the wrong path.
#   * `pgrep -x python3` matches on PROCESS NAME, and a venv interpreter reports something
#     else entirely (`python3.12`), so an exact-name match silently finds nothing and every
#     job looks finished.
set -u
pat=${1:?usage: running.sh <pattern>}
me=$(id -u)
found=0
for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    [ "$pid" = "$$" ] && continue
    [ "$pid" = "$PPID" ] && continue
    [ -r "$d/cmdline" ] || continue
    [ "$(stat -c %u "$d" 2>/dev/null)" = "$me" ] || continue
    cl=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)
    case "$cl" in
        *running.sh*) continue ;;
        *"$pat"*) echo "running pid=$pid $(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')"; found=1 ;;
    esac
done
[ "$found" = 1 ] || echo "not running"
