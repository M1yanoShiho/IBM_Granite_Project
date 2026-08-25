#!/bin/sh
# Pull, then say what the pull took away.
#
#   scripts/pull_and_report.sh                  # same args as git pull --rebase
#   git config alias.up '!scripts/pull_and_report.sh'   # then: git up
#
# A merge deletes cleanly. There is no conflict marker for "this file is gone now", so a pull
# can remove something the working tree depends on and say nothing -- which is how
# M0_PROTOCOL_FREEZE.md (1562 lines) and TRAINING_PLAN.md left this checkout on 2026-08-09
# without anyone noticing until hours later.
#
# The obvious answer is "check after every pull", and this repo now has four separate prose
# disciplines that were skipped within a day of being written down. So it is a script.
#
# post-merge would have been the natural hook, but it does not fire for `git pull --rebase`,
# which is what this branch actually uses. Wrapping the pull covers both.
#
# Nothing is lost either way: the previous HEAD is printed, and `git show <sha>:<path>` brings
# any of it back. The point is only to know at the moment it happens.
set -eu

before=$(git rev-parse HEAD)
git pull --rebase "$@"
after=$(git rev-parse HEAD)

if [ "$before" = "$after" ]; then
    echo "[pull] already up to date."
    exit 0
fi

# --diff-filter=D is deletions only. A rename shows as a delete plus an add, so a file that
# merely moved is reported too -- a false alarm costs one glance, a missed deletion costs a day.
deleted=$(git diff --diff-filter=D --name-only "$before" "$after" || true)

if [ -z "$deleted" ]; then
    echo "[pull] $before -> $after, nothing deleted."
    exit 0
fi

echo
echo "!! this pull DELETED $(printf '%s\n' "$deleted" | wc -l | tr -d ' ') file(s):"
printf '%s\n' "$deleted" | sed 's/^/     /'
echo
echo "   was at: $before"
echo "   recover: git show $before:<path> > <path>"
echo
