# Solution: keep the last n tool results verbatim, stub out the older ones.
import json
import re

from harness.exercises import check_03


def keep_last_results(messages, n=2):
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    old = set(tool_idx[:-n]) if n else set(tool_idx)
    out = []
    for i, m in enumerate(messages):
        m = dict(m)                                   # never mutate the caller's list
        if i in old:
            handle = re.search(r'"handle": "(r\d+)"', m["content"])
            m["content"] = ("[older result elided"
                            + (f"; full data in {handle.group(1)}, use expand_result" if handle else "")
                            + "]")
        out.append(m)
    return out


check_03(keep_last_results)
