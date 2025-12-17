from typing import List


def read_nonempty_lines(path:str) -> List[str]:
    with open(path, "r") as r:
        d = r.read().strip()
    return [v for v in d.splitlines(keepends=False) if len(v) > 0]
