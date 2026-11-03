"""Single bounded child stage; guard owned Python writes and C file extents."""

import argparse
import builtins
import fcntl
import io
import os
import resource
import runpy
import sys
from pathlib import Path

TEXT = {".json", ".jsonl", ".xml", ".log", ".txt"}


def usage(root):
    raw = text = 0
    for p in root.rglob("*"):
        if p.is_file() and not p.is_symlink():
            size = p.stat().st_size
            raw += size
            if any(s in TEXT for s in p.suffixes):
                text += size
    return raw, text


def main():
    ap = argparse.ArgumentParser()
    for name in ("root", "raw", "text", "file-max"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    rawmax, textmax, filemax = int(a.raw), int(a.text), int(a.file_max)
    resource.setrlimit(resource.RLIMIT_FSIZE, (filemax, filemax))
    original_open, original_io = builtins.open, io.open
    lock = original_open(root / ".write-credit.lock", "a+b", buffering=0)

    class Checked:
        def __init__(self, file, path):
            self.file, self.path = file, path

        def __getattr__(self, name):
            return getattr(self.file, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def write(self, data):
            size = (
                len(data.encode(self.file.encoding or "utf8"))
                if isinstance(data, str)
                else len(data)
            )
            with_credit = max(0, self.file.tell() + size - self.path.stat().st_size)
            if (
                self.path.name in {"primary-entered.bin", "audit-entered.bin"}
                and self.file.tell() + size <= 16
                and self.path.stat().st_size == 16
            ):
                return self.file.write(data)  # Already allocated fixed16byte counter.
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                raw, text = usage(root)
                text_credit = (
                    with_credit if any(s in TEXT for s in self.path.suffixes) else 0
                )
                if (
                    raw + with_credit + 65536 > rawmax
                    or text + text_credit + 65536 > textmax
                ):
                    raise TimeoutError("Owned Python pre-write raw/text first limit")
                result = self.file.write(data)
                self.file.flush()
                return result
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

        def writelines(self, lines):
            for line in lines:
                self.write(line)

    def guarded(original, path, mode="r", *args, **kwargs):
        p = (
            Path(path).resolve()
            if isinstance(path, (str, bytes, os.PathLike))
            else None
        )
        file = original(path, mode, *args, **kwargs)
        return (
            Checked(file, p)
            if p is not None
            and p.is_relative_to(root)
            and any(x in mode for x in "wax+")
            else file
        )

    builtins.open = lambda path, mode="r", *args, **kwargs: guarded(
        original_open, path, mode, *args, **kwargs
    )
    io.open = lambda path, mode="r", *args, **kwargs: guarded(
        original_io, path, mode, *args, **kwargs
    )
    # Torch's C path writer bypasses Python open. Serialize bounded bytes first.
    import torch

    save = torch.save

    class Bounded(io.BytesIO):
        def write(self, data):
            if self.tell() + len(data) > filemax:
                raise ValueError("Stage Torch serialization file bound")
            return super().write(data)

    def checked_save(obj, file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file).resolve().is_relative_to(
            root
        ):
            buffer = Bounded()
            save(obj, buffer, *args, **kwargs)
            with builtins.open(file, "wb") as f:
                f.write(buffer.getvalue())
        else:
            save(obj, file, *args, **kwargs)

    torch.save = checked_save
    # SB3 exposes singledispatch specifically for alternate file handles. Keep
    # the checked adapter instead of bypassing guards to satisfy its type list.
    from stable_baselines3.common.save_util import open_path

    @open_path.register(Checked)
    def checked_sb3_path(file, mode, verbose=0, suffix=None):
        if mode not in ("r", "read", "w", "write"):
            raise ValueError("Registered checked SB3 file mode")
        if (mode in ("r", "read") and not file.readable()) or (
            mode in ("w", "write") and not file.writable()
        ):
            raise ValueError("Checked SB3 file capability")
        return file

    cmd = a.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    sys.argv = cmd[1:] if cmd[0] == "-m" else cmd
    if cmd[0] == "-m":
        runpy.run_module(cmd[1], run_name="__main__", alter_sys=True)
    else:
        runpy.run_path(cmd[0], run_name="__main__")


if __name__ == "__main__":
    main()
