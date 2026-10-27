import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "scripts/validate_bout_render.py"
spec = importlib.util.spec_from_file_location("bout_video_qa", path)
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


class Reader:
    def __init__(self, count=26, shape=(576, 1024, 3)):
        self.count, self.shape, self.decoded = count, shape, 0

    def __len__(self):
        raise AssertionError("A streaming reader length hint is not a frame count")

    def __iter__(self):
        # Reused mutable decoder buffer: output must own its first/last copies.
        buffer = np.zeros(self.shape, np.uint8)
        for tick in range(self.count):
            self.decoded += 1
            buffer.fill(tick % 256)
            yield buffer


def test_stream_decode_never_asks_length_hint_and_owns_endpoint_copies():
    reader = Reader()
    first, last, count = qa.decode_bounded(reader)
    assert count == reader.decoded == 26
    assert np.all(first == 0) and np.all(last == 25)
    assert not np.shares_memory(first, last)


@pytest.mark.parametrize("count", [0, 25, 27, 100000])
def test_stream_decode_rejects_missing_extra_and_unbounded_frames(count):
    reader = Reader(count=count)
    with pytest.raises(ValueError):
        qa.decode_bounded(reader)
    assert reader.decoded <= 27


def test_stream_decode_rejects_foreign_geometry():
    with pytest.raises(ValueError):
        qa.decode_bounded(Reader(shape=(8, 8, 3)))
