from __future__ import annotations

import numpy as np

from EasyModel import DiscreteEventSimulator, integrate_simpson, pairwise_distances, stable_softmax
from easymodd import (
    ArrayDataset,
    MMapTextDataset,
    MSELoss,
    SGD,
    Trainer,
    audit_corpus,
    build_mlp,
)


def test_memory_mapped_text_windows(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"abcdefghijk")
    with MMapTextDataset(corpus, sequence_length=4, stride=4) as dataset:
        inputs, targets = dataset[0]
        assert len(dataset) == 2
        assert inputs.tolist() == [97, 98, 99, 100]
        assert targets.tolist() == [98, 99, 100, 101]


def test_streaming_corpus_audit_and_dedup(tmp_path):
    source = tmp_path / "corpus.txt"
    cleaned = tmp_path / "cleaned.txt"
    source.write_text("Alpha beta\n alpha   BETA \n\nGamma\n", encoding="utf-8")
    report = audit_corpus(source, deduplicated_output=cleaned)
    assert report.records == 4
    assert report.unique_records == 2
    assert report.duplicate_records == 1
    assert report.empty_records == 1
    assert cleaned.read_text(encoding="utf-8") == "Alpha beta\nGamma\n"


def test_train_model_through_shared_numpy_api():
    model = build_mlp(1, [4], 1)
    inputs = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    targets = np.asarray([[0.0], [2.0], [4.0]], dtype=np.float32)
    history = Trainer(
        SGD(model.parameters(), learning_rate=0.01), MSELoss(), max_steps=40, seed=12
    ).fit(model, ArrayDataset(inputs, targets))
    assert history.values["loss"][-1] < history.values["loss"][0]


def test_math_helpers_are_stable_and_blockable():
    probabilities = stable_softmax(np.asarray([[10000.0, 10001.0]]))
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), [1.0])
    distances = pairwise_distances([[0.0, 0.0], [1.0, 1.0]], [[3.0, 4.0]], row_block=1)
    assert np.allclose(distances[:, 0], [5.0, np.sqrt(13.0)])
    assert abs(integrate_simpson([0.0, 1.0, 4.0]) - 8.0 / 3.0) < 1e-10


def test_discrete_event_order_and_limit():
    simulator = DiscreteEventSimulator(seed=3)
    seen = []
    simulator.on("tick", lambda current, event: seen.append(event.time))
    simulator.schedule(2, "tick")
    simulator.schedule(1, "tick")
    result = simulator.run(until=3)
    assert seen == [1.0, 2.0]
    assert result.processed_events == 2
