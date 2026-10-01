from __future__ import annotations

import json
import zipfile

from EasyModel import EventBus, Workbench, create_passport
from easymodd import ArrayDataset, MSELoss, SGD, Trainer, audit_corpus, build_mlp
import numpy as np


def test_training_corpus_and_artifact_work_together(tmp_path):
    bus = EventBus(history_limit=50)
    workbench = Workbench(bus)

    corpus = tmp_path / "training.txt"
    corpus.write_text("one two three\none two three\nunique sample\n", encoding="utf-8")
    audit = workbench.audit_corpus(corpus)
    assert audit.duplicate_records == 1

    model = build_mlp(1, [4], 1)
    dataset = ArrayDataset(
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        np.asarray([[0.0], [2.0]], dtype=np.float32),
    )
    history = workbench.train(model, Trainer(SGD(model.parameters(), 0.01), MSELoss(), max_steps=8), dataset)
    assert history.values["loss"]

    archive = tmp_path / "package.jar"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("META-INF/MANIFEST.MF", "Main-Class: demo.Main\n")
        output.writestr("demo/Main.class", b"\xca\xfe\xba\xbe")
    passport = workbench.passport(archive)
    assert passport.kind == "jar"

    names = [event["name"] for event in bus.recent()]
    assert "corpus.audit.completed" in names
    assert "training.completed" in names
    assert "artifact.passport.completed" in names

    summary = {
        "audit": audit.to_dict(),
        "training_steps": len(history.values["loss"]),
        "artifact": passport.to_dict(),
    }
    assert json.loads(json.dumps(summary))["artifact"]["kind"] == "jar"


def test_passport_helpers_are_importable_individually(tmp_path):
    source = tmp_path / "sample.cpp"
    source.write_text("extern \"C\" int sum(int a,int b){return a+b;}\n", encoding="utf-8")
    report = create_passport(source)
    assert report.kind == "source"
    assert report.details["line_count"] == 1
