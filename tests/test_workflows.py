from __future__ import annotations

import zipfile

import numpy as np
import pytest

from EasyModel import EventBus, Workbench, Workflow, create_passport
from easymodd import ArrayDataset, MSELoss, SGD, Trainer, audit_corpus, build_mlp


def test_workflow_orders_dependencies_and_retries():
    bus = EventBus()
    workflow = Workflow("ordered", event_bus=bus)
    calls = []

    def prepare(context):
        calls.append("prepare")
        return 5

    def retry_once(context):
        calls.append("retry")
        if calls.count("retry") == 1:
            raise RuntimeError("temporary")
        return context.require("prepare") * 2

    workflow.add("consume", retry_once, depends_on=["prepare"], retries=1)
    workflow.add("prepare", prepare)
    report = workflow.run()
    assert report.ok
    assert report.context.results["consume"] == 10
    assert calls == ["prepare", "retry", "retry"]
    assert any(event["name"] == "workflow.step.retry" for event in bus.recent())


def test_workflow_rejects_unknown_dependencies_and_cycles():
    unknown = Workflow("unknown").add("a", lambda context: None, depends_on=["missing"])
    with pytest.raises(ValueError, match="unknown dependency"):
        unknown.validate()
    cyclic = Workflow("cycle")
    cyclic.add("a", lambda context: None, depends_on=["b"])
    cyclic.add("b", lambda context: None, depends_on=["a"])
    with pytest.raises(ValueError, match="cycle"):
        cyclic.validate()


def test_workflow_continue_skips_dependent_but_runs_independent():
    workflow = Workflow("failure-policy")
    workflow.add("fail", lambda context: 1 / 0)
    workflow.add("blocked", lambda context: "bad", depends_on=["fail"])
    workflow.add("independent", lambda context: "ok")
    report = workflow.run(fail_fast=False)
    assert report.statuses == {"fail": "failed", "blocked": "skipped", "independent": "succeeded"}


def test_combined_audit_training_and_passport_workflow(tmp_path):
    bus = EventBus(history_limit=100)
    hub = Workbench(bus)
    corpus = tmp_path / "train.txt"
    corpus.write_text("alpha beta\nalpha beta\ngamma delta\n", encoding="utf-8")
    package = tmp_path / "module.jar"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\n")
        archive.writestr("module/Main.class", b"\xca\xfe\xba\xbe")

    model = build_mlp(1, [3], 1)
    dataset = ArrayDataset(
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        np.asarray([[0.0], [1.0]], dtype=np.float32),
    )
    workflow = Workflow("release-preflight", event_bus=bus)
    workflow.add("audit", lambda context: hub.audit_corpus(corpus))
    workflow.add(
        "fit",
        lambda context: hub.train(model, Trainer(SGD(model.parameters(), .01), MSELoss(), max_steps=3), dataset),
        depends_on=["audit"],
    )
    workflow.add("passport", lambda context: hub.passport(package), depends_on=["fit"])
    report = hub.run_workflow(workflow)

    assert report.ok
    assert report.context.results["audit"].duplicate_records == 1
    assert report.context.results["passport"].kind == "jar"
    event_names = {event["name"] for event in bus.recent()}
    assert {"workflow.started", "workflow.completed", "training.completed", "artifact.passport.completed"} <= event_names
