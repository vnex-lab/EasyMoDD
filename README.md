# EasyMoDD

EasyMoDD is a configurable, training-first machine-learning library. It is
programmable from Python like a numerical library, with reusable tensors,
modules, datasets, losses, optimizers, trainers, model families, and checkpoint
policies.

Models initialize randomly unless a consuming project explicitly resumes from
its own checkpoint. EasyMoDD does not include a chatbot, web application,
built-in corpus, project `config.txt`, generated checkpoints, or demo data.

## Install

```powershell
python -m pip install easymodd
```

For a local checkout:

```powershell
python -m pip install -e .
```

The core distribution is source-only plus NumPy; it does not bundle model
weights, corpora, JDKs, native toolchains, browsers, or decompiler binaries.
The web console uses Python's standard library. CUDA, tabular-data tooling,
and AES-GCM encryption are optional extras, keeping a normal install far below
2 GB. Large datasets, generated apps, models, and user-selected toolchains stay
outside the library package.


The package version is controlled by `project.version` in `pyproject.toml`.
Increase it for every PyPI release, and do not reuse an already-uploaded
version.

Optional integrations:

```powershell
python -m pip install easymodd[cuda]
python -m pip install easymodd[data]
python -m pip install easymodd[dev]
```

## Python API

```python
import numpy as np
from easymodd import ArrayDataset, MSELoss, SGD, Trainer, build_mlp

model = build_mlp(input_size=1, hidden_sizes=[8], output_size=1)
dataset = ArrayDataset(
    np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
    np.array([[2.0], [4.0], [6.0]], dtype=np.float32),
)
history = Trainer(
    SGD(model.parameters(), learning_rate=0.01),
    MSELoss(),
    max_steps=100,
).fit(model, dataset)
print(history.values["loss"][-1])
```

The programmable foundation includes:

- NumPy and explicit CUDA backend helpers
- `Module`, `Parameter`, `Linear`, `ReLU`, and `Sequential`
- GELU, sigmoid, tanh, and dropout layers
- MLP model factory
- Model registries for custom factories
- `ArrayDataset` and packed `TextDataset`
- `BatchLoader`, deterministic splits, and `Subset`
- Composable transforms and batch collation
- MSE and cross-entropy loss contracts
- L1 and Huber losses
- SGD, Adam, and AdamW optimizers
- Cosine and warmup learning-rate schedulers
- Reusable `Trainer` and `TrainingHistory`
- Accuracy, MAE, and perplexity metrics
- JSONL metrics logging and early stopping callbacks
- Local experiment tracking and dependency-free summaries
- Versioned model/optimizer/scheduler serialization
- Environment-only secrets for optional integrations
- Config loading and validation for consuming projects
- Checkpoint enable/disable, retention, and compression policy
- Separate text Transformer and Vision Transformer configuration modules
- Streaming corpus fingerprints and bounded-memory duplicate audits
- Memory-mapped text windows for large byte-token corpora

### Corpus audit before training

The streaming auditor fingerprints the exact source file, estimates words and
tokens, reports blank or malformed records, and detects normalized duplicates.
It stores only record hashes in a temporary SQLite file rather than loading the
corpus or all hashes into RAM. Optionally write a deduplicated copy:

```powershell
easymodd audit-corpus --source corpus.jsonl --format jsonl `
  --text-field response.text --deduplicated-output cleaned.jsonl --json
```

Plain UTF-8 text is treated as newline-delimited records. JSONL is validated
one record at a time. The SHA-256 fingerprint and audit counts can be saved
with an experiment to make dataset changes visible before expensive training.

For training on a corpus larger than available RAM, use `MMapTextDataset`:

```python
from easymodd import MMapTextDataset

with MMapTextDataset("corpus.txt", sequence_length=512, stride=512) as dataset:
  inputs, next_tokens = dataset[0]
  print(len(dataset), inputs.shape, next_tokens.shape)
```

It uses raw UTF-8 bytes as token IDs 0-255, keeps the source file memory-mapped,
and materializes only the requested sequence window. Use `stride < sequence_length`
for overlapping windows; this increases training examples and I/O.

The restored prototype scripts remain available as compatibility programs for
existing custom checkpoints. New projects should import EasyMoDD directly.

## Consumer Project Configuration

EasyMoDD does not force users into one project layout. A consuming project can
create its own `config.txt` using TOML syntax and load it with:

```python
from easymodd import load_config
config = load_config("config.txt")
```

The config loader supports project-defined paths for datasets, runs, models,
logs, and checkpoints. The library never assumes a filename such as
`best.bin`.

A consumer project can configure checkpoint behavior like this:

```toml
[checkpoints]
enabled = true
keep_recent = 2
compress_old = true
```

When disabled, `CheckpointManager` creates no directory or files. When enabled,
older numbered checkpoints can be compressed while recent checkpoints remain
available.

## Package Layout

```text
src/easymodd/
  models/
    mlp.py
    text_transformer.py
    vision_transformer.py
  data.py
  losses.py
  nn.py
  tensor.py
  optim.py
  training.py
  checkpoints.py
  config.py
  cli.py
```

Each model family has its own module. The shared training contracts are designed
so new model types can be added without changing dataset or checkpoint code.

## Extending EasyMoDD

Register a custom model factory:

```python
from easymodd import ModelRegistry

registry = ModelRegistry()
registry.register("my_model", lambda width=8: build_my_model(width))
model = registry.create("my_model", width=32)
```

Optional integrations can read environment variables without storing secrets:

```python
from easymodd import SecretStore

secrets = SecretStore(prefix="MY_APP_")
token = secrets.get("TRACKING_TOKEN")
```

EasyMoDD never writes API keys to configs, checkpoints, experiment files, or
logs. Provider-specific integrations can be built on top of this helper.

## Data and Training

The library does not ship a general-purpose training database. Users provide
legally obtained text, image, or tabular data in their own projects. EasyMoDD
provides dataset contracts and validation helpers; it does not pretend that a
small demo corpus can produce a general assistant.

## Development Status

The tensor/module API, programmable MLP training loop, dataset contracts,
checkpoint policy, model-family layout, and config loader are implemented.
Text Transformer training, Vision Transformer training, full resume manifests,
evaluation, schedulers, callbacks, and additional model families are active
parts of the public-library roadmap.

Support development on [Ko-fi](https://ko-fi.com/vnexlab).

## Tests and examples

The `tests/` directory contains individual subsystem tests plus a full-stack
integration test. `examples/multi_file_project/` demonstrates a real consumer
project layout with TOML config, multiple Python plugin files, custom web
routes/pages, C++ source, a CMake shared-library recipe, and a custom DLL
metadata backend.

The multi-file test loads that consumer config, discovers the C++ source,
registers the custom backend/API/page, and inspects a synthetic PE DLL without
loading or executing it. A C++ compiler command is dry-run only and is skipped
when no compatible compiler is installed. Tests also cover Java JAR/class,
Python bytecode, WebAssembly, training, corpus auditing, encryption, web auth,
simulation, math helpers, and local forwarding.

Run the suite with:

```powershell
python -m pip install -e ".[dev,security]"
python -m pytest -q
```

If a native C++ compiler is installed, the suite also verifies the compiler
adapter's dry-run command. It does not execute generated DLL/EXE fixtures.

## EasyModel artifact tooling

This distribution also includes the separate `EasyModel` namespace for safe,
configurable artifact inspection, simulations, app scaffolding, networking, and
compiler/decompiler adapters. It does not replace the `easymodd` training API.

The namespaces connect through `EasyModel.Workbench` and a bounded,
thread-safe event bus. Training, simulations, corpus audits, artifact
inspection, encryption, and custom tools publish redacted events to one local
run history. `EzMHandler` is the root facade for coordinated operations.

```python
from EasyModel import SecurityPolicy, detect_artifact
from EzDecompiler import EzDecompiler

handler = EzDecompiler(policy=SecurityPolicy(allow_native_tools=False))
artifact = handler.detect("application.jar")
report = handler.inspect("application.jar")
print(artifact.kind, report.to_json())
```

From a source checkout:

```powershell
python EzDecompiler.py detect application.jar --json
python EzDecompiler.py inspect application.jar --json
python EzDecompiler.py decompile application.jar --dry-run --json
```

After installation:

```powershell
easymodel detect application.jar --json
easymodel inspect application.jar --json
easymodel passport application.jar --json
easymodel compare original.jar --against rebuilt.jar --json
easymodel web
easymodel lan
easymodel app --name my_toolbox --directory ./my_toolbox --template toolbox
```

An Artifact Passport adds a streaming SHA-256 identity plus safe structural
metadata: JAR entry and manifest inventory, archive traversal/duplicate-name
warnings, PE sections and .NET marker detection, Python bytecode header details,
and WebAssembly section sizes. It never extracts or executes the artifact.
Passports can be compared to see whether rebuilt or transformed outputs changed.

### Local web workbench

`easymodel web` starts a standard-library web console at `127.0.0.1:8765`. It
combines artifact passports, corpus audits, file encryption, and recent shared
events. The server prints a random session token; enter it in the page. Remote
binding is rejected unless explicitly enabled with a token, and remote use must
be behind HTTPS. Keyboard shortcuts only operate inside the open browser page;
the library does not capture system-wide keystrokes.

Trusted plugins can add isolated browser pages and authenticated POST API
routes. A custom page can call registered routes through the parent console
bridge without receiving the session token. Plugins execute Python code, so
load only modules you trust.

### Custom pages and API plugins

Consumer-owned TOML controls plugin loading and the web address:

```toml
[plugins]
enabled = true
modules = ["my_project.easy_plugins"]

[web]
host = "127.0.0.1"
port = 8765
allow_remote = false
```

Each configured module exposes `register(workbench)`:

```python
from EasyModel import ApiRoute, ArtifactKind, WebPage
from EasyModel.compilers import CompilerBackend
from EasyModel.decompilers import ExternalDecompiler

def register(workbench):
  workbench.register_component("tools", "hello", lambda: {"ready": True})
  workbench.register_compiler_backend(CompilerBackend("cc", "cc"))
  workbench.register_decompiler_backend(ExternalDecompiler(
    "my-java-decompiler", "my-decompiler", {ArtifactKind.JAR, ArtifactKind.CLASS}
  ))
  workbench.register_page(WebPage(
    "/pages/status", "Project status",
    "<main><h1>Project status</h1><p>Custom page content.</p></main>",
  ))
  workbench.register_api_route(ApiRoute(
    "/api/custom/status", lambda body, hub: {"ready": True}
  ))
```

The page runs in a sandboxed frame. It can ask the parent workbench to call
only registered custom POST routes:

```javascript
window.parent.postMessage({
  type: "easymodel-api",
  path: "/api/custom/status",
  payload: {}
}, "*");
```

The browser hosts custom page HTML in a sandboxed frame. Custom API routes are
POST-only and require the same session authentication as built-in operations.

### App builder, simulation, and math

Scaffold a separate consumer project without overwriting existing files:

```powershell
easymodel app --name lab_app --directory ./lab_app --template web
easymodel app --name queue_sim --directory ./queue_sim --template simulation
easymodel app --name model_lab --directory ./model_lab --template training
```

The `DiscreteEventSimulator` provides deterministic event scheduling, seeded
randomness, and a maximum event count. Register simulation factories under
the `simulations` component group and run them with `easymodel simulate --config
config.toml --component your_simulator`. `EasyModel.mathx` provides stable
softmax, blockwise pairwise distances, linear solves, and Simpson integration
using NumPy's optimized kernels, finite-difference gradients, and deterministic memory-bounded Monte Carlo integration.

The shared component registry includes starter providers for MLP models,
discrete-event simulations, JAR inspection, project scaffolding, corpus audit,
and artifact passports. Custom plugins may register more models, simulators,
compilers, decompilers, apps, browser pages, and authenticated API routes.

### Connected workflows

`Workflow` lets a consumer compose these separate tools into one dependency-
ordered run. Results become named inputs to later steps; failures, retries,
skips, durations, and lifecycle events are reported without forcing the tools
to share implementation details:

```python
from EasyModel import Workflow, Workbench

hub = Workbench()
workflow = Workflow("dataset-to-model")
workflow.add("audit", lambda context: hub.audit_corpus("data/train.txt"))
workflow.add(
  "train",
  lambda context: hub.train(model, trainer, dataset),
  depends_on=["audit"],
  retries=1,
)
workflow.add(
  "passport",
  lambda context: hub.passport("artifacts/model.bin"),
  depends_on=["train"],
)
report = hub.run_workflow(workflow)
if not report.ok:
  print(report.errors)
```

Workflows are Python callbacks by design: users control what runs, data passed
between steps, and retry policy. They are bounded to 500 steps by default and
do not evaluate arbitrary expression strings from configuration.

Compiler commands are blocked unless execution is explicitly allowed. Inspect
the structured command first with `--dry-run`; only execute a toolchain and
source you trust:

```powershell
easymodel compile main.c --backend cc --output app --dry-run --config config.toml
easymodel compile main.c --backend cc --output app --allow-execution --config config.toml
```

### LAN hosting and TCP forwarding

`easymodel lan` reports private interface addresses. To intentionally make the
console reachable on a trusted LAN, use an environment token and explicitly
enable remote binding:

```powershell
$env:EASYMODEL_WEB_TOKEN = "a-long-random-token"
easymodel web --host 0.0.0.0 --allow-remote
```

`easymodel forward --port 9000 --target-host 127.0.0.1 --target-port 8000`
forwards a local TCP port. Non-loopback bind addresses require
`--allow-remote`. Forwarding does not edit router/firewall rules, configure
UPnP, or create an Internet tunnel. External reachability remains the network
owner's responsibility; never expose an unauthenticated service.

### Streaming file encryption

Install the optional security extra and keep passphrases in the environment,
not shell history:

```powershell
python -m pip install "easymodd[security]"
$env:EASYMODEL_PASSPHRASE = "use-a-long-unique-passphrase"
easymodel encrypt confidential.bin --output confidential.bin.emenc
easymodel decrypt confidential.bin.emenc --output restored.bin
```

Encryption uses chunked AES-256-GCM with authenticated headers/chunks,
truncation detection, and atomic output. Files stream through fixed-size
buffers rather than loading entirely into RAM. Passphrases are never saved in
configs, event logs, or file metadata. Losing the passphrase means the file
cannot be recovered.

Supported artifact families include Java JAR/class, Windows PE EXE/DLL,
.NET assemblies, Python bytecode, and WebAssembly detection. The first
release provides safe built-in JAR inspection plus optional external-tool
adapters. Native binaries can produce headers, imports, assembly, or
pseudocode, but cannot reliably restore original optimized source code.

Compiler adapters use installed toolchains such as `javac`, `.NET`, Python,
GCC/Clang/MSVC, or WebAssembly tools. Tool execution is policy-controlled and
disabled by default unless explicitly enabled by the consumer project.

Only analyze software you own or have permission to inspect. EasyModel does
not provide DRM, anti-tamper, or access-control bypass functionality.
