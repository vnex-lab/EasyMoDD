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

## EasyModel artifact tooling

This distribution also includes the separate `EasyModel` namespace for safe,
configurable artifact inspection and compiler/decompiler adapters. It does not
replace the `easymodd` training API.

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
```

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
