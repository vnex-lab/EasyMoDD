# Multi-file EasyModel integration example

This consumer project demonstrates one TOML config loading a plugin that links:

- A custom tools component
- A custom browser page and authenticated API route
- A C++ source file and CMake shared-library build definition
- A configurable C++ compiler adapter
- A custom PE EXE/DLL metadata backend

From the repository root, with EasyMoDD/EasyModel installed in the active environment:

```powershell
python examples/multi_file_project/main.py
python examples/multi_file_project/main.py --web
```

The C++ source is intentionally not compiled automatically. Build it only if a
C++ toolchain is installed and you choose to run it:

```powershell
cmake -S examples/multi_file_project/native -B examples/multi_file_project/native/build
cmake --build examples/multi_file_project/native/build
```

The example DLL backend only reads metadata; it never loads or executes a DLL.
