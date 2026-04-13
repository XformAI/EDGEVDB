# EdgeVDB Native Libraries

Platform-specific shared libraries for the EdgeVDB Python SDK.

## Structure

```
lib/
  windows/   → edgevdb_shared.dll, libedgevdb_shared.dll
  linux/     → libedgevdb_shared.so
  darwin/    → libedgevdb_shared.dylib
```

## Building

Copy the built shared library from `build/desktop-release/core/` into the
appropriate platform directory:

```bash
# Linux
cp build/desktop-release/core/libedgevdb_shared.so python/edgevdb/lib/linux/

# macOS
cp build/desktop-release/core/libedgevdb_shared.dylib python/edgevdb/lib/darwin/

# Windows (PowerShell)
copy build\desktop-release\core\edgevdb_shared.dll python\edgevdb\lib\windows\
```

The Python SDK (`__init__.py`) automatically detects the host platform and
loads the correct library from the matching subdirectory.
