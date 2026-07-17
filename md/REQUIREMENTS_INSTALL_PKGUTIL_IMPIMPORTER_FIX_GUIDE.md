# requirements.txt Install Failure — `pkgutil.ImpImporter` / Legacy sdist Build Fix

This document explains the `requirements.txt` installation failure reported for an
embedded (portable) ComfyUI environment on Python 3.12, why it happens, how it is
fixed, the files involved, the exact code that implements the fix, and what that
code does.

The failure is **not** caused by this project's own code or by any version pin in
`requirements.txt`. It is caused by an outdated `setuptools` in the target Python
environment combined with a transitive dependency that only ships as a legacy
source distribution.

---

## 1. Error content (from the issue)

Reported in issue #3 — **"Installing requirements.txt failed"**:
<https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader/issues/3>

The user ran, from the custom node directory of a portable install:

```
E:\ComfyUI-Easy-Install\python_embeded\python.exe -m pip install -r requirements.txt
```

Most requirements were already satisfied. The failure occurred while pip tried to
build `filterpy`, which is pulled in transitively by `facexlib`:

```
Collecting filterpy (from facexlib->-r requirements.txt (line 11))
  Downloading filterpy-1.4.5.zip (177 kB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error

  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [31 lines of output]
      Traceback (most recent call last):
        ...
        File "E:\ComfyUI-Easy-Install\python_embeded\Lib\site-packages\setuptools\__init__.py", line 16, in <module>
          import setuptools.version
        File "E:\ComfyUI-Easy-Install\python_embeded\Lib\site-packages\setuptools\version.py", line 1, in <module>
          import pkg_resources
        File "E:\ComfyUI-Easy-Install\python_embeded\Lib\site-packages\pkg_resources\__init__.py", line 2191, in <module>
          register_finder(pkgutil.ImpImporter, find_on_path)
                          ^^^^^^^^^^^^^^^^^^^
      AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'?
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed to build 'filterpy' when getting requirements to build wheel
```

The install stops here; `filterpy` (and therefore `facexlib`) cannot be installed.

---

## 2. Root cause

Three facts combine to produce the failure:

1. **`filterpy 1.4.5` ships only as an sdist (`filterpy-1.4.5.zip`).** There is no
   prebuilt wheel, so pip must build it from source. Building a legacy sdist runs
   its `setup.py` through the `setuptools` build backend, which imports
   `pkg_resources`.

2. **`pkg_resources` from an old `setuptools` references `pkgutil.ImpImporter`.**
   Line 2191 of that old `pkg_resources/__init__.py` calls
   `register_finder(pkgutil.ImpImporter, find_on_path)`.

3. **Python 3.12 removed `pkgutil.ImpImporter`.** The attribute was deprecated for
   years and deleted in 3.12. Any old `setuptools`/`pkg_resources` that still
   references it raises `AttributeError` on import under Python 3.12.

The traceback shows the build backend importing `setuptools` from the environment's
own `site-packages` (`...\python_embeded\Lib\site-packages\setuptools`), i.e. the
build is using the **environment's** (outdated) `setuptools` rather than an
isolated up-to-date one. As soon as that old `setuptools` imports `pkg_resources`,
the missing `pkgutil.ImpImporter` aborts the wheel build.

In short: **an old `setuptools` in a Python 3.12 environment cannot build a legacy
sdist such as `filterpy`.** Modern `setuptools` (v66+) no longer references
`pkgutil.ImpImporter`, so upgrading `setuptools` removes the crash.

---

## 3. Countermeasure overview

Upgrade the build tooling (`pip`, `setuptools`, `wheel`) in the target environment
**before** installing `requirements.txt`. With a modern `setuptools` present, the
`pkg_resources` import no longer touches `pkgutil.ImpImporter`, and the legacy
`filterpy` sdist builds normally.

To make this automatic, the upgrade is performed by an `install.py` script.
ComfyUI-Manager runs a custom node's `install.py` on both **install** and
**update**, executing it in the same Python environment that runs ComfyUI. The
script upgrades the build tooling first, then installs `requirements.txt` itself,
so a fresh install or an update through the Manager no longer hits the
`pkgutil.ImpImporter` error.

`requirements.txt` is left unchanged; a plain `requirements.txt` cannot upgrade
`setuptools` before its own dependencies are resolved in the same pip run, so the
ordering guarantee has to live in a script.

---

## 4. Added / modified files

| File | Status | Role |
|------|--------|------|
| `install.py` | Added | Upgrades `pip` / `setuptools` / `wheel`, then installs `requirements.txt`. Auto-run by ComfyUI-Manager on install and update. |
| `requirements.txt` | Unchanged | Dependency list. Kept as-is; the ordering fix is provided by `install.py`, not by editing this file. |

---

## 5. Full code

### 5.1 `install.py` (complete)

```python
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(HERE, "requirements.txt")


def _pip(*args):
    cmd = [sys.executable, "-m", "pip", *args]
    print("[nunchaku-unofficial-loader][install] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main():
    # Python 3.12 removed pkgutil.ImpImporter. Some transitive source builds
    # (e.g. facexlib -> filterpy, which has no wheel) fail with
    # "AttributeError: module 'pkgutil' has no attribute 'ImpImporter'"
    # when the environment ships an old setuptools. Upgrade build tooling
    # first so those legacy sdist builds succeed.
    _pip("install", "-U", "pip", "setuptools", "wheel")

    if os.path.isfile(REQUIREMENTS):
        rc = _pip("install", "-r", REQUIREMENTS)
        if rc != 0:
            print(
                "[nunchaku-unofficial-loader][install] requirements install "
                "returned code %d" % rc,
                flush=True,
            )
            sys.exit(rc)
    else:
        print(
            "[nunchaku-unofficial-loader][install] requirements.txt not found: %s"
            % REQUIREMENTS,
            flush=True,
        )


if __name__ == "__main__":
    main()
```

### 5.2 `requirements.txt` (unchanged, shown for reference)

```
diffusers>=0.35
transformers>=4.54
sentencepiece
protobuf
huggingface_hub>=0.34
tomli
peft>=0.17
accelerate>=1.10
insightface
opencv-python
facexlib
onnxruntime
timm
```

`filterpy` does not appear here directly; it is pulled in transitively by
`facexlib`, which is why upgrading the build tooling is required rather than
changing this list.

---

## 6. Meaning of the code

### 6.1 `HERE` / `REQUIREMENTS`

```python
HERE = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(HERE, "requirements.txt")
```

`install.py` may be launched from an arbitrary working directory. Resolving the
script's own directory and building an absolute path to `requirements.txt` makes
the install work regardless of where it is invoked from, matching ComfyUI-Manager's
"executed from the root path of the custom node" behavior.

### 6.2 `_pip()`

```python
def _pip(*args):
    cmd = [sys.executable, "-m", "pip", *args]
    print("[nunchaku-unofficial-loader][install] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)
```

pip is invoked as `sys.executable -m pip`. Using `sys.executable` guarantees the
install targets the **same interpreter** running the script (the ComfyUI embedded
Python), not some other Python on `PATH`. Each command is printed with a clear
prefix and `flush=True` so the actions are visible in the ComfyUI console log, and
the pip return code is passed back to the caller for error handling.

### 6.3 Build-tooling upgrade (the actual fix)

```python
_pip("install", "-U", "pip", "setuptools", "wheel")
```

This is the line that resolves the issue. Upgrading `setuptools` to a modern
version replaces the old `pkg_resources` that referenced the removed
`pkgutil.ImpImporter`. `wheel` is upgraded so sdists can be built into wheels
cleanly, and `pip` is upgraded for current resolver/build behavior. Running this
**before** `requirements.txt` guarantees the legacy `filterpy` build sees a working
`setuptools`.

### 6.4 Requirements install and error propagation

```python
if os.path.isfile(REQUIREMENTS):
    rc = _pip("install", "-r", REQUIREMENTS)
    if rc != 0:
        print(... "requirements install returned code %d" % rc, flush=True)
        sys.exit(rc)
else:
    print(... "requirements.txt not found: %s" % REQUIREMENTS, flush=True)
```

After the tooling upgrade, `requirements.txt` is installed with the normal pip
command. If pip fails, the non-zero return code is reported and re-raised via
`sys.exit(rc)`, so ComfyUI-Manager sees the install as failed instead of silently
succeeding. If `requirements.txt` is missing, a diagnostic line is printed rather
than crashing.

### 6.5 Why not edit `requirements.txt` instead

A `requirements.txt` is a flat dependency list resolved in a single pip run; it
cannot force `setuptools` to be upgraded **before** other entries (such as
`facexlib` → `filterpy`) are built in that same run. The ordering guarantee — "new
build tooling first, project dependencies second" — can only be expressed by two
sequential pip invocations, which is why the fix lives in `install.py` and
`requirements.txt` is left untouched.

---

## 7. Result

- **Fresh install via ComfyUI-Manager:** `install.py` runs, upgrades
  `pip`/`setuptools`/`wheel`, then installs `requirements.txt`; the `filterpy`
  build succeeds and the `pkgutil.ImpImporter` error no longer appears.
- **Update via ComfyUI-Manager:** `install.py` is executed again on update, so an
  updated environment is repaired the same way.
- **Manual `pip install -r requirements.txt`:** this path does not run `install.py`
  automatically. In that case, run the build-tooling upgrade first, or run the
  script directly:

  ```
  <python_embeded>\python.exe -m pip install -U pip setuptools wheel
  <python_embeded>\python.exe -m pip install -r requirements.txt
  ```

  or

  ```
  <python_embeded>\python.exe install.py
  ```

With `install.py` in place, the ComfyUI-Manager install and update paths no longer
require any manual step: the build tooling is upgraded automatically before
`requirements.txt` is installed, so the `pkgutil.ImpImporter` failure on outdated
Python 3.12+ environments is resolved. The manual commands above are only needed
when installing dependencies by hand, without going through ComfyUI-Manager.
