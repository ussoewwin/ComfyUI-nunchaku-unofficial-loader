import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(HERE, "requirements.txt")

# Windows public wheels (woct0rdho / triton-lang). Upper bound avoids pulling a
# Triton that breaks common torch builds; adjust when torch support matrix moves.
_WINDOWS_TRITON_SPEC = "triton-windows<3.7"
_LINUX_TRITON_SPEC = "triton"


def _pip(*args):
    cmd = [sys.executable, "-m", "pip", *args]
    print("[nunchaku-unofficial-loader][install] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _pip_quiet_uninstall(package):
    """Best-effort uninstall; missing package is not an error."""
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        package,
    ]
    print("[nunchaku-unofficial-loader][install] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _probe_triton_import():
    """Return True if `import triton` works under this interpreter."""
    code = (
        "try:\n"
        "    import triton  # noqa: F401\n"
        "except Exception:\n"
        "    raise SystemExit(1)\n"
        "raise SystemExit(0)\n"
    )
    rc = subprocess.call([sys.executable, "-c", code])
    return rc == 0


def _cuda_likely_available():
    """True if torch reports CUDA; False if no CUDA; None if torch missing."""
    code = (
        "try:\n"
        "    import torch\n"
        "    raise SystemExit(0 if torch.cuda.is_available() else 2)\n"
        "except Exception:\n"
        "    raise SystemExit(1)\n"
    )
    rc = subprocess.call([sys.executable, "-c", code])
    if rc == 0:
        return True
    if rc == 2:
        return False
    # torch not importable yet — still try Triton on Windows/Linux for GPU users
    # who install torch later.
    return None


def _install_triton_for_int8_speed():
    """
    Install a Triton runtime so Plan B INT8 fused Linear kernels can run.

    Windows needs triton-windows (stock triton wheels are not usable natively).
    Linux uses the standard triton package (often already pulled by torch).
    """
    system = platform.system()
    print(
        "[nunchaku-unofficial-loader][install] --- INT8 Triton speed environment ---",
        flush=True,
    )

    if system == "Darwin":
        print(
            "[nunchaku-unofficial-loader][install] macOS: skipping Triton "
            "(INT8 Triton acceleration requires NVIDIA CUDA).",
            flush=True,
        )
        print(
            "[nunchaku-unofficial-loader][install] INT8 Triton speed path: UNAVAILABLE",
            flush=True,
        )
        return

    cuda = _cuda_likely_available()
    if cuda is False:
        print(
            "[nunchaku-unofficial-loader][install] torch.cuda.is_available() is False; "
            "still attempting Triton install for users who switch to a CUDA torch later.",
            flush=True,
        )

    if _probe_triton_import() and system != "Windows":
        print(
            "[nunchaku-unofficial-loader][install] triton already importable.",
            flush=True,
        )
        print(
            "[nunchaku-unofficial-loader][install] INT8 Triton speed path: READY",
            flush=True,
        )
        return

    if system == "Windows":
        # Stock Linux `triton` on Windows is a common footgun.
        _pip_quiet_uninstall("triton")
        rc = _pip("install", "-U", _WINDOWS_TRITON_SPEC)
        if rc != 0:
            print(
                "[nunchaku-unofficial-loader][install] ERROR: failed to install %s "
                "(exit %d)." % (_WINDOWS_TRITON_SPEC, rc),
                flush=True,
            )
            print(
                "[nunchaku-unofficial-loader][install] Remediation (same python as ComfyUI):\n"
                '  "%s" -m pip install -U "%s"'
                % (sys.executable, _WINDOWS_TRITON_SPEC),
                flush=True,
            )
        else:
            print(
                "[nunchaku-unofficial-loader][install] installed %s"
                % _WINDOWS_TRITON_SPEC,
                flush=True,
            )
    else:
        # Linux (and other non-Darwin UNIX): standard triton.
        if not _probe_triton_import():
            rc = _pip("install", "-U", _LINUX_TRITON_SPEC)
            if rc != 0:
                print(
                    "[nunchaku-unofficial-loader][install] ERROR: failed to install %s "
                    "(exit %d)." % (_LINUX_TRITON_SPEC, rc),
                    flush=True,
                )
                print(
                    "[nunchaku-unofficial-loader][install] Remediation:\n"
                    '  "%s" -m pip install -U %s'
                    % (sys.executable, _LINUX_TRITON_SPEC),
                    flush=True,
                )
            else:
                print(
                    "[nunchaku-unofficial-loader][install] installed %s"
                    % _LINUX_TRITON_SPEC,
                    flush=True,
                )

    if _probe_triton_import():
        print(
            "[nunchaku-unofficial-loader][install] INT8 Triton speed path: READY",
            flush=True,
        )
    else:
        print(
            "[nunchaku-unofficial-loader][install] INT8 Triton speed path: UNAVAILABLE "
            "— INT8 will fall back to eager/_int_mm until Triton imports successfully.",
            flush=True,
        )
        if system == "Windows":
            print(
                "[nunchaku-unofficial-loader][install] Windows tips: use this exact "
                "python (-m pip), not a different pip.exe; portable embeds may need "
                "matching Python include/libs for Triton runtime.",
                flush=True,
            )


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

    # After base deps: install Triton so Plan B INT8 kernels can deliver speed
    # for public users (does not rely on Comfy --enable-triton-backend).
    _install_triton_for_int8_speed()


if __name__ == "__main__":
    main()
