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
