"""Build SDK2's native CycloneDDS dependency locally; requires CMake and a C compiler."""

import hashlib
from pathlib import Path
import subprocess
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
SOURCE_HASH = "bc84e137e0c8a055b8cd97fbeafec94e36de1b0c2e88800896a82384fd867ae5"
URL = "https://codeload.github.com/eclipse-cyclonedds/cyclonedds/tar.gz/refs/tags/0.10.2"


def main():
    DEPS.mkdir(exist_ok=True)
    archive = DEPS / "cyclonedds-0.10.2.tar.gz"
    if not archive.exists():
        with urlopen(URL, timeout=60) as response:
            archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError("CycloneDDS source checksum mismatch")
    source = DEPS / "cyclonedds-0.10.2"
    if not source.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(DEPS, filter="data")
    build = DEPS / "cyclonedds-build"
    prefix = DEPS / "cyclonedds"
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), f"-DCMAKE_INSTALL_PREFIX={prefix}",
                    "-DBUILD_EXAMPLES=OFF", "-DBUILD_TESTING=OFF", "-DENABLE_SSL=NO", "-DENABLE_SECURITY=NO"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", "4"], check=True)
    subprocess.run(["cmake", "--install", str(build)], check=True)
    print(f"CycloneDDS installed locally: {prefix}")
    print('Next: CYCLONEDDS_HOME="$PWD/.deps/cyclonedds" uv sync --extra train --extra unitree --locked')


if __name__ == "__main__":
    main()
