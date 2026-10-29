"""Restore official SDK Linux CRC libraries omitted by its upstream wheel packaging."""

import argparse
import hashlib
from importlib.util import find_spec
import io
from pathlib import Path
import tarfile
from urllib.request import urlopen

REVISION = "65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5"
SOURCE_HASH = "b74e43fb6535929ec2a9efc1685e96111932ac6a9d7105b9cc7564ebaf6b7d58"
URL = f"https://github.com/unitreerobotics/unitree_sdk2_python/archive/{REVISION}.tar.gz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded pinned source archive")
    args = parser.parse_args()
    spec = find_spec("unitree_sdk2py")
    if spec is None:
        raise RuntimeError("Install the unitree extra first")
    if args.archive:
        data = args.archive.read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != SOURCE_HASH:
        raise ValueError("Unitree SDK source checksum mismatch")
    target = Path(next(iter(spec.submodule_search_locations))) / "utils/lib"
    target.mkdir(exist_ok=True)
    root = f"unitree_sdk2_python-{REVISION}/"
    members = {"crc_amd64.so": "unitree_sdk2py/utils/lib/crc_amd64.so",
               "crc_aarch64.so": "unitree_sdk2py/utils/lib/crc_aarch64.so", "LICENSE": "LICENSE"}
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for name, member in members.items():
            with archive.extractfile(root + member) as stream:
                (target / name).write_bytes(stream.read())
    print(f"Official CRC libraries and license installed: {target}")


if __name__ == "__main__":
    main()
