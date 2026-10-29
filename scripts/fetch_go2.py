"""Download the same pinned Menagerie Go2 MJCF for both backends."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import xml.etree.ElementTree as ET

REVISION = "f3b6013629c7b94f8ee93fd977b161fa8350f9bc"
BASE_URL = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/unitree_go2/"
DEST = Path(__file__).resolve().parents[1] / "assets/robots/go2"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": "https://github.com/google-deepmind/mujoco_menagerie", "revision": REVISION, "sha256": {}}

    def fetch(name):
        path = DEST / name
        with urlopen(BASE_URL + name, timeout=60) as response:
            data = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest["sha256"][name] = hashlib.sha256(data).hexdigest()
        print(name, flush=True)
        return data

    root = ET.fromstring(fetch("go2.xml"))
    fetch("LICENSE")
    for mesh in root.findall("asset/mesh"):
        fetch("assets/" + mesh.attrib["file"])
    (DEST / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
