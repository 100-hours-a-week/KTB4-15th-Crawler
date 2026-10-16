"""Print runtime identity without network access or environment secrets."""

import json
import platform
import ssl
import sys
from pathlib import Path


def runtime_info():
    release = Path("/etc/os-release")
    return {
        "executable": sys.executable,
        "python_version": platform.python_version(),
        "ssl_version": ssl.OPENSSL_VERSION,
        "os": platform.system(),
        "architecture": platform.machine(),
        "os_release": release.read_text().splitlines() if release.exists() else None,
    }


if __name__ == "__main__":
    print(json.dumps(runtime_info(), ensure_ascii=False, indent=2))
