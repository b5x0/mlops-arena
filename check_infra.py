"""
check_infra.py — Infrastructure Health Check
=============================================
Pings all three Docker services before you run any pipeline code.
Prints a GREEN ✔ or RED ✗ for each endpoint.

Usage:
    python check_infra.py
"""

import sys
import urllib.request
import urllib.error

SERVICES = [
    ("MinIO  ", "http://localhost:9000/minio/health/live"),
    ("MLflow ", "http://localhost:5000"),
    ("ZenML  ", "http://localhost:8080/api/v1/info"),
]

GREEN  = "\033[92m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def check(name: str, url: str, timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            ok = resp.status < 500
    except (urllib.error.URLError, OSError):
        ok = False
    status = f"{GREEN}✔  UP{RESET}" if ok else f"{RED}✗  DOWN{RESET}"
    print(f"  {BOLD}{name}{RESET}  {url:<40}  {status}")
    return ok


def main() -> None:
    print()
    print("  Docker Compose - Service Health Check")
    print("  " + "-" * 60)
    results = [check(name, url) for name, url in SERVICES]
    print("  " + "-" * 60)

    if all(results):
        print(f"\n  All services are UP. You may now run setup_zenml.py\n")
    else:
        down = [SERVICES[i][0].strip() for i, ok in enumerate(results) if not ok]
        print(f"\n  The following services are DOWN: {', '.join(down)}")
        print("  Run:  docker-compose up -d\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
