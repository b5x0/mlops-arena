"""
setup_zenml.py v4 - Robust Python-based ZenML Stack Registration
================================================================
- Uses the ZenML Python Client for registration (bypasses CLI parsing bugs)
- ASCII-only logging (no emojis) to avoid Windows encoding crashes
- Handles 'connect' via CLI then performs registration via SDK
- Supports --reset to wipe previous state

Usage:
    python setup_zenml.py           # normal run
    python setup_zenml.py --reset   # clean slate
"""

import os
import sys
import argparse
import subprocess
import urllib.request
import urllib.error

# For Python Client
try:
    from zenml.client import Client
    from zenml.enums import StackComponentType
    from zenml.exceptions import EntityExistsError
except ImportError:
    print("ZenML not installed in this environment. Run pip install zenml.")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Disable analytics
os.environ["ZENML_ANALYTICS_OPT_IN"] = "false"

# Resolve ZenML binary for the 'connect' part
_scripts = os.path.join(
    os.path.dirname(sys.executable),
    "Scripts" if sys.platform == "win32" else "bin",
)
_bin = os.path.join(_scripts, "zenml.exe" if sys.platform == "win32" else "zenml")
_fallback_bin = r"C:\Users\MSI\AppData\Roaming\Python\Python311\Scripts\zenml.exe"
if os.path.isfile(_bin):
    ZENML = _bin
elif os.path.isfile(_fallback_bin):
    ZENML = _fallback_bin
else:
    ZENML = "zenml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_print(msg: str):
    """Prints ASCII-scrubbed message to avoid Windows encoding errors."""
    if not msg:
        return
    print(msg.encode("ascii", "replace").decode("ascii"))


def run_cli(args: list, check: bool = False):
    """Run a ZenML CLI command."""
    cmd = [ZENML] + [str(a) for a in args]
    clean_print(f"\n  [CLI] zenml {' '.join(args)}")
    proc = subprocess.run(cmd, capture_output=True)
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    
    if out.strip():
        clean_print(out.strip())
    
    # Filter out daemon warnings
    err_filtered = "\n".join(
        l for l in err.splitlines() if "Daemon functionality" not in l
    ).strip()
    if err_filtered:
        clean_print(err_filtered)
        
    if check and proc.returncode != 0:
        raise RuntimeError(f"CLI command failed (exit {proc.returncode})")
    return proc.returncode


def ping(url: str, timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status < 500
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    minio_bucket   = os.getenv("MINIO_BUCKET",          "mlops-bucket")
    aws_key        = os.getenv("AWS_ACCESS_KEY_ID",     "admin")
    aws_secret     = os.getenv("AWS_SECRET_ACCESS_KEY", "password123")
    zenml_url      = os.getenv("ZENML_SERVER_URL",      "http://localhost:8080")
    zenml_user     = os.getenv("ZENML_USERNAME",        "admin")
    zenml_password = os.getenv("ZENML_PASSWORD",        "Password123.") # Default fallback
    mlflow_uri     = "http://localhost:5000"
    minio_endpoint = "http://localhost:9000"

    clean_print("=" * 60)
    clean_print("  ZenML Setup (v4 - Python Client)")
    clean_print("=" * 60)

    # 0. Health Check
    clean_print("\n[0] Checking Docker services...")
    services = [
        ("ZenML ", zenml_url + "/api/v1/info"),
        ("MLflow", mlflow_uri),
        ("MinIO ", minio_endpoint + "/minio/health/live"),
    ]
    all_ok = True
    for name, url in services:
        up = ping(url)
        clean_print(f"    {'OK' if up else 'DOWN'} : {name} at {url}")
        if not up: all_ok = False
        
    if not all_ok:
        clean_print("\n  ERROR: One or more services are down. run docker-compose up -d")
        sys.exit(1)

    # 1. Connect via CLI (standard way to setup profile)
    clean_print("\n[1] Initializing and connecting...")
    run_cli(["init"])
    
    # Logout/Login to ensure clean session
    run_cli(["logout"])
    connect_cmd = ["connect", "--url", zenml_url, "--username", zenml_user]
    if zenml_password:
        connect_cmd.extend(["--password", zenml_password])
    run_cli(connect_cmd, check=True)

    # Now use Python Client for registration
    client = Client()
    
    if args.reset:
        clean_print("\n[2] --reset: Removing active stack and components...")
        try:
            client.delete_stack("cifar_stack", recursive=True)
            clean_print("    Stack 'cifar_stack' deleted.")
            # Also try to delete components explicitly if they weren't part of the stack
            for c_type, c_name in [
                (StackComponentType.ARTIFACT_STORE, "minio_store"),
                (StackComponentType.EXPERIMENT_TRACKER, "mlflow_tracker"),
                (StackComponentType.MODEL_REGISTRY, "mlflow_registry")
            ]:
                try:
                    client.delete_stack_component(c_type, c_name)
                    clean_print(f"    {c_type} '{c_name}' deleted.")
                except Exception:
                    pass
        except Exception:
            clean_print("    Reset encountered issues (some items might not exist).")

    # 3. Artifact Store
    clean_print("\n[3] Registering Artifact Store...")
    try:
        client.create_stack_component(
            name="minio_store",
            flavor="s3",
            component_type=StackComponentType.ARTIFACT_STORE,
            configuration={
                "path": f"s3://{minio_bucket}",
                "key": aws_key,
                "secret": aws_secret,
                "client_kwargs": {"endpoint_url": minio_endpoint}
            }
        )
        clean_print("    Artifact store 'minio_store' registered successfully.")
    except EntityExistsError:
        clean_print("    Artifact store 'minio_store' already exists.")

    # 4. MLflow Tracker
    clean_print("\n[4] Registering Experiment Tracker...")
    try:
        client.create_stack_component(
            name="mlflow_tracker",
            flavor="mlflow",
            component_type=StackComponentType.EXPERIMENT_TRACKER,
            configuration={
                "tracking_uri": mlflow_uri,
                "tracking_username": zenml_user,
                "tracking_password": zenml_password
            }
        )
        clean_print("    Experiment tracker 'mlflow_tracker' registered successfully.")
    except EntityExistsError:
        clean_print("    Experiment tracker 'mlflow_tracker' already exists.")

    # 5. Model Registry (MLflow)
    clean_print("\n[5] Registering Model Registry...")
    try:
        client.create_stack_component(
            name="mlflow_registry",
            flavor="mlflow",
            component_type=StackComponentType.MODEL_REGISTRY,
            configuration={}
        )
        clean_print("    Model registry 'mlflow_registry' registered successfully.")
    except EntityExistsError:
        clean_print("    Model registry 'mlflow_registry' already exists.")

    # 6. Stack
    clean_print("\n[6] Registering and Activating Stack...")
    try:
        # Get component IDs
        artifact_store = client.get_stack_component(StackComponentType.ARTIFACT_STORE, "minio_store")
        mlflow_tracker = client.get_stack_component(StackComponentType.EXPERIMENT_TRACKER, "mlflow_tracker")
        mlflow_registry = client.get_stack_component(StackComponentType.MODEL_REGISTRY, "mlflow_registry")
        orchestrator   = client.get_stack_component(StackComponentType.ORCHESTRATOR, "default")
        
        client.create_stack(
            name="cifar_stack",
            components={
                StackComponentType.ARTIFACT_STORE: artifact_store.id,
                StackComponentType.EXPERIMENT_TRACKER: mlflow_tracker.id,
                StackComponentType.MODEL_REGISTRY: mlflow_registry.id,
                StackComponentType.ORCHESTRATOR: orchestrator.id
            }
        )
        clean_print("    Stack 'cifar_stack' registered successfully.")
    except EntityExistsError:
        clean_print("    Stack 'cifar_stack' already exists.")

    # Set active
    run_cli(["stack", "set", "cifar_stack"], check=True)

    clean_print("\n" + "=" * 60)
    clean_print("  Setup complete! Active stack:")
    run_cli(["stack", "describe"])
    clean_print("=" * 60)

if __name__ == "__main__":
    main()
