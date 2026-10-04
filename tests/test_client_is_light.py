"""The robot-side install must not need torch, LeRobot or the web server stack.

Runs in a subprocess that makes those packages unimportable, then exercises everything a robot
computer uses: client, executor, profiles, spec, codec, and the inspect/check/list CLI.
"""
import json
import subprocess
import sys
import textwrap

BLOCKED = ["torch", "torchvision", "lerobot", "fastapi", "uvicorn", "starlette", "transformers",
           "onnxruntime"]


def test_client_side_imports_without_server_dependencies(ckpt, tmp_path):
    script = textwrap.dedent(f"""
        import sys, importlib.abc
        BLOCKED = {BLOCKED!r}
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"blocked for this test: {{name}}")
        sys.meta_path.insert(0, Block())

        from vlaserve.client import PolicyClient, ChunkExecutor, ActionQueue
        from vlaserve.profiles import RobotProfile
        from vlaserve.spec import PolicySpec
        from vlaserve import codec
        from vlaserve.cli import main
        assert main(["inspect", {str(ckpt)!r}]) == 0
        assert main(["list"]) == 0
        assert main(["serve"]) == 2          # clear message, not a ModuleNotFoundError
        print("LOADED", sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED))
    """)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "LOADED []" in r.stdout
