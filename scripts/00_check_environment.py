import sys
import importlib

packages = ["numpy", "pandas", "sklearn", "scipy", "yaml"]
print(f"Python: {sys.version.split()[0]}")
for package in packages:
    mod = importlib.import_module(package)
    version = getattr(mod, "__version__", "available")
    print(f"{package}: {version}")
