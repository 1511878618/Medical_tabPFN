from pathlib import Path

# project_dir = Path("V3")
project_dir = Path("V1")
project_dir.mkdir(exist_ok=True, parents=True)

outputDir = project_dir / "output"
dataDir = project_dir / "data"

outputDir.mkdir(exist_ok=True)
dataDir.mkdir(exist_ok=True)
