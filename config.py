from pathlib import Path

# project_dir = Path("V3")
project_dir = Path("V4")
project_dir.mkdir(exist_ok=True, parents=True)

outputDir = project_dir / "output"
dataDir = project_dir / "data"

outputDir.mkdir(exist_ok=True)
dataDir.mkdir(exist_ok=True)

# config output dir
assocDir = outputDir / "01-assoc"
riskModelDir = outputDir / "02-risk-model"

assocDir.mkdir(exist_ok=True)
riskModelDir.mkdir(exist_ok=True)
