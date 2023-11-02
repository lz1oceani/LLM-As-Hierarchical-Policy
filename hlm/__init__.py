from pathlib import Path
import os

HLM = Path(__file__).parent
DATASETS = Path(os.environ.get("DATASETS", str(HLM.parent / "data")))

PROMPTS = HLM / "prompts"
PROCESSED_RAW = DATASETS / "processed_raw"
RESULTS = HLM.parent / "results"
