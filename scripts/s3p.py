"""Run the repository directly without installing its package."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s3p_repro.cli import main
main()
