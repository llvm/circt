from pathlib import Path

from pycde.system import System

system = System([])
system.import_hw_modules(Path(__file__).parent / "hot_chips_2022_example.mlir")
system.print()
