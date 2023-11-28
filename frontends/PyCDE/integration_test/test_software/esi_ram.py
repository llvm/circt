import esi
import sys

platform = sys.argv[1]
acc = esi.Accelerator(platform, sys.argv[2])

m = acc.manifest()
d = m.build_design(acc)
