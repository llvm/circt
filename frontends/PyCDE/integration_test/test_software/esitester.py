import esiaccel as esi
import sys
import time

esi.accelerator.ctxt.set_stdio_logger(esi.accelerator.cpp.LogLevel.Debug)

platform = sys.argv[1]
acc = esi.AcceleratorConnection(platform, sys.argv[2])

d = acc.build_accelerator()


def print_callback(x):
  print(f"PrintfExample: {x}")


printf = d.ports[esi.AppID("PrintfExample")]
printf.connect(print_callback)
time.sleep(0.1)
