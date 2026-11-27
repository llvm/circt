# The cycle-level tests of fifo_latency.py, run on fifo_spare.mlir's FIFO: a
# depth of 7, one short of a power of two, and a read latency of 1.
import fifo_latency
from fifo_latency import (test_fill, test_overwrite_pattern, test_random,
                          test_throughput)

fifo_latency.FIFO_DEPTH = 7
fifo_latency.RD_LATENCY = 1
