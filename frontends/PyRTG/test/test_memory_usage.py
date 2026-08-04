# RUN: %rtgtool% %s --seed=0 --output-format=mlir --memory-usage=super-strict | FileCheck %s --check-prefix=STRICT
# RUN: %rtgtool% %s --seed=0 --output-format=mlir --memory-usage=normal | FileCheck %s --check-prefix=NORMAL

from pyrtg import test, sequence, config, Config, embed_comment
from pyrtg.core import MemoryUsage


@sequence([])
def mem_usage_seq():
  if mem_usage_seq.context.memory_usage != MemoryUsage.SUPER_STRICT:
    embed_comment("sequence: memory usage allows extra data")


@config
class MemUsageConfig(Config):
  pass


# STRICT-LABEL: rtg.test @test_conditional_op
# STRICT-CHECK-NOT: rtg.comment

# STRICT-LABEL: rtg.sequence @mem_usage_seq
# STRICT-NEXT: }

# NORMAL-LABEL: rtg.test @test_conditional_op
# NORMAL: [[STR:%.+]] = rtg.constant "test: memory usage allows extra data" : !rtg.string
# NORMAL: rtg.comment [[STR]]

# NORMAL-LABEL: rtg.sequence @mem_usage_seq
# NORMAL: [[SEQSTR:%.+]] = rtg.constant "sequence: memory usage allows extra data" : !rtg.string
# NORMAL: rtg.comment [[SEQSTR]]


@test(MemUsageConfig)
def test_conditional_op(config):
  mem_usage_seq()
  if test_conditional_op.context.memory_usage != MemoryUsage.SUPER_STRICT:
    embed_comment("test: memory usage allows extra data")
