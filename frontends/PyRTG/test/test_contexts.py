# RUN: %rtgtool% %s --seed=0 --output-format=mlir | FileCheck %s

from pyrtg import test, sequence, Integer, IntegerType, CPUCore, CPUCoreType, SequenceType, config, Param, Config


@sequence([IntegerType()])
def consumer(arg):
  pass


@sequence([CPUCoreType(), CPUCoreType(), SequenceType([])])
def switch(from_ctxt, to_ctxt, seq):
  pass


# MLIR-LABEL: rtg.target @Tgt0 : !rtg.dict<cpu: !rtgtest.cpu>
# MLIR-NEXT: [[V0:%.+]] = rtg.constant #rtgtest.cpu<0> : !rtgtest.cpu
# MLIR-NEXT: [[V1:%.+]] = rtg.get_sequence @switch : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtg.sequence>
# MLIR-NEXT: rtg.context_switch #rtg.any_context : !rtgtest.cpu -> #rtgtest.cpu<0> : !rtgtest.cpu, [[V1]] : !rtg.sequence<!rtgtest.cpu, !rtgtest.cpu, !rtg.sequence>
# MLIR-NEXT: rtg.yield [[V0]] : !rtgtest.cpu


@config
class Tgt0(Config):

  a = Param(loader=lambda: Integer(0))
  b = Param(loader=lambda: Integer(1))
  cpu0 = Param(loader=lambda: CPUCore(0))
  cpu1 = Param(loader=lambda: CPUCore(1))

  def load(self):
    CPUCore.register_switch(CPUCore.any(), CPUCore(0), switch.get())


# CHECK-LABEL: rtg.test @test0_context_args
# CHECK-NEXT:   [[IDX4:%.+]] = index.constant 4
# CHECK-NEXT:   [[SEQ0:%.+]] = rtg.get_sequence @_context_seq_0 : !rtg.sequence<index, index, !rtgtest.cpu, !rtgtest.cpu, index>
# CHECK-NEXT:   [[SUB0:%.+]] = rtg.substitute_sequence [[SEQ0]](%a, %b, %cpu0, %cpu1, [[IDX4]]) : !rtg.sequence<index, index, !rtgtest.cpu, !rtgtest.cpu, index>
# CHECK-NEXT:   rtg.on_context %cpu0, [[SUB0]] : !rtgtest.cpu
# CHECK-NEXT:   [[SEQ1:%.+]] = rtg.get_sequence @consumer : !rtg.sequence<index>
# CHECK-NEXT:   [[SUB1:%.+]] = rtg.substitute_sequence [[SEQ1]](%b) : !rtg.sequence<index>
# CHECK-NEXT:   [[RAND:%.+]] = rtg.randomize_sequence [[SUB1]]
# CHECK-NEXT:   rtg.embed_sequence [[RAND]]

#      CHECK: rtg.sequence @_context_seq_0([[ARG0:%.+]]: index, {{%.+}}: index, {{%.+}}: !rtgtest.cpu, {{%.+}}: !rtgtest.cpu, [[ARG3:%.+]]: index) {
# CHECK-NEXT:   [[SEQ2:%.+]] = rtg.get_sequence @consumer : !rtg.sequence<index>
# CHECK-NEXT:   [[SUB2:%.+]] = rtg.substitute_sequence [[SEQ2]]([[ARG0]]) : !rtg.sequence<index>
# CHECK-NEXT:   [[RAND1:%.+]] = rtg.randomize_sequence [[SUB2]]
# CHECK-NEXT:   rtg.embed_sequence [[RAND1]]
# CHECK-NEXT:   [[SUB3:%.+]] = rtg.substitute_sequence [[SEQ2]]([[ARG3]]) : !rtg.sequence<index>
# CHECK-NEXT:   [[RAND2:%.+]] = rtg.randomize_sequence [[SUB3]]
# CHECK-NEXT:   rtg.embed_sequence [[RAND2]]
# CHECK-NEXT: }


@test(Tgt0)
def test0_context_args(config):
  c = Integer(4)
  with config.cpu0:
    consumer(config.a)
    consumer(c)
  consumer(config.b)


# CHECK-LABEL: rtg.test @test1_context_nested
# CHECK-NEXT:   [[V1:%.+]] = rtg.get_sequence @_context_seq_1 : !rtg.sequence<index, index, !rtgtest.cpu, !rtgtest.cpu>
# CHECK-NEXT:   [[V2:%.+]] = rtg.substitute_sequence [[V1]](%a, %b, %cpu0, %cpu1) : !rtg.sequence<index, index, !rtgtest.cpu, !rtgtest.cpu>
# CHECK-NEXT:   rtg.on_context %cpu0, [[V2]] : !rtgtest.cpu

#      CHECK: rtg.sequence @_context_seq_1([[ARG0:%.+]]: index, [[ARG1:%.+]]: index, [[ARG2:%.+]]: !rtgtest.cpu, [[ARG3:%.+]]: !rtgtest.cpu) {
# CHECK-NEXT:   [[V0:%.+]] = rtg.get_sequence @consumer : !rtg.sequence<index>
# CHECK-NEXT:   [[V1:%.+]] = rtg.substitute_sequence [[V0]]([[ARG0]]) : !rtg.sequence<index>
# CHECK-NEXT:   [[V2:%.+]] = rtg.randomize_sequence [[V1]]
# CHECK-NEXT:   rtg.embed_sequence [[V2]]
# CHECK-NEXT:   [[V3:%.+]] = rtg.get_sequence @_context_seq_2 : !rtg.sequence<index, index, !rtgtest.cpu, !rtgtest.cpu>
# CHECK-NEXT:   [[V4:%.+]] = rtg.substitute_sequence [[V3]]([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]]) : !rtg.sequence<index, index, !rtgtest.cpu, !rtgtest.cpu>
# CHECK-NEXT:   rtg.on_context [[ARG3]], [[V4]] : !rtgtest.cpu
# CHECK-NEXT:   [[V5:%.+]] = rtg.get_sequence @_context_seq_3 : !rtg.sequence<index, index, !rtgtest.cpu, !rtgtest.cpu>
# CHECK-NEXT:   [[V6:%.+]] = rtg.substitute_sequence [[V5]]([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]]) : !rtg.sequence<index, index, !rtgtest.cpu, !rtgtest.cpu>
# CHECK-NEXT:   rtg.on_context [[ARG2]], [[V6]] : !rtgtest.cpu
# CHECK-NEXT: }
# CHECK-NEXT: rtg.sequence @_context_seq_2([[ARG0:%.+]]: index, [[ARG1:%.+]]: index, [[ARG2:%.+]]: !rtgtest.cpu, [[ARG3:%.+]]: !rtgtest.cpu) {
# CHECK-NEXT:   [[V0:%.+]] = rtg.get_sequence @consumer : !rtg.sequence<index>
# CHECK-NEXT:   [[V1:%.+]] = rtg.substitute_sequence [[V0]]([[ARG1]]) : !rtg.sequence<index>
# CHECK-NEXT:   [[V2:%.+]] = rtg.randomize_sequence [[V1]]
# CHECK-NEXT:   rtg.embed_sequence [[V2]]
# CHECK-NEXT: }
# CHECK-NEXT: rtg.sequence @_context_seq_3([[ARG0:%.+]]: index, [[ARG1:%.+]]: index, [[ARG2:%.+]]: !rtgtest.cpu, [[ARG3:%.+]]: !rtgtest.cpu) {
# CHECK-NEXT:   [[V0:%.+]] = rtg.get_sequence @consumer : !rtg.sequence<index>
# CHECK-NEXT:   [[V1:%.+]] = rtg.substitute_sequence [[V0]]([[ARG1]]) : !rtg.sequence<index>
# CHECK-NEXT:   [[V2:%.+]] = rtg.randomize_sequence [[V1]]
# CHECK-NEXT:   rtg.embed_sequence [[V2]]
# CHECK-NEXT: }


@test(Tgt0)
def test1_context_nested(config):
  with config.cpu0:
    consumer(config.a)
    with config.cpu1:
      consumer(config.b)
    with config.cpu0:
      consumer(config.b)
