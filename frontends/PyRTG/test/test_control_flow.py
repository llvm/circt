# RUN: %rtgtool% %s --seed=0 --output-format=mlir | FileCheck %s

from pyrtg import test, sequence, Integer, Array, Bool, If, Else, EndIf, For, Foreach, config, Config, Param


@sequence(Integer.ty())
def consumer(arg):
  pass


@config
class Config0(Config):

  a = Param(loader=lambda: Integer(0))
  b = Param(loader=lambda: Integer(1))
  cond = Param(loader=lambda: Bool(True))
  cond2 = Param(loader=lambda: Bool(False))


# CHECK-LABEL: rtg.test @test0_if_nested
# CHECK-NEXT: [[V0:%.+]] = index.add %a, %b
# CHECK-NEXT: [[V1:%.+]]:2 = scf.if %cond -> (index, index) {
# CHECK-NEXT:   [[V8:%.+]] = arith.select %cond2, %a, %b : index
# CHECK-NEXT:   scf.if %cond2 {
# CHECK-NEXT:   } else {
# CHECK-NEXT:     [[V11:%.+]] = rtg.get_sequence @consumer : !rtg.sequence<index>
# CHECK-NEXT:     [[V12:%.+]] = rtg.substitute_sequence [[V11]]([[V0]]) : !rtg.sequence<index>
# CHECK-NEXT:     [[V13:%.+]] = rtg.randomize_sequence [[V12]]
# CHECK-NEXT:     rtg.embed_sequence [[V13]]
# CHECK-NEXT:   }
# CHECK-NEXT:   [[V9:%.+]] = index.add %a, [[V8]]
# CHECK-NEXT:   [[V10:%.+]] = index.add [[V9]], %b
# CHECK-NEXT:   scf.yield [[V10]], [[V9]] : index, index
# CHECK-NEXT: } else {
# CHECK-NEXT:   [[V8:%.+]] = arith.select %cond2, %a, %b : index
# CHECK-NEXT:   [[V9:%.+]] = index.add %b, [[V8]]
# CHECK-NEXT:   [[V10:%.+]] = index.add [[V9]], %a
# CHECK-NEXT:   scf.yield [[V10]], [[V9]] : index, index
# CHECK-NEXT: }
#      CHECK: rtg.substitute_sequence {{.+}}([[V1]]#0) : !rtg.sequence<index>
#      CHECK: [[V5:%.+]] = index.add [[V1]]#1, [[V0]]
# CHECK-NEXT: rtg.substitute_sequence {{.+}}([[V5]]) : !rtg.sequence<index>


@test(Config0)
def test0_if_nested(config):
  w = config.a + config.b
  with If(config.cond):
    with If(config.cond2):
      x = config.a
    with Else():
      x = config.b
      consumer(w)
    EndIf()
    v = config.a + x
    u = v + config.b
  with Else():
    with If(config.cond2):
      p = config.a
    with Else():
      p = config.b
    EndIf()
    v = config.b + p
    u = v + config.a
  EndIf()

  consumer(u)
  consumer(v + w)


# CHECK-LABEL: rtg.test @test1_if_default
# CHECK-NEXT: [[V0:%.+]] = arith.andi %cond, %cond2 : i1
# CHECK-NEXT: [[V1:%.+]] = arith.select [[V0]], %b, %a : index
# CHECK-NEXT: scf.if [[V0]] {
# CHECK-NEXT:   [[V5:%.+]] = rtg.get_sequence @consumer : !rtg.sequence<index>
# CHECK-NEXT:   [[V6:%.+]] = rtg.substitute_sequence [[V5]](%b) : !rtg.sequence<index>
# CHECK-NEXT:   [[V7:%.+]] = rtg.randomize_sequence [[V6]]
# CHECK-NEXT:   rtg.embed_sequence [[V7]]
# CHECK-NEXT: }
#      CHECK: rtg.substitute_sequence {{.+}}([[V1]]) : !rtg.sequence<index>


@test(Config0)
def test1_if_default(config):
  v = config.a
  with If(config.cond):
    with If(config.cond2):
      v = config.b
      consumer(v)
    EndIf()
  EndIf()
  consumer(v)


# CHECK-LABEL: rtg.test @test2_for
# CHECK-NEXT: [[IDX1:%.+]] = index.constant 1
# CHECK-NEXT: [[IDX0:%.+]] = index.constant 0
# CHECK-NEXT: [[RES0:%.+]]:2 = scf.for [[ARG0:%.+]] = %lower to %upper step %step iter_args([[ARG1:%.+]] = %iter_arg0, [[ARG2:%.+]] = %iter_arg1) -> (index, index) {
# CHECK-NEXT:   [[VAL7:%.+]] = index.add [[ARG1]], [[ARG2]]
# CHECK-NEXT:   [[VAL8:%.+]] = index.add [[ARG2]], [[ARG0]]
# CHECK-NEXT:   scf.yield [[VAL7]], [[VAL8]] : index, index
# CHECK-NEXT: }
#      CHECK: rtg.substitute_sequence {{%.+}}([[RES0]]#0) : !rtg.sequence<index>
#      CHECK: [[RES4:%.+]] = scf.for [[ARG0:%.+]] = [[IDX0]] to %upper step [[IDX1]] iter_args([[ARG1:%.+]] = [[IDX0]]) -> (index) {
# CHECK-NEXT:   [[VAL7:%.+]] = index.add [[ARG1]], [[ARG0]]
# CHECK-NEXT:   scf.yield [[VAL7]] : index
# CHECK-NEXT: }
#      CHECK: rtg.substitute_sequence {{%.+}}([[RES4]]) : !rtg.sequence<index>


@config
class Config1(Config):

  iter_arg0 = Param(loader=lambda: Integer(0))
  iter_arg1 = Param(loader=lambda: Integer(0))
  lower = Param(loader=lambda: Integer(0))
  step = Param(loader=lambda: Integer(1))
  upper = Param(loader=lambda: Integer(5))


@test(Config1)
def test2_for(config):
  with For(config.lower, config.upper, config.step) as i:
    config.iter_arg0 += config.iter_arg1
    config.iter_arg1 += i
  consumer(config.iter_arg0)

  v = Integer(0)
  with For(config.upper) as i:
    v += i
  consumer(v)


# CHECK-LABEL: rtg.test @test2_for
# CHECK-NEXT: [[IDX0:%.+]] = index.constant 0
# CHECK-NEXT: [[IDX1:%.+]] = index.constant 1
# CHECK-NEXT: [[ARR:%.+]] = rtg.array_create [[IDX1]], [[IDX1]], [[IDX1]] : index
# CHECK-NEXT: [[UPPER:%.+]] = rtg.array_size %arr0 : !rtg.array<index>
# CHECK-NEXT: [[RESULT:%.+]] = scf.for [[ARG0:%.+]] = [[IDX0]] to [[UPPER]] step [[IDX1]] iter_args([[ARG1:%.+]] = [[IDX0]]) -> (index) {
# CHECK-NEXT:   [[VAL1:%.+]] = rtg.array_extract %arr0[[[ARG0]]] : !rtg.array<index>
# CHECK-NEXT:   [[VAL2:%.+]] = rtg.array_extract [[ARR]][[[ARG0]]] : !rtg.array<index>
# CHECK-NEXT:   [[VAL3:%.+]] = index.add [[VAL1]], [[VAL2]]
# CHECK-NEXT:   [[VAL4:%.+]] = index.add [[VAL3]], [[ARG0]]
# CHECK-NEXT:   [[VAL5:%.+]] = index.add [[ARG1]], [[VAL4]]
# CHECK-NEXT:   scf.yield [[VAL5]] : index
#      CHECK: rtg.substitute_sequence {{.*}}([[RESULT]]) : !rtg.sequence<index>


@config
class Config2(Config):

  arr0 = Param(loader=lambda: Array.create([Integer(0)], Integer.ty()))


@test(Config2)
def test2_foreach(config):
  v = Integer(0)
  with Foreach(config.arr0,
               Array.create([Integer(1) for _ in range(3)],
                            Integer.ty())) as (i, a0, a1):
    v += a0 + a1 + i
  consumer(v)
