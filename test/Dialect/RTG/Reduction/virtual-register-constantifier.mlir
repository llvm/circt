// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "rtg.test @test" --keep-best=0 --include rtg-virtual-register-constantifier | FileCheck %s

// CHECK-LABEL: rtg.test @test
rtg.test @test() {
  // CHECK-NEXT: [[V0:%.+]] = rtg.constant #rtgtest.t0 : !rtgtest.ireg
  // CHECK-NEXT: rtgtest.rv32i.add [[V0]], [[V0]], [[V0]]
  %reg = rtg.virtual_reg [#rtgtest.t0, #rtgtest.t1, #rtgtest.t2]
  rtgtest.rv32i.add %reg, %reg, %reg
}
