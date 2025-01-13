// RUN: circt-opt --lower-to-bmc="top-module=comb bound=10" %s | FileCheck %s

// CHECK:  llvm.func @printf(!llvm.ptr, ...)
// CHECK:  func.func @comb() {
// CHECK:    [[BMC:%.+]] = verif.bmc bound 20 num_regs 0 initial_values [] init {
// CHECK:    } loop {
// CHECK:    } circuit {
// CHECK:    ^bb0([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32):
// CHECK:      [[OP0:%.+]] = comb.add [[ARG0]], [[ARG1]]
// CHECK:      [[OP1:%.+]] = comb.icmp eq [[OP0]], [[ARG0]]
// CHECK:      verif.assert [[OP1]]
// CHECK:      verif.yield [[OP0]]
// CHECK:    }
// CHECK:    [[SSTR_ADDR:%.+]] = llvm.mlir.addressof [[SSTR:@.+]] : !llvm.ptr
// CHECK:    [[FSTR_ADDR:%.+]] = llvm.mlir.addressof [[FSTR:@.+]] : !llvm.ptr
// CHECK:    [[SEL:%.+]] = llvm.select [[BMC]], [[SSTR_ADDR]], [[FSTR_ADDR]]
// CHECK:    llvm.call @printf([[SEL]])
// CHECK:    return
// CHECK:  }
// CHECK:  llvm.mlir.global private constant [[SSTR]]("Bound reached with no violations!\0A\00") {addr_space = 0 : i32}
// CHECK:  llvm.mlir.global private constant [[FSTR]]("Assertion can be violated!\0A\00") {addr_space = 0 : i32}

hw.module @comb(in %in0: i32, in %in1: i32, out out: i32) attributes {num_regs = 0 : i32, initial_values = []} {
  %0 = comb.add %in0, %in1 : i32
  %prop = comb.icmp eq %0, %in0 : i32
  verif.assert %prop : i1
  hw.output %0 : i32
}

// RUN: circt-opt --lower-to-bmc="top-module=seq bound=10" %s | FileCheck %s --check-prefix=CHECK1

// CHECK1:  llvm.func @printf(!llvm.ptr, ...)
// CHECK1:  func.func @seq() {
// CHECK1:    [[BMC:%.+]] = verif.bmc bound 20 num_regs 1 initial_values [unit] init {
// CHECK1:      [[FALSE:%.+]] = hw.constant false
// CHECK1:      [[INIT_CLK:%.+]] = seq.to_clock [[FALSE]]
// CHECK1:      verif.yield [[INIT_CLK]]
// CHECK1:    } loop {
// CHECK1:    ^bb0([[CLK:%.+]]: !seq.clock):
// CHECK1:      [[FROM_CLK:%.+]] = seq.from_clock [[CLK]]
// CHECK1:      [[TRUE:%.+]] = hw.constant true
// CHECK1:      [[NCLK:%.+]] = comb.xor [[FROM_CLK]], [[TRUE]]
// CHECK1:      [[NEW_CLOCK:%.+]] = seq.to_clock [[NCLK]]
// CHECK1:      verif.yield [[NEW_CLOCK]]
// CHECK1:    } circuit {
// CHECK1:    ^bb0([[CLK:%.+]]: !seq.clock, [[ARG1:%.+]]: i32, [[ARG2:%.+]]: i32, [[ARG3:%.+]]: i32):
// CHECK1:      [[OP0:%.+]] = comb.add [[ARG1]], [[ARG2]]
// CHECK1:      [[OP2:%.+]] = comb.icmp eq [[OP0]], [[ARG1]]
// CHECK1:      verif.assert [[OP2]]
// CHECK1:      verif.yield [[ARG3]], [[OP0]]
// CHECK1:    }
// CHECK1:    [[SSTR_ADDR:%.+]] = llvm.mlir.addressof [[SSTR:@.+]] : !llvm.ptr
// CHECK1:    [[FSTR_ADDR:%.+]] = llvm.mlir.addressof [[FSTR:@.+]] : !llvm.ptr
// CHECK1:    [[SEL:%.+]] = llvm.select [[BMC]], [[SSTR_ADDR]], [[FSTR_ADDR]]
// CHECK1:    llvm.call @printf([[SEL]])
// CHECK1:    return
// CHECK1:  }
// CHECK1:  llvm.mlir.global private constant [[SSTR]]("Bound reached with no violations!\0A\00") {addr_space = 0 : i32}
// CHECK1:  llvm.mlir.global private constant [[FSTR]]("Assertion can be violated!\0A\00") {addr_space = 0 : i32}
hw.module @seq(in %clk : !seq.clock, in %in0 : i32, in %in1 : i32, in %reg_state : i32, out out : i32, out reg_input : i32) attributes {num_regs = 1 : i32, initial_values = [unit]} {
  %0 = comb.add %in0, %in1 : i32
  %1 = comb.icmp eq %0, %in0 : i32
  verif.assert %1 : i1
  hw.output %reg_state, %0 : i32, i32
}

// RUN: circt-opt --lower-to-bmc="top-module=nondominance bound=10" %s | FileCheck %s --check-prefix=CHECK2

// CHECK2:  llvm.func @printf(!llvm.ptr, ...)
// CHECK2:  func.func @nondominance() {
// CHECK2:    [[BMC:%.+]] = verif.bmc bound 20 num_regs 1 initial_values [unit] init {
// CHECK2:      [[FALSE:%.+]] = hw.constant false
// CHECK2:      [[INIT_CLK:%.+]] = seq.to_clock [[FALSE]]
// CHECK2:      verif.yield [[INIT_CLK]]
// CHECK2:    } loop {
// CHECK2:    ^bb0([[CLK:%.+]]: !seq.clock):
// CHECK2:      [[FROM_CLK:%.+]] = seq.from_clock [[CLK]]
// CHECK2:      [[TRUE:%.+]] = hw.constant true
// CHECK2:      [[NCLK:%.+]] = comb.xor [[FROM_CLK]], [[TRUE]]
// CHECK2:      [[NEW_CLOCK:%.+]] = seq.to_clock [[NCLK]]
// CHECK2:      verif.yield [[NEW_CLOCK]]
// CHECK2:    } circuit {
// CHECK2:    ^bb0([[CLK:%.+]]: !seq.clock, [[ARG1:%.+]]: i32, [[ARG2:%.+]]: i32, [[ARG3:%.+]]: i32):
// CHECK2:      [[OP0:%.+]] = comb.add [[ARG1]], [[ARG2]]
// CHECK2:      [[OP2:%.+]] = comb.icmp eq [[OP0]], [[ARG1]]
// CHECK2:      verif.assert [[OP2]]
// CHECK2:      verif.yield [[ARG3]], [[OP0]]
// CHECK2:    }
// CHECK2:    [[SSTR_ADDR:%.+]] = llvm.mlir.addressof [[SSTR:@.+]] : !llvm.ptr
// CHECK2:    [[FSTR_ADDR:%.+]] = llvm.mlir.addressof [[FSTR:@.+]] : !llvm.ptr
// CHECK2:    [[SEL:%.+]] = llvm.select [[BMC]], [[SSTR_ADDR]], [[FSTR_ADDR]]
// CHECK2:    llvm.call @printf([[SEL]])
// CHECK2:    return
// CHECK2:  }
// CHECK2:  llvm.mlir.global private constant [[SSTR]]("Bound reached with no violations!\0A\00") {addr_space = 0 : i32}
// CHECK2:  llvm.mlir.global private constant [[FSTR]]("Assertion can be violated!\0A\00") {addr_space = 0 : i32}
hw.module @nondominance(in %clk : !seq.clock, in %in0 : i32, in %in1 : i32, in %reg_state : i32, out out : i32, out reg_input : i32) attributes {num_regs = 1 : i32, initial_values = [unit]} {
  %0 = comb.icmp eq %1, %in0 : i32
  %1 = comb.add %in0, %in1 : i32
  verif.assert %0 : i1
  hw.output %reg_state, %1 : i32, i32
}
