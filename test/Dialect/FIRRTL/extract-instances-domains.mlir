// RUN: circt-opt --firrtl-extract-instances %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Domains
//===----------------------------------------------------------------------===//

// Domain ports of an extracted instance are wired with domain defines.
// The ports added to each parent mirror the instance's ports, including their
// domain associations.
//
// CHECK-LABEL: firrtl.circuit "ExtractBlackBoxesDomains"
firrtl.circuit "ExtractBlackBoxesDomains" {
  firrtl.domain @ClockDomain
  firrtl.extmodule private @MyBlackBox(
    in A: !firrtl.domain<@ClockDomain()>,
    in in: !firrtl.uint<8> domains [A],
    out out: !firrtl.uint<8> domains [A]
  ) attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation", filename = "BlackBoxes.txt", prefix = "bb"}], defname = "MyBlackBox"}
  // CHECK-LABEL: firrtl.module private @Child(
  // CHECK-SAME:    in %D: !firrtl.domain<@ClockDomain()>
  // CHECK-SAME:    in %in: !firrtl.uint<8> domains [%D]
  // CHECK-SAME:    out %out: !firrtl.uint<8> domains [%D]
  // CHECK-SAME:    out %bb_0_A: !firrtl.domain<@ClockDomain()>
  // CHECK-SAME:    out %bb_0_in: !firrtl.uint<8> domains [%bb_0_A]
  // CHECK-SAME:    in %bb_0_out: !firrtl.uint<8> domains [%bb_0_A]
  // CHECK-NEXT:    firrtl.domain.define %bb_0_A, %D
  // CHECK-NEXT:    firrtl.matchingconnect %bb_0_in, %in
  // CHECK-NEXT:    firrtl.matchingconnect %out, %bb_0_out
  // CHECK-NEXT:  }
  firrtl.module private @Child(
    in %D: !firrtl.domain<@ClockDomain()>,
    in %in: !firrtl.uint<8> domains [%D],
    out %out: !firrtl.uint<8> domains [%D]
  ) {
    %bb_A, %bb_in, %bb_out = firrtl.instance bb @MyBlackBox(in A: !firrtl.domain<@ClockDomain()>, in in: !firrtl.uint<8> domains [A], out out: !firrtl.uint<8> domains [A])
    firrtl.domain.define %bb_A, %D : !firrtl.domain<@ClockDomain()>
    firrtl.matchingconnect %bb_in, %in : !firrtl.uint<8>
    firrtl.matchingconnect %out, %bb_out : !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module private @DUTModule(
  // CHECK-SAME:    in %D: !firrtl.domain<@ClockDomain()>
  // CHECK-SAME:    in %clock: !firrtl.clock domains [%D]
  // CHECK-SAME:    in %in: !firrtl.uint<8> domains [%D]
  // CHECK-SAME:    out %out: !firrtl.uint<8> domains [%D]
  // CHECK-SAME:    out %bb_0_A: !firrtl.domain<@ClockDomain()>
  // CHECK-SAME:    out %bb_0_in: !firrtl.uint<8> domains [%bb_0_A]
  // CHECK-SAME:    in %bb_0_out: !firrtl.uint<8> domains [%bb_0_A]
  // CHECK-NEXT:    firrtl.instance child
  // CHECK-SAME:      out bb_0_A: !firrtl.domain<@ClockDomain()>
  // CHECK-SAME:      out bb_0_in: !firrtl.uint<8> domains [bb_0_A]
  // CHECK-SAME:      in bb_0_out: !firrtl.uint<8> domains [bb_0_A]
  // CHECK-NEXT:    firrtl.domain.define %bb_0_A, %child_bb_0_A
  // CHECK-NEXT:    firrtl.matchingconnect %bb_0_in, %child_bb_0_in
  // CHECK-NEXT:    firrtl.matchingconnect %child_bb_0_out, %bb_0_out
  firrtl.module private @DUTModule(
    in %D: !firrtl.domain<@ClockDomain()>,
    in %clock: !firrtl.clock domains [%D],
    in %in: !firrtl.uint<8> domains [%D],
    out %out: !firrtl.uint<8> domains [%D]
  ) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %child_D, %child_in, %child_out = firrtl.instance child @Child(in D: !firrtl.domain<@ClockDomain()>, in in: !firrtl.uint<8> domains [D], out out: !firrtl.uint<8> domains [D])
    firrtl.domain.define %child_D, %D : !firrtl.domain<@ClockDomain()>
    firrtl.matchingconnect %child_in, %in : !firrtl.uint<8>
    firrtl.matchingconnect %out, %child_out : !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @ExtractBlackBoxesDomains(
  // CHECK-NEXT:    firrtl.instance dut
  // CHECK-SAME:      out bb_0_A: !firrtl.domain<@ClockDomain()>
  // CHECK-SAME:      out bb_0_in: !firrtl.uint<8> domains [bb_0_A]
  // CHECK-SAME:      in bb_0_out: !firrtl.uint<8> domains [bb_0_A]
  // CHECK-NEXT:    %bb_A, %bb_in, %bb_out = firrtl.instance bb @MyBlackBox
  // CHECK-NEXT:    firrtl.domain.define %bb_A, %dut_bb_0_A
  // CHECK-NEXT:    firrtl.matchingconnect %bb_in, %dut_bb_0_in
  // CHECK-NEXT:    firrtl.matchingconnect %dut_bb_0_out, %bb_out
  firrtl.module @ExtractBlackBoxesDomains(
    in %D: !firrtl.domain<@ClockDomain()>,
    in %clock: !firrtl.clock domains [%D],
    in %in: !firrtl.uint<8> domains [%D],
    out %out: !firrtl.uint<8> domains [%D]
  ) {
    %dut_D, %dut_clock, %dut_in, %dut_out = firrtl.instance dut @DUTModule(in D: !firrtl.domain<@ClockDomain()>, in clock: !firrtl.clock domains [D], in in: !firrtl.uint<8> domains [D], out out: !firrtl.uint<8> domains [D])
    firrtl.domain.define %dut_D, %D : !firrtl.domain<@ClockDomain()>
    firrtl.matchingconnect %dut_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %dut_in, %in : !firrtl.uint<8>
    firrtl.matchingconnect %out, %dut_out : !firrtl.uint<8>
  }
}

// CHECK-LABEL: firrtl.circuit "ExtractClockGatesDomains"
firrtl.circuit "ExtractClockGatesDomains" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation", filename = "ClockGates.txt"}]} {
  firrtl.domain @ClockDomain
  firrtl.extmodule private @EICG_wrapper(
    in D: !firrtl.domain<@ClockDomain()>,
    in in: !firrtl.clock domains [D],
    in en: !firrtl.uint<1> domains [D],
    out out: !firrtl.clock domains [D]
  ) attributes {defname = "EICG_wrapper"}
  // CHECK-LABEL: firrtl.module private @DUTModule(
  // CHECK-SAME:    out %clock_gate_0_D: !firrtl.domain<@ClockDomain()>
  // CHECK-SAME:    out %clock_gate_0_in: !firrtl.clock domains [%clock_gate_0_D]
  // CHECK-SAME:    out %clock_gate_0_en: !firrtl.uint<1> domains [%clock_gate_0_D]
  // CHECK-SAME:    in %clock_gate_0_out: !firrtl.clock domains [%clock_gate_0_D]
  // CHECK-NEXT:    firrtl.domain.define %clock_gate_0_D, %D
  // CHECK-NEXT:    firrtl.matchingconnect %clock_gate_0_in, %clock
  // CHECK-NEXT:    firrtl.matchingconnect %clock_gate_0_en, %en
  // CHECK-NEXT:  }
  firrtl.module private @DUTModule(
    in %D: !firrtl.domain<@ClockDomain()>,
    in %clock: !firrtl.clock domains [%D],
    in %en: !firrtl.uint<1> domains [%D]
  ) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %gate_D, %gate_in, %gate_en, %gate_out = firrtl.instance gate @EICG_wrapper(in D: !firrtl.domain<@ClockDomain()>, in in: !firrtl.clock domains [D], in en: !firrtl.uint<1> domains [D], out out: !firrtl.clock domains [D])
    firrtl.domain.define %gate_D, %D : !firrtl.domain<@ClockDomain()>
    firrtl.matchingconnect %gate_in, %clock : !firrtl.clock
    firrtl.matchingconnect %gate_en, %en : !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @ExtractClockGatesDomains(
  // CHECK-NEXT:    firrtl.instance dut
  // CHECK-NEXT:    %gate_D, %gate_in, %gate_en, %gate_out = firrtl.instance gate @EICG_wrapper
  // CHECK-NEXT:    firrtl.domain.define %gate_D, %dut_clock_gate_0_D
  // CHECK-NEXT:    firrtl.matchingconnect %gate_in, %dut_clock_gate_0_in
  // CHECK-NEXT:    firrtl.matchingconnect %gate_en, %dut_clock_gate_0_en
  // CHECK-NEXT:    firrtl.matchingconnect %dut_clock_gate_0_out, %gate_out
  firrtl.module @ExtractClockGatesDomains(
    in %D: !firrtl.domain<@ClockDomain()>,
    in %clock: !firrtl.clock domains [%D],
    in %en: !firrtl.uint<1> domains [%D]
  ) {
    %dut_D, %dut_clock, %dut_en = firrtl.instance dut @DUTModule(in D: !firrtl.domain<@ClockDomain()>, in clock: !firrtl.clock domains [D], in en: !firrtl.uint<1> domains [D])
    firrtl.domain.define %dut_D, %D : !firrtl.domain<@ClockDomain()>
    firrtl.matchingconnect %dut_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %dut_en, %en : !firrtl.uint<1>
  }
}

// The wrapper module's ports mirror the grouped instances' ports,
// including their domain associations.
//
// The memory without domain ports adds ports without domain associations.
//
// CHECK-LABEL: firrtl.circuit "ExtractSeqMemsDomains"
firrtl.circuit "ExtractSeqMemsDomains" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation", filename = "SeqMems.txt", group = "Memories"}]} {
  firrtl.domain @ClockDomain
  firrtl.memmodule @domain_ext(in D: !firrtl.domain<@ClockDomain()>, in W0_clk: !firrtl.clock domains [D]) attributes {dataWidth = 8 : ui32, depth = 8 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  firrtl.memmodule @plain_ext(in W0_clk: !firrtl.clock) attributes {dataWidth = 8 : ui32, depth = 8 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  // CHECK-LABEL: firrtl.module private @mem(
  // CHECK-SAME:    out %mem_wiring_0_W0_clk: !firrtl.clock,
  // CHECK-SAME:    out %mem_wiring_1_D: !firrtl.domain<@ClockDomain()>
  // CHECK-SAME:    out %mem_wiring_1_W0_clk: !firrtl.clock domains [%mem_wiring_1_D]
  // CHECK-NEXT:    firrtl.domain.define %mem_wiring_1_D, %D
  // CHECK-NEXT:    firrtl.matchingconnect %mem_wiring_1_W0_clk, %clock
  // CHECK-NEXT:    firrtl.matchingconnect %mem_wiring_0_W0_clk, %clock
  // CHECK-NEXT:  }
  firrtl.module private @mem(
    in %D: !firrtl.domain<@ClockDomain()>,
    in %clock: !firrtl.clock domains [%D]
  ) {
    %domain_D, %domain_W0_clk = firrtl.instance domain_ext @domain_ext(in D: !firrtl.domain<@ClockDomain()>, in W0_clk: !firrtl.clock domains [D])
    %plain_W0_clk = firrtl.instance plain_ext @plain_ext(in W0_clk: !firrtl.clock)
    firrtl.domain.define %domain_D, %D : !firrtl.domain<@ClockDomain()>
    firrtl.matchingconnect %domain_W0_clk, %clock : !firrtl.clock
    firrtl.matchingconnect %plain_W0_clk, %clock : !firrtl.clock
  }
  // CHECK-LABEL: firrtl.module private @Memories(
  // CHECK-SAME:    in %mem_wiring_0_W0_clk: !firrtl.clock,
  // CHECK-SAME:    in %mem_wiring_1_D: !firrtl.domain<@ClockDomain()>
  // CHECK-SAME:    in %mem_wiring_1_W0_clk: !firrtl.clock domains [%mem_wiring_1_D]
  // CHECK-NEXT:    %plain_ext_W0_clk = firrtl.instance plain_ext @plain_ext
  // CHECK-NEXT:    firrtl.matchingconnect %plain_ext_W0_clk, %mem_wiring_0_W0_clk
  // CHECK-NEXT:    %domain_ext_D, %domain_ext_W0_clk = firrtl.instance domain_ext {{.*}}@domain_ext
  // CHECK-NEXT:    firrtl.domain.define %domain_ext_D, %mem_wiring_1_D
  // CHECK-NEXT:    firrtl.matchingconnect %domain_ext_W0_clk, %mem_wiring_1_W0_clk
  // CHECK-NEXT:  }
  // CHECK-LABEL: firrtl.module private @DUTModule(
  // CHECK-NEXT:    firrtl.instance Memories sym @Memories @Memories(
  // CHECK-SAME:      in mem_wiring_0_W0_clk: !firrtl.clock,
  // CHECK-SAME:      in mem_wiring_1_D: !firrtl.domain<@ClockDomain()>
  // CHECK-SAME:      in mem_wiring_1_W0_clk: !firrtl.clock domains [mem_wiring_1_D]
  // CHECK-NEXT:    firrtl.instance mem
  // CHECK-NEXT:    firrtl.domain.define %Memories_mem_wiring_1_D, %mem_mem_wiring_1_D
  // CHECK-NEXT:    firrtl.matchingconnect %Memories_mem_wiring_1_W0_clk, %mem_mem_wiring_1_W0_clk
  // CHECK-NEXT:    firrtl.matchingconnect %Memories_mem_wiring_0_W0_clk, %mem_mem_wiring_0_W0_clk
  firrtl.module private @DUTModule(
    in %D: !firrtl.domain<@ClockDomain()>,
    in %clock: !firrtl.clock domains [%D]
  ) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %mem_D, %mem_clock = firrtl.instance mem @mem(in D: !firrtl.domain<@ClockDomain()>, in clock: !firrtl.clock domains [D])
    firrtl.domain.define %mem_D, %D : !firrtl.domain<@ClockDomain()>
    firrtl.matchingconnect %mem_clock, %clock : !firrtl.clock
  }
  firrtl.module @ExtractSeqMemsDomains(
    in %D: !firrtl.domain<@ClockDomain()>,
    in %clock: !firrtl.clock domains [%D]
  ) {
    %dut_D, %dut_clock = firrtl.instance dut @DUTModule(in D: !firrtl.domain<@ClockDomain()>, in clock: !firrtl.clock domains [D])
    firrtl.domain.define %dut_D, %D : !firrtl.domain<@ClockDomain()>
    firrtl.matchingconnect %dut_clock, %clock : !firrtl.clock
  }
}
