// RUN: circt-verilog %s | FileCheck %s --check-prefixes=CHECK,MEMON
// RUN: circt-verilog --detect-memories=1 %s | FileCheck %s --check-prefixes=CHECK,MEMON
// RUN: circt-verilog --detect-memories=0 %s | FileCheck %s --check-prefixes=CHECK,MEMOFF
// REQUIRES: slang
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK-LABEL: hw.module @Memory(
module Memory(
  input  bit clock,
  input  bit [3:0] waddr,
  input  bit [41:0] wdata,
  input  bit wenable,
  input  bit [3:0] raddr,
  output bit [41:0] rdata
);
  // CHECK-DAG: [[CLK:%.+]] = seq.to_clock %clock

  // MEMON-DAG: [[MEM:%.+]] = seq.firmem 0, 1, undefined, undefined : <16 x 42, mask 1>
  // MEMON-DAG: [[RDATA:%.+]] = seq.firmem.read_port [[MEM]][%raddr]
  // MEMON-DAG: seq.firmem.write_port %mem[%waddr] = %wdata, clock [[CLK]] enable %wenable

  // MEMOFF-DAG: [[REG:%.+]] = seq.firreg [[NEXT:%.+]] clock [[CLK]] : !hw.array<16xi42>
  // MEMOFF-DAG: [[TMP:%.+]] = hw.array_inject [[REG]][%waddr], %wdata
  // MEMOFF-DAG: [[NEXT]] = comb.mux bin %wenable, [[TMP]], [[REG]]
  // MEMOFF-DAG: [[RDATA:%.+]] = hw.array_get [[REG]][%raddr]

  // CHECK: hw.output [[RDATA]]
  bit [41:0] storage [15:0];
  always_ff @(posedge clock)
    if (wenable)
      storage[waddr] <= wdata;
  assign rdata = storage[raddr];
endmodule
