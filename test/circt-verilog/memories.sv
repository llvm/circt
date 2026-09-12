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

// CHECK-LABEL: hw.module @MaskedMemory(
module MaskedMemory(
  input  bit clock,
  input  bit [1:0] addr,
  input  bit [31:0] wdata,
  input  bit wenable,
  input  bit [3:0] wmask,
  output bit [31:0] rdata
);
  // MEMON: [[MASKED_MEM:%.+]] = seq.firmem 0, 1, undefined, undefined : <4 x 32, mask 4>
  // MEMON: [[MASKED_READ:%.+]] = seq.firmem.read_port [[MASKED_MEM]]
  // MEMON: seq.firmem.write_port [[MASKED_MEM]]{{.*}}mask
  // MEMON-NOT: hw.array_inject

  // MEMOFF-DAG: hw.array_inject
  // MEMOFF-DAG: seq.firreg {{.*}} : !hw.array<4xi32>

  // CHECK: hw.output %{{.*}}
  bit [31:0] storage [3:0];
  always_ff @(posedge clock) begin
    if (wenable && wmask[0])
      storage[addr][7:0] <= wdata[7:0];
    if (wenable && wmask[1])
      storage[addr][15:8] <= wdata[15:8];
    if (wenable && wmask[2])
      storage[addr][23:16] <= wdata[23:16];
    if (wenable && wmask[3])
      storage[addr][31:24] <= wdata[31:24];
  end
  assign rdata = storage[addr];
endmodule
