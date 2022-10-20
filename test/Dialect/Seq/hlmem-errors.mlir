// RUN: circt-opt %s -verify-diagnostics --split-input-file

hw.module @MultipleUse(%input: i1, %clk: i1) {
  // expected-error@+1 {{'seq.hlmem' op output port #0 has multiple uses.}}
  %read0, %write0 = seq.hlmem @myMemory %clk {
    NReadPorts = 1 : i32,
    NWritePorts = 1 : i32,
    readLatency = 0 : i32,
    writeLatency = 1 : i32} : !hw.array<4xi32>
  
  %c0_i2 = hw.constant 0 : i2
  %c42_i32 = hw.constant 42 : i32

  %out0 = seq.read %read0[%c0_i2] : !seq.read_port<<4xi32>>
  %out1 = seq.read %read0[%c0_i2] : !seq.read_port<<4xi32>>
}
