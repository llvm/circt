// RUN: circt-opt %s | circt-opt | FileCheck %s

hw.module @mod(%clk : i1) -> () {
// CHECK: %read0, %write0 = seq.hlmem @myMemory %clk {NReadPorts = 1 : i32, NWritePorts = 1 : i32, readLatency = 0 : i32, writeLatency = 1 : i32} : !hw.array<4xi32>
  %read0, %write0 = seq.hlmem @myMemory %clk {
    NReadPorts = 1 : i32,
    NWritePorts = 1 : i32,
    readLatency = 0 : i32,
    writeLatency = 1 : i32} : !hw.array<4xi32>
  
  %c0_i2 = hw.constant 0 : i2
  %c42_i32 = hw.constant 42 : i32

// CHECK: %data = seq.read %read0[%c0_i2] : !seq.read_port<<4xi32>>
  %out = seq.read %read0[%c0_i2] : !seq.read_port<<4xi32>>

// CHECK: seq.write %write0[%c0_i2] %c42_i32 : !seq.write_port<<4xi32>>
  seq.write %write0[%c0_i2] %c42_i32 : !seq.write_port<<4xi32>>
  hw.output
}
