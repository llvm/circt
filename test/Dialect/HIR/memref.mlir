// RUN: circt-opt %s -split-input-file -canonicalize -hir-lower-memref
#rd = {rd_latency=1}
#wr = {wr_latency=1}

hir.func @test1 at %t(
%a :!hir.memref<16x16x(bank 2)x(bank 2)xf32> ports [#rd,#wr],
%b : i32
){

  %0 = constant 0:index
  %1 = constant 1:index
  %c1_i4 = constant 1:i4
  %c1_f32 = constant 1.0:f32

  %v = hir.load %a[port 0][%c1_i4,%c1_i4,%0,%1]  at %t: !hir.memref<16x16x(bank 2)x(bank 2)xf32> delay 1
  //hir.store %v to %a[port 1][%c1_i4,%c1_i4,%0,%0] at %t + 1: !hir.memref<16x16x(bank 2)x(bank 2)xf32> delay 1

  hir.return
}{inline}

// -----

#reg_wr = {wr_latency=1}
#reg_rd = {rd_latency=0}
hir.func @test2 at %t(){
  %0 = constant 0:index
  %1 = constant 1:index
  %c1_i1 = constant 1:i1
  %c1_i2 = constant 1:i2
  %c1_f32 = constant 1.0:f32

  %a = hir.alloca("BRAM_2P") : !hir.memref<(bank 2)x(bank 3)x2x4xi8> ports [#reg_rd,#reg_wr]
  %u = hir.load %a[port 0][%0,%1,%c1_i1,%c1_i2]  at %t: !hir.memref<(bank 2)x(bank 3)x2x4xi8> delay 1
  %a_w = hir.memref.extract %a[port 1] :!hir.memref<(bank 2)x(bank 3)x2x4xi8> ports [#reg_wr]
  hir.for %i:index = %0 to %1 step %1 iter_time(%ti=%t){
    hir.store %u to %a_w[port 0][%0,%0,%c1_i1,%c1_i2] at %ti + 1: !hir.memref<(bank 2)x(bank 3)x2x4xi8> delay 1
    hir.next_iter at %ti+1
  }
  %v = hir.load %a[port 0][%0,%1,%c1_i1,%c1_i2]  at %t: !hir.memref<(bank 2)x(bank 3)x2x4xi8> delay 1
  hir.store %v to %a_w[port 0][%0,%0,%c1_i1,%c1_i2] at %t + 1: !hir.memref<(bank 2)x(bank 3)x2x4xi8> delay 1
  hir.return
}

// -----

#bram_wr = {wr_latency=1}
#bram_rd = {rd_latency=1}
hir.func @test3 at %t() -> (%res: i8){
  %0 = constant 0:index
  %1 = constant 1:index
  %c1_i1 = constant 1:i1
  %c1_i2 = constant 1:i2
  %c1_f32 = constant 1.0:f32

  %a = hir.alloca("BRAM_2P") : !hir.memref<(bank 2)x(bank 3)x2x4xi8> ports [#bram_rd,#bram_wr]
  %u = hir.load %a[port 0][%0,%1,%c1_i1,%c1_i2]  at %t: !hir.memref<(bank 2)x(bank 3)x2x4xi8> delay 1
  %v = hir.load %a[port 0][%1,%1,%c1_i1,%c1_i2]  at %t: !hir.memref<(bank 2)x(bank 3)x2x4xi8> delay 1
  %r = hir.addi (%u, %v) at %t+1:i8
  hir.return (%r) :(i8)
}
// -----

#reg_wr = {wr_latency=1}
#reg_rd = {rd_latency=0}
hir.func @test4 at %t(){
  %0 = constant 0:index
  %1 = constant 1:index
  %c1_i1 = constant 1:i1
  %c1_i2 = constant 1:i2
  %c1_f32 = constant 1.0:f32

  %a = hir.alloca("BRAM_2P") : !hir.memref<(bank 2)x(bank 3)x2x4xi8> ports [#reg_rd,#reg_wr]
  %u = hir.load %a[port 0][%0,%1,%c1_i1,%c1_i2]  at %t: !hir.memref<(bank 2)x(bank 3)x2x4xi8> delay 1
  hir.store %u to %a[port 1][%0,%0,%c1_i1,%c1_i2] at %t + 1: !hir.memref<(bank 2)x(bank 3)x2x4xi8> delay 1
  hir.return
}

// -----

#reg_wr = {wr_latency=1}
#reg_rd = {rd_latency=0}
hir.func @test5 at %t(){
  %0 = constant 0:index
  %1 = constant 1:index
  %c1_i1 = constant 1:i1
  %c1_i2 = constant 1:i2
  %c1_f32 = constant 1.0:f32

  %a = hir.alloca("BRAM_2P") : !hir.memref<(bank 2)x(bank 3)x2x4xi8> ports [#reg_rd,#reg_wr]
  %a_r = hir.memref.extract %a[port 0] :!hir.memref<(bank 2)x(bank 3)x2x4xi8> ports [#reg_rd]
  hir.call @foo(%a_r) at %t : !hir.func<(!hir.memref<(bank 2)x(bank 3)x2x4xi8> ports [#reg_rd]) -> ()>
  %a_w = hir.memref.extract %a[port 1] :!hir.memref<(bank 2)x(bank 3)x2x4xi8> ports [#reg_wr]
  hir.call @bar(%a_w) at %t : !hir.func<(!hir.memref<(bank 2)x(bank 3)x2x4xi8> ports [#reg_wr]) -> ()>
  hir.return
}

// -----

#reg_wr = {wr_latency=1}
#reg_rd = {rd_latency=0}
hir.func @test5 at %t(){
  %0 = constant 0:index
  %1 = constant 1:index
  %c1_i1 = constant 1:i1
  %c1_i2 = constant 1:i2
  %c1_f32 = constant 1.0:f32

  %a = hir.alloca("REG") : !hir.memref<(bank 2)x(bank 3)x2x4xi8> ports [#reg_rd,#reg_wr]
  %a_r1 = hir.memref.extract %a[port 0] :!hir.memref<(bank 2)x(bank 3)x2x4xi8> ports [#reg_rd]
  hir.call @foo(%a_r1) at %t : !hir.func<(!hir.memref<(bank 2)x(bank 3)x2x4xi8> ports [#reg_rd]) -> ()>
  %a_r2 = hir.memref.extract %a[port 0] :!hir.memref<(bank 2)x(bank 3)x2x4xi8> ports [#reg_rd]
  hir.call @bar(%a_r2) at %t : !hir.func<(!hir.memref<(bank 2)x(bank 3)x2x4xi8> ports [#reg_rd]) -> ()>
  hir.return
}
