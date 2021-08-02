#rd = {rd_latency=1}
#wr = {wr_latency=1}
#reg_wr = {wr_latency=1}
#reg_rw = {rd_latency=0, wr_latency=1}

hir.func @test at %t(
%a :!hir.memref<16x16x(bank 2)x(bank 2)xf32> ports [#rd,#wr],
%b :f32 delay 2,
%c :tensor<16x16x!hir.bus<i8>> ports [send]
) -> (i1 delay 1, f32){

  %c0_i1 = constant 0:i1
  %c1_i4 = constant 0:i4
  %c1_f32 = constant 1.0:f32
  %c2_f32 = constant 2.0:f32
  %0 = constant 0:index
  %1 = constant 1:index
  %16 = constant 16:index
  //%x = hir.alloca("BRAM_2P") : !hir.memref<(bank 4)x(bank 4)xi8> ports [#reg_rw,#reg_wr]

  hir.load %a[port 0][%c1_i4,%c1_i4,%0,%1]  at %t: !hir.memref<16x16x(bank 2)x(bank 2)xf32> delay 1
  hir.store %c1_f32 to %a[port 1][%c1_i4,%c1_i4,%0,%0] at %t + 1: !hir.memref<16x16x(bank 2)x(bank 2)xf32> delay 1
  hir.store %c2_f32 to %a[port 1][%c1_i4,%c1_i4,%0,%0] at %t + 1: !hir.memref<16x16x(bank 2)x(bank 2)xf32> delay 1

  hir.return
}
