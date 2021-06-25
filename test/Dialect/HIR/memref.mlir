#rd = {rd_delay=1}
#wr = {wr_delay=1}
#reg_wr = {wr_delay=1}
#reg_rw = {rd_delay=0, wr_delay=1}

hir.func @test at %t(
%a :!hir.memref<16x16x(bank 4)x(bank 3)xf32> ports [#rd,#wr],
%b :f32 delay 2,
%c :!hir.bus<i1,flip i1, f32> ports [send]
) -> (i1 delay 1, f32){

  %c0_i1 = constant 0:i1
  %c1_i4 = constant 0:i4
  %0 = constant 0:index
  %1 = constant 0:index
  %x = hir.memref.alloca("BRAM_2P") : !hir.memref<(bank 4)x(bank 4)xi8> ports [#reg_rw,#reg_wr]
  %y = hir.bus.instantiate : !hir.bus<i1, flip i1, i8> 
  %z = hir.bus.instantiate : tensor<4x!hir.bus<i1, flip i1, i8>>

  //It does not make sense to parameterize the bus element to be selected.
  //Thus the corresponding index must be an integer literal. 
  hir.send %c0_i32 to %y[1] : i1 to !hir.bus<i1, flip i1, i8>
  hir.send %c0_i32 to %z[%0,1] : i1 to !hir.bus<i1, flip i1, i8>
  hir.load %x[%c1_i4,%1,%0,%0] port 0 delay 0: !hir.memref<16x16x(bank 4)x(bank 3)xf32>
  hir.store %a[%c1_i4,%1,%0,%0] port 1 delay 1: !hir.memref<16x16x(bank 4)x(bank 3)xf32>
}
