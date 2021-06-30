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
  %c1_f32 = constant 1.0:f32
  %0 = constant 0:index
  %1 = constant 1:index
  %16 = constant 16:index
  %x = hir.alloca("BRAM_2P") : !hir.memref<(bank 4)x(bank 4)xi8> ports [#reg_rw,#reg_wr]
  %y = hir.bus.instantiate : !hir.bus<i1, flip i1, i8> 
  %z = hir.bus.instantiate : tensor<4x!hir.bus<i1, flip i1, i8>>

  //It does not make sense to parameterize the bus element to be selected.
  //Thus the corresponding index must be an integer literal. 
  %f0 = hir.recv %c[1] at %t : !hir.bus<i1, flip i1, f32> -> i1
  %bus = hir.tensor.extract %z[%0] : tensor<4x!hir.bus<i1, flip i1, i8>> -> !hir.bus<i1, flip i1, i8>
  hir.send %c0_i1 to %bus[0] at %t + %1: i1 to !hir.bus<i1, flip i1, i8>

  //hir.load %x[%0,%1] port 0 delay 0 at %t: !hir.memref<(bank 4)x(bank 4)xi8>
  hir.store %c1_f32 to %a[%c1_i4,%c1_i4,%0,%0] port 1 at %t + %1: !hir.memref<16x16x(bank 4)x(bank 3)xf32>

  hir.for %i : index = %0 to %16 step %1 iter_time(%ti = %t + %1 ){
    hir.yield at %ti + %1
  }

  hir.return
}
