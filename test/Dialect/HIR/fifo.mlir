// RUN: circt-opt %s
#bram_r = {"rd_latency" = 1}
#bram_w = {"wr_latency" = 1}
#reg_r = {"rd_latency" = 0}
#reg_w = {"wr_latency" = 1}

hir.func @fifo_push at %t (
%v :f32,
%wr_idx :!hir.memref<(bank 1)xi6> ports[#reg_r,#reg_w],
%buffer :!hir.memref<64xf32> ports [#bram_w]
)
{
  %0 = arith.constant 0:index
  %c1_i6 = hw.constant 1:i6

  %idx = hir.load %wr_idx[port 0][%0] at %t :!hir.memref<(bank 1)xi6>
  hir.store %v to %buffer[port 0][%idx] at %t : !hir.memref<64xf32> delay 1
  %idx_next = comb.add %idx, %c1_i6:i6
  hir.store %idx_next to %wr_idx[port 1][%0] at %t: !hir.memref<(bank 1)xi6> delay 1
  hir.return
}

hir.func @fifo_pop at %t (
%rd_idx :!hir.memref<(bank 1)xi6> ports[#reg_r,#reg_w],
%buffer :!hir.memref<64xf32> ports [#bram_r]
) -> (%result : f32 delay 1)
{
  %0 = arith.constant 0:index
  %c1_i6 = hw.constant 1:i6

  %idx = hir.load %rd_idx[port 0][%0] at %t :!hir.memref<(bank 1)xi6>
  hir.load %buffer[port 0][%idx] at %t : !hir.memref<64xf32> delay 1
  %idx_next = comb.add %idx, %c1_i6:i6
  hir.store %idx_next to %rd_idx[port 1][%0] at %t: !hir.memref<(bank 1)xi6> delay 1
  hir.return
}


