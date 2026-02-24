// RUN: circt-bmc %s --module=top -b 15 --run --shared-libs=/usr/lib/x86_64-linux-gnu/libz3.so | FileCheck %s --check-prefix=CHECK-ASYNC
// RUN: circt-bmc %s --module=top -b 15 --run --shared-libs=/usr/lib/x86_64-linux-gnu/libz3.so --sync-clocks | FileCheck %s --check-prefix=CHECK-SYNC

hw.module @top(in %clk_tx: !seq.clock, in %clk_rx: !seq.clock, in %en: i1, out res: i1) {
  %c1_i1 = hw.constant 1 : i1
  %c0_i1 = hw.constant 0 : i1
  
  %g0_toggle = comb.xor %g_tx_0, %c1_i1 : i1
  %flip_g0 = comb.icmp eq %g_tx_0, %g_tx_1 : i1
  %g0_target = comb.mux %flip_g0, %g0_toggle, %g_tx_0 : i1
  %g0_val = comb.mux %en, %g0_target, %g_tx_0 : i1
  
  %g1_toggle = comb.xor %g_tx_1, %c1_i1 : i1
  %flip_g1 = comb.icmp ne %g_tx_0, %g_tx_1 : i1
  %g1_target = comb.mux %flip_g1, %g1_toggle, %g_tx_1 : i1
  %g1_val = comb.mux %en, %g1_target, %g_tx_1 : i1

  %init0 = seq.initial () {
    %c0 = hw.constant 0 : i1
    seq.yield %c0 : i1 
  } : () -> !seq.immutable<i1>
  %g_tx_0 = seq.compreg %g0_val, %clk_tx initial %init0 : i1
  %g_tx_1 = seq.compreg %g1_val, %clk_tx initial %init0 : i1

  %s0 = hw.instance "u0" @sync_2ff(clk_dst: %clk_rx: !seq.clock, d_in: %g_tx_0: i1) -> (d_out: i1)
  %s1 = hw.instance "u1" @sync_2ff(clk_dst: %clk_rx: !seq.clock, d_in: %g_tx_1: i1) -> (d_out: i1)
  %g_rx = comb.concat %s1, %s0 : i1, i1

  %init2 = seq.initial () { 
    %c0_i2 = hw.constant 0 : i2
    seq.yield %c0_i2 : i2 
  } : () -> !seq.immutable<i2>
  %g_rx_prev = seq.compreg %g_rx, %clk_rx initial %init2 : i2

  %diff = comb.xor %g_rx, %g_rx_prev : i2
  %c3_i2 = hw.constant 3 : i2
  %is_bad_jump = comb.icmp eq %diff, %c3_i2 : i2
  
  %error_next = comb.or %error_reg, %is_bad_jump : i1
  %error_reg = seq.compreg %error_next, %clk_rx initial %init0 : i1

  %ok = comb.xor %error_reg, %c1_i1 : i1
  verif.assert %ok : i1
  hw.output %error_reg : i1
}

hw.module private @sync_2ff(in %clk_dst: !seq.clock, in %d_in: i1, out d_out: i1) {
  %init = seq.initial () { 
    %c0 = hw.constant 0 : i1
    seq.yield %c0 : i1 
  } : () -> !seq.immutable<i1>
  %stage1 = seq.compreg %d_in, %clk_dst initial %init : i1
  %stage2 = seq.compreg %stage1, %clk_dst initial %init : i1
  hw.output %stage2 : i1
}

// CHECK-ASYNC: Assertion can be violated!
// CHECK-SYNC: Bound reached with no violations!
