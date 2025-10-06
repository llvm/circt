// REQUIRES: esi-cosim, esi-runtime, rtl-sim

// Generate SV files
// RUN: rm -rf %t6 && mkdir %t6 && cd %t6
// RUN: mkdir hw && cd hw
// RUN: circt-opt %s --esi-connect-services --esi-appid-hier=top=top --esi-build-manifest=top=top --esi-clean-metadata --lower-esi-to-physical --lower-esi-bundles --lower-esi-ports --lower-esi-to-hw=platform=cosim --lower-seq-to-sv --lower-hwarith-to-hw --canonicalize --export-split-verilog

// Test cosimulation
// RUN: cp %esi_prims %t6/hw
// RUN: esi-cosim.py --source %t6/hw --top top -- esitester cosim env wait | FileCheck %s

hw.module @top(in %clk : !seq.clock, in %rst : i1) {
  hw.instance "PrintfExample" sym @PrintfExample @PrintfExample(clk: %clk: !seq.clock, rst: %rst: i1) -> ()
  esi.service.instance #esi.appid<"cosim_default"> impl as "cosim" (%clk, %rst) : (!seq.clock, i1) -> ()
}

// CHECK:  PrintfExample: 7
hw.module @PrintfExample(in %clk : !seq.clock, in %rst : i1) {
  %0 = hwarith.constant 7 : ui32
  %true = hw.constant true
  %false = hw.constant false
  %1 = seq.compreg.ce %true, %clk, %5 reset %rst, %false : i1
  %true_0 = hw.constant true
  %2 = comb.xor bin %1, %true_0 : i1
  %true_1 = hw.constant true
  %3 = comb.xor bin %rst, %true_1 {sv.namehint = "inv_rst"} : i1
  %4 = comb.and bin %2, %3 : i1
  %chanOutput, %ready = esi.wrap.vr %0, %4 : ui32
  %5 = comb.and bin %ready, %4 {sv.namehint = "sent_signal"} : i1
  %6 = esi.service.req <@_CallService::@call>(#esi.appid<"PrintfExample">) : !esi.bundle<[!esi.channel<ui32> from "arg", !esi.channel<i0> to "result"]>
  %result = esi.bundle.unpack %chanOutput from %6 : !esi.bundle<[!esi.channel<ui32> from "arg", !esi.channel<i0> to "result"]>
}
esi.service.std.call @_CallService
