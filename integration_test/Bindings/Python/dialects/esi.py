# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt import esi
from circt.dialects import rtl

from mlir.ir import *
from mlir.dialects import builtin

from os import path
import sys

thisDir = path.dirname(__file__)

with Context() as ctxt, Location.unknown():
  circt.register_dialects(ctxt)
  esisys = esi.System()
  esisys.load_mlir(path.join(thisDir, "esi_load1.mlir"))
  esisys.load_mlir(path.join(thisDir, "esi_load2.mlir"))

  i1 = IntegerType.get_signless(1)
  i32 = IntegerType.get_signless(32)
  i32_chan = esi.ChannelType.get(i32)
  esisys.print()

  with InsertionPoint(esisys.body):
    op = rtl.RTLModuleOp(name='MyWidget',
                         input_ports=[('foo', i32), ('foo_valid', i1)],
                         output_ports=[('foo_ready', i1)],
                         body_builder=lambda module: rtl.OutputOp(
                             [module.entry_block.arguments[1]]))

    snoop = rtl.RTLModuleOp(name='I32Snoop',
                            input_ports=[('foo_in', i32_chan)],
                            output_ports=[('foo_out', i32_chan)],
                            body_builder=lambda module: rtl.OutputOp(
                                [module.entry_block.arguments[0]]))

  esi.buildWrapper(op.operation, ["foo"])
  esisys.print()
  # CHECK-LABEL:  rtl.module @MyWidget_esi(%foo: !esi.channel<i32>) {
  # CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vr %foo, %pearl.foo_ready : i32
  # CHECK-NEXT:     %pearl.foo_ready = rtl.instance "pearl" @MyWidget(%rawOutput, %valid) : (i32, i1) -> i1
  # CHECK-NEXT:     rtl.output
  # CHECK-LABEL:  rtl.module @MyWidget(%foo: i32, %foo_valid: i1) -> (%foo_ready: i1) {
  # CHECK-NEXT:     rtl.output %foo_valid : i1
  # CHECK-LABEL:  rtl.module @I32Snoop(%foo_in: !esi.channel<i32>) -> (%foo_out: !esi.channel<i32>) {
  # CHECK-NEXT:     rtl.output %foo_in : !esi.channel<i32>

  prod = esisys.lookup("IntProducer")
  assert (prod is not None)
  prod.print()
  print()  # Newline.
  # CHECK: rtl.module.extern @IntProducer(%clk: i1) -> (%ints: !esi.channel<i32>)

  acc = esisys.lookup("IntAccumulator")
  assert (acc is not None)
  acc.print()
  print()  # Newline.
  # CHECK: rtl.module.extern @IntAccumulator(%clk: i1, %ints: i32, %ints_valid: i1) -> (%ints_ready: i1, %sum: i32)

  print("\n\n=== Verilog ===")
  # CHECK-LABEL: === Verilog ===
  # CHECK: interface IValidReady_i32;
  # CHECK: // external module IntProducer
  # CHECK: // external module IntAccumulator
  # CHECK: module MyWidget_esi
  # CHECK: module MyWidget
  # CHECK: module I32Snoop
  esisys.print_verilog()

  print("\n\n=== Capnp ===")
  # CHECK-LABEL: === Capnp ===
  # CHECK:       interface CosimDpiServer @0x85e029b5352bcdb5 {
  # CHECK:         list @0 () -> (ifaces :List(EsiDpiInterfaceDesc));
  # CHECK:         open @1 [S, T] (iface :EsiDpiInterfaceDesc) -> (iface :EsiDpiEndpoint(S, T));

  esisys.print_cosim_schema(sys.stdout)
