# REQUIRES: bindings_python
# REQUIRES: capnp
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt import esi
from circt.esi import types
from circt.dialects import hw

from mlir.ir import *
from mlir.dialects import builtin

from os import path
import sys

thisDir = path.dirname(__file__)


class TestESISys(esi.System):

  def declare_externs(self):
    """Declare all of the external modules"""
    self.load_mlir(path.join(thisDir, "esi_load1.mlir"))
    self.load_mlir(path.join(thisDir, "esi_load2.mlir"))

    op = hw.HWModuleOp(
        name='MyWidget',
        input_ports=[('foo', types.i32), ('foo_valid', types.i1)],
        output_ports=[('foo_ready', types.i1)],
        body_builder=lambda module: hw.OutputOp([module.foo_valid]))

    i32chan = types.chan(types.i32)
    hw.HWModuleOp(name='I32Snoop',
                  input_ports=[('foo_in', i32chan)],
                  output_ports=[('foo_out', i32chan)],
                  body_builder=lambda module: hw.OutputOp([module.foo_in]))

    esi.buildWrapper(op.operation, ["foo"])

  def build(self, top):
    return


esisys = TestESISys()

prod = esisys.lookup("IntProducer")
assert (prod is not None)
prod.print()
print()  # Newline.
# CHECK: hw.module.extern @IntProducer(%clk: i1) -> (%ints: !esi.channel<i32>)

acc = esisys.lookup("IntAccumulator")
assert (acc is not None)
acc.print()
print()  # Newline.
# CHECK: hw.module.extern @IntAccumulator(%clk: i1, %ints: i32, %ints_valid: i1) -> (%ints_ready: i1, %sum: i32)

esisys.print()
# CHECK-LABEL:  hw.module @MyWidget_esi(%foo: !esi.channel<i32>) {
# CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vr %foo, %pearl.foo_ready : i32
# CHECK-NEXT:     %pearl.foo_ready = hw.instance "pearl" @MyWidget(%rawOutput, %valid) : (i32, i1) -> i1
# CHECK-NEXT:     hw.output
# CHECK-LABEL:  hw.module @MyWidget(%foo: i32, %foo_valid: i1) -> (%foo_ready: i1) {
# CHECK-NEXT:     hw.output %foo_valid : i1
# CHECK-LABEL:  hw.module @I32Snoop(%foo_in: !esi.channel<i32>) -> (%foo_out: !esi.channel<i32>) {
# CHECK-NEXT:     hw.output %foo_in : !esi.channel<i32>

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
