# REQUIRES: bindings_python
# REQUIRES: capnp
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt import esi
from circt.esi import types
from circt.dialects import hw

from mlir.ir import *
from mlir import passmanager

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


with Context() as ctx:
  circt.register_dialects(ctx)
  any_type = esi.AnyType.get()
  print(any_type)
  print()
  # CHECK: !esi.any

esisys = TestESISys()

prod = esisys.lookup("IntProducer")
assert (prod is not None)
prod.print()
print()  # Newline.
# CHECK: hw.module.extern @IntProducer(%clk: i1) -> (ints: !esi.channel<i32>)

acc = esisys.lookup("IntAccumulator")
assert (acc is not None)
acc.print()
print()  # Newline.
# CHECK: hw.module.extern @IntAccumulator(%clk: i1, %ints: i32, %ints_valid: i1) -> (ints_ready: i1, sum: i32)

esisys.print()
# CHECK-LABEL:  hw.module @MyWidget_esi(%foo: !esi.channel<i32>) {
# CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vr %foo, %pearl.foo_ready : i32
# CHECK-NEXT:     %pearl.foo_ready = hw.instance "pearl" @MyWidget(foo: %rawOutput: i32, foo_valid: %valid: i1) -> (foo_ready: i1)
# CHECK-NEXT:     hw.output
# CHECK-LABEL:  hw.module @MyWidget(%foo: i32, %foo_valid: i1) -> (foo_ready: i1) {
# CHECK-NEXT:     hw.output %foo_valid : i1
# CHECK-LABEL:  hw.module @I32Snoop(%foo_in: !esi.channel<i32>) -> (foo_out: !esi.channel<i32>) {
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


# CHECK-LABEL: === testGen called with op:
# CHECK:       %0:2 = esi.service.impl_req @HostComms impl as "test"(%clk, %m1.loopback_fromhw) : (i1, !esi.channel<i8>) -> (i8, !esi.channel<i8>) {
# CHECK:       ^bb0(%arg0: !esi.channel<i8>):
# CHECK:         %2 = esi.service.req.to_client <@HostComms::@Recv>(["m1", "loopback_tohw"]) : !esi.channel<i8>
# CHECK:         esi.service.req.to_server %arg0 -> <@HostComms::@Send>(["m1", "loopback_fromhw"]) : !esi.channel<i8>
def testGen(reqOp: esi.ServiceImplementReqOp) -> bool:
  print("=== testGen called with op:")
  reqOp.print()
  print()
  return True


esi.registerServiceGenerator("test", testGen)

with Context() as ctx:
  circt.register_dialects(ctx)
  mod = Module.parse("""
esi.service.decl @HostComms {
  esi.service.to_server @Send : !esi.channel<!esi.any>
  esi.service.to_client @Recv : !esi.channel<i8>
}

msft.module @MsTop {} (%clk: i1) -> (chksum: i8) {
  %c = esi.service.instance @HostComms impl as  "test" (%clk) : (i1) -> (i8)
  msft.instance @m1 @MsLoopback (%clk) : (i1) -> ()
  msft.output %c : i8
}

msft.module @MsLoopback {} (%clk: i1) -> () {
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i8>
  esi.service.req.to_server %dataIn -> <@HostComms::@Send> (["loopback_fromhw"]) : !esi.channel<i8>
  msft.output
}
""")
  pm = passmanager.PassManager.parse("esi-connect-services")
  pm.run(mod)
