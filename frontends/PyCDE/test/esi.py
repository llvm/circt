# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s

import pycde
from pycde import (Clock, Input, InputChannel, OutputChannel, module, generator,
                   types)
from pycde import esi


@module
class Producer:
  clk = Input(types.i1)
  int_out = OutputChannel(types.i32)

  @generator
  def construct(ports):
    chan = esi.HostComms.from_host(types.i32, "loopback_in")
    ports.int_out = chan


@module
class Consumer:
  clk = Input(types.i1)
  int_in = InputChannel(types.i32)

  @generator
  def construct(ports):
    esi.HostComms.to_host(ports.int_in, "loopback_out")


@module
class Top:
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    p = Producer(clk=ports.clk)
    Consumer(clk=ports.clk, int_in=p.int_out)
    # Use Cosim to implement the standard 'HostComms' service.
    esi.Cosim(esi.HostComms, ports.clk, ports.rst)


s = pycde.System([Top], name="EsiSys")

s.generate()
s.print()

# CHECK-LABEL: msft.module @Top {} (%clk: i1, %rst: i1) attributes {fileName = "Top.sv"} {
# CHECK:         %Producer.int_out = msft.instance @Producer @Producer(%clk)  : (i1) -> !esi.channel<i32>
# CHECK:         msft.instance @Consumer @Consumer(%clk, %Producer.int_out)  : (i1, !esi.channel<i32>) -> ()
# CHECK:         esi.service.instance @HostComms impl as "cosim"(%clk, %rst) : (i1, i1) -> ()
# CHECK:         msft.output
# CHECK-LABEL: msft.module @Producer {} (%clk: i1) -> (int_out: !esi.channel<i32>) attributes {fileName = "Producer.sv"} {
# CHECK:         [[R0:%.+]] = esi.service.req.to_client <@HostComms::@from_host>(["loopback_in"]) : !esi.channel<i32>
# CHECK:         msft.output [[R0]] : !esi.channel<i32>
# CHECK-LABEL: msft.module @Consumer {} (%clk: i1, %int_in: !esi.channel<i32>) attributes {fileName = "Consumer.sv"} {
# CHECK:         esi.service.req.to_server %int_in -> <@HostComms::@to_host>(["loopback_out"]) : !esi.channel<i32>
# CHECK:         msft.output
# CHECK-LABEL: esi.service.decl @HostComms {
# CHECK:         esi.service.to_server @to_host : !esi.channel<!esi.any>
# CHECK:         esi.service.to_client @from_host : !esi.channel<!esi.any>
