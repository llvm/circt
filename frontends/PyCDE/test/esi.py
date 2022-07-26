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
    chan = esi.HostComms.FromHost(types.i32, "loopback_in")
    ports.int_out = chan


@module
class Consumer:
  clk = Input(types.i1)
  int_in = InputChannel(types.i32)

  @generator
  def construct(ports):
    esi.HostComms.ToHost(ports.int_in, "loopback_out")


@module
class Top:
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    p = Producer(clk=ports.clk)
    Consumer(clk=ports.clk, int_in=p.int_out)
    # Use Cosim to implement the standard 'HostComms' service.
    esi.Cosim(esi.HostComms)


s = pycde.System([Top], name="EsiSys")

s.generate()
s.print()

# CHECK-LABEL: msft.module @Top {} (%clk: i1)
# CHECK:         %Producer.const_out = msft.instance @Producer @Producer(%clk)  : (i1) -> !esi.channel<i32>
# CHECK:         msft.instance @Consumer @Consumer(%clk, %Producer.const_out)  : (i1, !esi.channel<i32>) -> ()
# CHECK:         msft.output

# CHECK-LABEL: msft.module @Producer {} (%clk: i1) -> (const_out: !esi.channel<i32>)
# CHECK:         %c42_i32 = hw.constant 42 : i32
# CHECK:         %true = hw.constant true
# CHECK:         %chanOutput, %ready = esi.wrap.vr %c42_i32, %true : i32
# CHECK:         msft.output %chanOutput : !esi.channel<i32>

# CHECK-LABEL: msft.module @Consumer {} (%clk: i1, %int_in: !esi.channel<i32>)
# CHECK:         %true = hw.constant true
# CHECK:         %rawOutput, %valid = esi.unwrap.vr %int_in, %true : i32
# CHECK:         msft.output
