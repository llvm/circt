# REQUIRES: esi-runtime, esi-cosim, rtl-sim
# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1
# RUN: esi-cosim-runner.py --tmpdir %t --exec %S/esi_test.sw.py `ls %t/hw/*.sv | grep -v driver.sv`

import pycde
from pycde import (AppID, Clock, Input, Module, generator, types)
from pycde.bsp import cosim
from pycde.constructs import Wire
from pycde.esi import ServiceDecl
from pycde.types import Bits, Bundle, BundledChannel, ChannelDirection

import sys

TestBundle = Bundle([
    BundledChannel("resp", ChannelDirection.TO, Bits(16)),
    BundledChannel("req", ChannelDirection.FROM, Bits(24))
])


@ServiceDecl
class HostComms:
  req_resp = TestBundle


class LoopbackInOutAdd7(Module):

  @generator
  def construct(ports):
    loopback = Wire(types.channel(types.i16))
    call_bundle, froms = TestBundle.pack(resp=loopback)
    from_host = froms['req']
    HostComms.req_resp(call_bundle, AppID("loopback_inout", 0))

    ready = Wire(types.i1)
    data, valid = from_host.unwrap(ready)
    plus7 = data.as_uint(15) + types.ui8(7)
    data_chan, data_ready = loopback.type.wrap(plus7.as_bits(), valid)
    ready.assign(data_ready)
    loopback.assign(data_chan)


class Top(Module):
  clk = Clock()
  rst = Input(types.i1)

  @generator
  def construct(ports):
    LoopbackInOutAdd7()


if __name__ == "__main__":
  s = pycde.System(cosim.CosimBSP(Top),
                   name="ESILoopback",
                   output_directory=sys.argv[1])
  s.compile()
  s.package()
