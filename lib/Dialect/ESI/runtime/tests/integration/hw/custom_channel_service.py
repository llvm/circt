#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

from pycde import AppID, Clock, Module, Reset, System, esi, generator
from pycde.types import UInt

from esiaccel.bsp.cosim import CosimBSP
from esiaccel.bsp.dma import OneItemBuffersFromHost, OneItemBuffersToHost


class Loopbacks(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def build(ports):
    for i in range(2):
      channel = esi.ChannelService.from_host(AppID(f"loopback_in_{i}"),
                                             UInt(256))
      esi.ChannelService.to_host(AppID(f"loopback_out_{i}"), channel)


class CustomChannelService(esi.ServiceImplementation):
  """Serve this test's channels without delegating to ChannelEngineService."""

  clk = Clock()
  rst = Reset()

  @generator
  def build(ports, bundles):
    assert len(bundles.to_client_reqs) == 4
    for bundle in bundles.to_client_reqs:
      assert bundle.port in ("from_host", "to_host")
      from_host = bundle.port == "from_host"
      engine_module = (OneItemBuffersFromHost(UInt(256))
                       if from_host else OneItemBuffersToHost(UInt(256)))
      name = "custom_" + bundle.client_name_str
      mmio_appid = AppID(name + ".mmio")
      inputs = {
          "clk":
              ports.clk,
          "rst":
              ports.rst,
          "mmio":
              esi.MMIO.read_write(mmio_appid),
          "hostmem_write":
              esi.HostMem.write_from_bundle(AppID(name + ".write"),
                                            engine_module.hostmem_write.type),
      }
      if from_host:
        inputs["hostmem_read"] = esi.HostMem.read_from_bundle(
            AppID(name + ".read"), engine_module.hostmem_read.type)
      else:
        client_bundle, channels = bundle.type.pack()
        bundle.assign(client_bundle)
        inputs["input_channel"] = channels["data"]

      engine = engine_module(appid=AppID(name), **inputs)
      if from_host:
        client_bundle, _ = bundle.type.pack(data=engine.output_channel)
        bundle.assign(client_bundle)

      record = bundles.emit_engine(engine,
                                   details={
                                       "engine_inst": engine.appid,
                                       "mmio": mmio_appid,
                                   })
      record.add_record(bundle, {"data": {}})


if __name__ == "__main__":
  system = System(CosimBSP(Loopbacks, channel_service=CustomChannelService),
                  name="CustomChannelServiceTest",
                  output_directory=sys.argv[1])
  system.compile()
  system.package()
