#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from ..common import AppID, Clock, Input, InputChannel, Reset
from ..constructs import Mux, NamedWire, Reg, Wire
from ..module import modparams, generator
from ..types import Bits, Channel, StructType, Type, UInt
from ..support import clog2
from .. import esi


@modparams
def OneItemBuffersToHost(client_type: Type):

  class OneItemBuffersToHost(esi.EngineModule):

    @property
    def TypeName(self):
      return "OneItemBuffersToHost"

    clk = Clock()
    rst = Reset()
    input_channel = InputChannel(client_type)

    mmio = Input(esi.MMIO.read_write.type)

    xfer_data_type = StructType([("valid", Bits(8)),
                                 ("client_data", client_type)])
    hostmem = Input(esi.HostMem.write_req_bundle_type(xfer_data_type))

    @generator
    def build(ports):
      clk = ports.clk
      rst = ports.rst

      mmio_resp_chan = Wire(Channel(Bits(64)))

      mmio_rw = ports.mmio
      mmio_cmd_chan_raw = mmio_rw.unpack(data=mmio_resp_chan)['cmd']
      mmio_cmd_chan, mmio_cmd_fork_resp = mmio_cmd_chan_raw.fork(clk, rst)

      mmio_resp_data = Wire(Bits(64), "mmio_resp_data")
      # Always respond 0.
      mmio_resp_data.assign(0)
      mmio_resp_chan.assign(
          mmio_cmd_fork_resp.transform(lambda _: mmio_resp_data))

      _, _, mmio_cmd = mmio_cmd_chan.snoop()
      num_sinks = 2
      mmio_offset_words = NamedWire((mmio_cmd.offset.as_bits()[3:]).as_uint(),
                                    "mmio_offset_words")
      addr_above = mmio_offset_words >= UInt(32)(num_sinks)
      addr_is_zero = mmio_offset_words == UInt(32)(0)
      force_to_null = NamedWire(addr_above | ~addr_is_zero | mmio_cmd.write,
                                "force_to_null")
      cmd_sink_sel = Mux(force_to_null,
                         Bits(clog2(num_sinks))(0),
                         mmio_offset_words.as_bits()[:clog2(num_sinks)])
      mmio_data_only_chan = mmio_cmd_chan.transform(lambda m: m.data)
      mailbox_names = ["null", "buffer_loc"]
      demuxed = esi.ChannelDemux(mmio_data_only_chan, cmd_sink_sel, num_sinks)
      mailbox_mod = esi.Mailbox(Bits(64))
      mailboxes = [
          mailbox_mod(clk=clk,
                      rst=rst,
                      input=c,
                      instance_name="mailbox_" + name)
          for name, c in zip(mailbox_names, demuxed)
      ]
      [_, buffer_loc] = mailboxes

      next_buffer_loc_chan = buffer_loc.output

      hostwr_type = esi.HostMem.write_req_channel_type(
          OneItemBuffersToHost.xfer_data_type)
      hostwr_joined = Channel.join(next_buffer_loc_chan, ports.input_channel)
      hostwr = hostwr_joined.transform(lambda joined: hostwr_type({
          "address": joined.a.as_uint(),
          "tag": 0,
          "data": {
              "valid": 1,
              "client_data": joined.b
          },
      }))
      ports.hostmem.unpack(req=hostwr)

  return OneItemBuffersToHost
