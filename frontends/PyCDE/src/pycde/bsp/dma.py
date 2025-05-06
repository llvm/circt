#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from ..common import Clock, Input, InputChannel, OutputChannel, Reset
from ..constructs import Mux, NamedWire, Wire
from ..module import modparams, generator
from ..types import Bits, Channel, StructType, Type, UInt
from ..support import clog2
from .. import esi


@modparams
def OneItemBuffersToHost(client_type: Type):
  """Create a simple, non-performant DMA-based channel communication. Protocol:

  1) Host sends address of buffer address via MMIO write.
  2) Device writes data on channel with a byte '1' to said buffer address.
  3) Host polls the last byte in buffer for '1'.
  4) Data is copied out of buffer, last byte is set to '0', goto 1.

  Future improvement: support more than one buffer at once."""

  class OneItemBuffersToHost(esi.EngineModule):

    @property
    def TypeName(self):
      return "OneItemBuffersToHost"

    clk = Clock()
    rst = Reset()
    # The channel whose messages we are sending to the host.
    input_channel = InputChannel(client_type)

    # Since we cannot produce service requests (shortcoming of the ESI
    # compiler), the module (usually a service implementation) must issue the
    # service requests for us then connect the input ports.
    mmio = Input(esi.MMIO.read_write.type)
    xfer_data_type = StructType([("valid", Bits(8)),
                                 ("client_data", client_type)])
    hostmem_write = Input(esi.HostMem.write_req_bundle_type(xfer_data_type))

    @generator
    def build(ports):
      clk = ports.clk
      rst = ports.rst

      # Set up the MMIO interface to receive the buffer locations.
      mmio_resp_chan = Wire(Channel(Bits(64)))
      mmio_rw = ports.mmio
      mmio_cmd_chan_raw = mmio_rw.unpack(data=mmio_resp_chan)['cmd']
      mmio_cmd_chan, mmio_cmd_fork_resp = mmio_cmd_chan_raw.fork(clk, rst)

      # Create a response channel which always responds with 0.
      mmio_resp_data = NamedWire(Bits(64)(0), "mmio_resp_data")
      mmio_resp_chan.assign(
          mmio_cmd_fork_resp.transform(lambda _: mmio_resp_data))

      # Create a mailbox for each register. Overkill for one register, but may
      # be useful later on.
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

      # Join the buffer location channel with the input channel and send a write
      # message to hostmem.
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
      ports.hostmem_write.unpack(req=hostwr)

  return OneItemBuffersToHost


def OneItemBuffersFromHost(client_type: Type):
  """Create a simple, non-performant DMA-base from host communication channel.

  1) Host sends address of buffer address via MMIO write to register 0x08.
  2) Host sends address of completion address via MMIO write to register 0x10.
  3) Device reads data from said buffer and sends down the channel. Only
     initiates the transfer if the output channel is ready so as to not cause
     contention / deadlock in the HostMem service. This feature should probably
     be implemented by HostMem.
  4) Device writes '1' to the first byte of the completion buffer to signal that
     the transfer is done.
  """

  class OneItemBuffersFromHost(esi.EngineModule):

    @property
    def TypeName(self):
      return "OneItemBuffersFromHost"

    clk = Clock()
    rst = Reset()

    # The channel whose messages we are sending to the host.
    output_channel = OutputChannel(client_type)

    # Since we cannot produce service requests (shortcoming of the ESI
    # compiler), the module (usually a service implementation) must issue the
    # service requests for us then connect the input ports.
    mmio = Input(esi.MMIO.read_write.type)
    xfer_data_type = Bits(8)
    hostmem_write = Input(esi.HostMem.write_req_bundle_type(xfer_data_type))
    hostmem_read = Input(esi.HostMem.read_bundle_type(client_type))

    @generator
    def build(ports):
      clk = ports.clk
      rst = ports.rst

      # Set up the MMIO interface to receive the buffer locations.
      mmio_resp_chan = Wire(Channel(Bits(64)))
      mmio_rw = ports.mmio
      mmio_cmd_chan_raw = mmio_rw.unpack(data=mmio_resp_chan)['cmd']
      mmio_cmd_chan, mmio_cmd_fork_resp = mmio_cmd_chan_raw.fork(clk, rst)

      # Create a response channel which always responds with 0.
      mmio_resp_data = NamedWire(Bits(64)(0), "mmio_resp_data")
      mmio_resp_chan.assign(
          mmio_cmd_fork_resp.transform(lambda _: mmio_resp_data))

      # Route the MMIO command
      _, _, mmio_cmd = mmio_cmd_chan.snoop()
      mailbox_names = ["null", "buffer_loc", "completion_addr"]
      num_sinks = len(mailbox_names)
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
      demuxed = esi.ChannelDemux(mmio_data_only_chan, cmd_sink_sel, num_sinks)
      mailbox_mod = esi.Mailbox(Bits(64))
      mailboxes = [
          mailbox_mod(clk=clk,
                      rst=rst,
                      input=c,
                      instance_name="mailbox_" + name)
          for name, c in zip(mailbox_names, demuxed)
      ]
      [_, buffer_loc, completion_loc] = mailboxes
      buffer_loc_for_read = buffer_loc.output

      output_chan = Wire(Channel(client_type))
      ports.output_channel = output_chan

      # Only issue a read request if the output channel is ready. (And the
      # buffer location has been written.)
      read_req = buffer_loc_for_read.wait_for_ready(output_chan).transform(
          lambda m: esi.HostMem.ReadReqType({
              "address": m.as_uint(),
              "tag": 0,
          }))
      # Issue the read data while getting the trigger signal for the completion
      # write.
      read_resp = ports.hostmem_read.unpack(req=read_req)['resp']
      read_resp_for_out, read_resp_write_trigger = read_resp.fork(clk, rst)
      output_chan.assign(read_resp_for_out.transform(lambda d: d.data))

      # Issue the completion write request once the data has been read.
      write_ch_type = esi.HostMem.write_req_channel_type(
          OneItemBuffersFromHost.xfer_data_type)
      completion_write = Channel.join(completion_loc.output,
                                      read_resp_write_trigger)
      write_done_chan = completion_write.transform(lambda m: write_ch_type({
          "address": m.a.as_uint(),
          "tag": 0,
          "data": Bits(8)(1)
      }))

      # Unpack the write port, but ignore the write confirmation.
      ports.hostmem_write.unpack(req=write_done_chan)

  return OneItemBuffersFromHost
