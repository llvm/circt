# ===- mmio.py - MMIO register-file component -----------------------------===//
#
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//
#
#  A timing-friendly MMIO register file built on `esi.MMIO.read_write`. It owns
#  the MMIO request bundle, presents automatically-numbered read-only,
#  read-write, and write-only registers to the host, and exposes a FF-bounded
#  command surface so the BSP MMIO mux only ever sees registered outputs.
#
# ===----------------------------------------------------------------------===//

import pycde.esi as esi
from pycde import AppID, Clock, Input, Module, Output, Reset, generator
from pycde.constructs import ControlReg, Mux, Wire
from pycde.module import modparams
from pycde.types import Array, Bits, Channel, StructType, UInt
from pycde.support import clog2

# A presented MMIO write command. Unlike ``esi.MMIOReadWriteCmdType`` there is
# no redundant ``write`` field -- every command on this surface is a write.
MMIOWriteCmdType = StructType([
    ("offset", UInt(32)),
    ("data", Bits(64)),
])


@modparams
def MmioRegistry(num_ro: int, num_rw: int, num_wo: int):
  """MMIO register file with automatically-numbered registers (RO < RW < WO)
  addressed at an 8-byte stride (``index = offset >> 3``).

  Numbering::

      [0, num_ro)                              -> read-only  (RO)
      [num_ro, num_ro + num_rw)                -> read-write (RW)
      [num_ro + num_rw, num_ro+num_rw+num_wo)  -> write-only (WO)

  The RO and RW registers ("read registers", ``num_ro + num_rw`` of them) are
  held internally. The client updates any of them via the per-read-register
  ``read_reg_ce`` / ``read_reg_data`` arrays and observes their current value
  on ``read_reg_value`` (all length ``num_ro + num_rw``, indexed RO-first then
  RW). A host write to an RW register additionally stores the written data into
  that register directly; a host write takes priority over a coincident client
  update of the same register.

  Reads -- and the read-back value returned for a write -- respond with the
  selected register's value, or all-ones (-1) if the offset selects a WO
  register or is out of bounds.

  Writes to RW/WO registers are registered and presented to the client on
  ``write_cmd_r`` / ``write_cmd_xact_r`` (writes to RO offsets or out-of-bounds
  offsets are dropped). ``mmio_write_we`` derives per-offset write strobes from
  these.

  The MMIO bundle is created inside this submodule's ``@generator``, so the
  request's AppID is anchored at this instance's hierarchical position.

  Ports:
      read_reg_ce    : Input  Array(Bits(1),  num_ro + num_rw)
      read_reg_data  : Input  Array(Bits(64), num_ro + num_rw)
      read_reg_value : Output Array(Bits(64), num_ro + num_rw)
      write_cmd_r    : Output MMIOWriteCmdType (last presented write)
      write_cmd_xact_r : Output Bits(1) (1-cycle pulse on a presented write)
  """
  num_read = num_ro + num_rw
  num_total = num_ro + num_rw + num_wo
  assert num_read >= 1, "MmioRegistry needs at least one read register."

  class MmioRegistryImpl(Module):
    clk = Clock()
    rst = Reset()

    read_reg_ce = Input(Array(Bits(1), num_read))
    read_reg_data = Input(Array(Bits(64), num_read))
    read_reg_value = Output(Array(Bits(64), num_read))

    write_cmd_r = Output(MMIOWriteCmdType)
    write_cmd_xact_r = Output(Bits(1))

    @generator
    def construct(ports):
      clk = ports.clk
      rst = ports.rst

      mmio_bundle = esi.MMIO.read_write(appid=AppID("cmd"))

      cmd_chan_wire = Wire(Channel(esi.MMIOReadWriteCmdType))
      cmd_ready_wire = Wire(Bits(1))
      cmd, cmd_valid = cmd_chan_wire.unwrap(cmd_ready_wire)

      resp_pending_wire = Wire(Bits(1))
      resp_data_r = Wire(Bits(64))
      resp_chan, resp_ready = Channel(Bits(64)).wrap(resp_data_r,
                                                     resp_pending_wire)
      resp_xact = resp_pending_wire & resp_ready
      cmd_xact = cmd_valid & ~resp_pending_wire
      cmd_ready_wire.assign(~resp_pending_wire)
      resp_pending_wire.assign(
          ControlReg(
              clk=clk,
              rst=rst,
              asserts=[cmd_xact],
              resets=[resp_xact],
              name="resp_pending",
          ))

      # Register index = MMIO byte offset >> 3 (8-byte stride).
      idx = cmd.offset.as_bits()[3:].as_uint()
      idx_w = idx.type.width

      def is_index(i):
        return idx == UInt(idx_w)(i)

      # Read registers (RO first, then RW). RW registers also auto-store a
      # host write (host takes priority over a coincident client update).
      read_values = []
      for i in range(num_read):
        if i >= num_ro:
          host_we = cmd_xact & cmd.write & is_index(i)
          reg_ce = host_we | ports.read_reg_ce[i]
          reg_data = Mux(host_we, ports.read_reg_data[i], cmd.data)
        else:
          reg_ce = ports.read_reg_ce[i]
          reg_data = ports.read_reg_data[i]
        read_values.append(
            reg_data.reg(
                clk=clk,
                rst=rst,
                rst_value=Bits(64)(0),
                ce=reg_ce,
                name=f"read_reg_{i}",
            ))
      read_values_arr = Array(Bits(64), num_read)(read_values)
      ports.read_reg_value = read_values_arr

      # Read / write-response value: the selected read register, or -1 for a
      # WO register or an out-of-bounds offset.
      sel_read_value = read_values_arr[idx.as_bits(clog2(num_read))]
      resp_sel = Mux(
          idx < UInt(idx_w)(num_read),
          Bits(64)(2**64 - 1),
          sel_read_value,
      )
      resp_data_r.assign(
          resp_sel.reg(
              clk=clk,
              rst=rst,
              rst_value=Bits(64)(0),
              ce=cmd_xact,
              name="resp_data_r",
          ))

      # Present RW/WO writes to the client; drop RO / out-of-bounds writes.
      presented = (cmd_xact & cmd.write & (idx >= UInt(idx_w)(num_ro)) &
                   (idx < UInt(idx_w)(num_total)))
      write_cmd = MMIOWriteCmdType({"offset": cmd.offset, "data": cmd.data})
      ports.write_cmd_r = write_cmd.reg(clk=clk,
                                        rst=rst,
                                        ce=presented,
                                        name="write_cmd_r")
      ports.write_cmd_xact_r = presented.reg(clk=clk,
                                             rst=rst,
                                             rst_value=Bits(1)(0),
                                             name="write_cmd_xact_r")

      mmio_rw_cmd_chan = mmio_bundle.unpack(data=resp_chan)["cmd"]
      cmd_chan_wire.assign(mmio_rw_cmd_chan)

  return MmioRegistryImpl


def mmio_write_we(mmio, offset: int):
  """Registered 1-cycle write-enable strobe for a host write to ``offset``.

  Only pulses for *presented* writes (RW/WO registers); asserts the cycle
  after the write is accepted. Use as a register ``ce`` or a start pulse.
  ``offset`` is the register's byte offset (``index << 3``).
  """
  return mmio.write_cmd_xact_r & (mmio.write_cmd_r.offset == UInt(32)(offset))
