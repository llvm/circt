#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pycde import Clock, Input, InputChannel, Output, OutputChannel, System
from pycde import module, generator, esi, types
from ..value import And, Or, BitVectorValue
from ..constructs import ControlReg, Reg, Wire
from ..module import _GeneratorPortAccess
from .fifo import SimpleXilinxFifo
from ..dialects import hw

from dataclasses import asdict, dataclass
import glob
from io import FileIO
import json
import math
import pathlib
import shutil
from typing import List, Tuple

__dir__ = pathlib.Path(__file__).parent

# Parameters for AXI4-Lite interface
axil_addr_width = 24
axil_data_width = 32
axil_data_width_bytes = int(axil_data_width / 8)
DONE_BIT = 0x2


# Signals from master
def axil_in_type(addr_width, data_width):
  return types.struct({
      "awvalid": types.i1,
      "awaddr": types.int(addr_width),
      "wvalid": types.i1,
      "wdata": types.int(data_width),
      "wstrb": types.int(data_width // 8),
      "arvalid": types.i1,
      "araddr": types.int(addr_width),
      "rready": types.i1,
      "bready": types.i1
  })


# Signals to master
def axil_out_type(data_width):
  return types.struct({
      "awready": types.i1,
      "wready": types.i1,
      "arready": types.i1,
      "rvalid": types.i1,
      "rdata": types.int(data_width),
      "rresp": types.i2,
      "bvalid": types.i1,
      "bresp": types.i2
  })


@dataclass
class RegSpec:
  size: int
  offset: int
  data_offset: int
  name: str
  client_path: list
  type: str


def generate_mmio_map(channels, base_offset=0x0) -> Tuple[List, List]:
  # curr_offset is in bytes.
  curr_offset = base_offset

  def get_reg(req, write):
    nonlocal curr_offset
    width = req.type.inner_type.bitwidth

    reg = RegSpec(size=width,
                  offset=curr_offset,
                  data_offset=(curr_offset +
                               axil_data_width_bytes) if write else curr_offset,
                  name="_".join(req.client_name),
                  client_path=req.client_name,
                  type="direct_mmio")

    # A write reg requires extra space for control.
    if write:
      width += axil_data_width
    curr_offset += int(
        (axil_data_width / 8) * math.ceil(width / axil_data_width))
    return reg

  from_host_regs = [get_reg(req, False) for req in channels.to_client_reqs]
  to_host_regs = [get_reg(req, True) for req in channels.to_server_reqs]
  return (from_host_regs, to_host_regs)


def output_tcl(from_host_regs, to_host_regs, os: FileIO):
  from jinja2 import Environment, FileSystemLoader, StrictUndefined

  env = Environment(loader=FileSystemLoader(str(__dir__)),
                    undefined=StrictUndefined)
  template = env.get_template("xrt_package.tcl.j2")
  os.write(
      template.render(system_name=System.current().name,
                      from_host_regs=from_host_regs,
                      to_host_regs=to_host_regs))


@module
def RegsToChannel(type, offset):

  class RegsToChannel:
    """Convert a series of writes to a message."""

    clk = Clock()
    rst = Input(types.i1)

    # `address` and `data` are both valid on `valid`.
    valid = Input(types.i1)
    address = Input(types.int(axil_addr_width))
    data = Input(types.int(axil_data_width))

    fifo_almost_full = Output(types.i1)

    msgs = OutputChannel(type)

    _offset = offset
    _type = type

    @generator
    def build(ports):
      clk = ports.clk
      rst = ports.rst
      msg_type = RegsToChannel._type

      fifo_wr = Wire(types.i1)
      reg_valids = []
      data_regs = []
      for reg_offset in range(math.ceil(msg_type.bitwidth / axil_data_width)):
        reg_num = RegsToChannel._offset + (reg_offset * axil_data_width_bytes)
        addr_match = ports.address == types.int(axil_addr_width)(reg_num)
        write_reg = addr_match & ports.valid
        reg_valid = ControlReg(clk, rst, [write_reg], [fifo_wr])
        reg_valids.append(reg_valid)
        data_reg = ports.data.reg(clk=clk,
                                  rst=rst,
                                  ce=write_reg,
                                  name=f"reg{reg_num}")
        data_regs.append(data_reg)
      fifo_wr.assign(And(*reg_valids))
      data_regs.reverse()
      msg = BitVectorValue.concat(data_regs)[0:msg_type.bitwidth]

      outch_ready_wire = Wire(types.i1)
      fifo_has_item = Wire(types.i1)
      fifo = SimpleXilinxFifo(msg_type, depth=32, almost_full=4)(
          clk=clk,
          rst=rst,
          rd_en=outch_ready_wire & fifo_has_item,
          wr_en=fifo_wr,
          wr_data=msg)
      fifo_has_item.assign(~fifo.empty)
      outch, outch_ready = types.channel(msg_type).wrap(
          fifo.rd_data, fifo_has_item.reg(clk, rst))
      outch_ready_wire.assign(outch_ready)
      ports.msgs = outch
      ports.fifo_almost_full = fifo.almost_full

  return RegsToChannel


@module
def ChannelToRegs(type):

  class ChannelToRegs:
    """Convert a message stream into a set of `axil_data_width` registers."""
    clk = Clock()
    rst = Input(types.i1)
    num_regs = int(math.ceil(type.inner.bitwidth / axil_data_width))

    regs_valid = Output(types.i1)
    regs_ready = Input(types.i1)
    regs = Output(types.array(types.int(axil_data_width), num_regs))

    msgs = InputChannel(type)

    @generator
    def generate(ports):
      (msg, valid) = ports.msgs.unwrap(ports.regs_ready)
      ports.regs_valid = valid

      width = msg.type.bitwidth
      bits = hw.BitcastOp(types.int(width), msg)
      bits_padded = bits.pad_or_truncate(ChannelToRegs.num_regs *
                                         axil_data_width)
      ports.regs = hw.BitcastOp(
          types.array(types.int(axil_data_width), ChannelToRegs.num_regs),
          bits_padded)

  return ChannelToRegs


def XrtBSP(user_module):

  @esi.ServiceImplementation(None)
  class XrtService:
    clk = Clock(types.i1)
    rst = Input(types.i1)

    axil_in = Input(axil_in_type(axil_addr_width, axil_data_width))
    axil_out = Output(axil_out_type(axil_data_width))

    @generator
    def generate(ports: _GeneratorPortAccess, channels):
      clk = ports.clk
      rst = ports.rst

      # Generate MMIO map and output it both to tcl and to json.
      from_host_regs, to_host_regs = generate_mmio_map(channels)
      sys: System = System.current()
      output_tcl(from_host_regs, to_host_regs,
                 (sys.hw_output_dir / "xrt_package.tcl").open("w"))
      chan_md_file = (sys.sys_runtime_output_dir /
                      "xrt_mmio_descriptor.json").open("w")
      chan_md_file.write(
          json.dumps(
              {
                  "from_host_regs": [asdict(r) for r in from_host_regs],
                  "to_host_regs": [asdict(r) for r in to_host_regs]
              },
              indent=2))

      # Build the write side.
      write_fifos_full = Wire(types.i1)
      reg_write_happened = Wire(types.i1)
      address_valid = ControlReg(clk, rst, [ports.axil_in.awvalid],
                                 [reg_write_happened])
      awready = ~address_valid
      data_valid = ControlReg(clk, rst, [ports.axil_in.wvalid],
                              [reg_write_happened])
      wready = ~data_valid & ~write_fifos_full
      reg_write_happened.assign(address_valid & data_valid & ~write_fifos_full)
      reg_write_happened.name = "reg_write_happened"

      wr_address_reg = ports.axil_in.awaddr.reg(clk=clk,
                                                rst=rst,
                                                ce=ports.axil_in.awvalid)
      wr_data_reg = ports.axil_in.wdata.reg(clk=clk,
                                            rst=rst,
                                            ce=ports.axil_in.wvalid)

      from_host_fifos_full = [types.i1(0)]
      for req, reg_spec in zip(channels.to_client_reqs, from_host_regs):
        if reg_spec.type != "direct_mmio":
          continue
        adapter = RegsToChannel(req.type.inner,
                                reg_spec.offset)(clk=clk,
                                                 rst=rst,
                                                 valid=reg_write_happened,
                                                 address=wr_address_reg,
                                                 data=wr_data_reg)
        req.assign(adapter.msgs)
        from_host_fifos_full.append(adapter.fifo_almost_full)

      # Build the read side.
      rd_addr_data = {}
      for req, reg_spec in zip(channels.to_server_reqs, to_host_regs):
        if reg_spec.type != "direct_mmio":
          continue
        ready = Wire(types.i1, "rready")
        adapter = ChannelToRegs(req.type)(clk=clk,
                                          rst=rst,
                                          regs_ready=ready,
                                          msgs=req.value)

        # Control reg
        reg_write_happened_control = reg_write_happened & (
            ports.axil_in.awaddr == types.int(axil_addr_width)(reg_spec.offset))
        done = reg_write_happened_control & (
            wr_data_reg == types.int(axil_data_width)(DONE_BIT))
        valid = ControlReg(clk, rst, [adapter.regs_valid & ready], [done])
        ready.assign(~valid)
        rd_addr_data[reg_spec.offset] = BitVectorValue.concat(
            [valid]).pad_or_truncate(axil_data_width)
        registered_regs = adapter.regs.reg(clk,
                                           rst,
                                           ce=adapter.regs_valid & ready)
        for i in range(adapter.num_regs):
          rd_addr_data[reg_spec.data_offset +
                       (i * axil_data_width_bytes)] = registered_regs[i]

      max_addr_log2 = int(
          math.ceil(math.log2(max([a for a in rd_addr_data.keys()]) + 1)))
      zero = types.int(axil_data_width)(0)
      rd_space = [zero] * int(math.pow(2, max_addr_log2))
      for (addr, val) in rd_addr_data.items():
        rd_space[addr] = val

      addr_slice = ports.axil_in.araddr.slice(
          types.int(axil_addr_width)(0), max_addr_log2)
      rd_addr = addr_slice.reg(clk, rst)
      rvalid = ports.axil_in.arvalid.reg(clk, rst)
      rdata = types.array(types.int(axil_data_width),
                          len(rd_space))(rd_space)[rd_addr]

      write_fifos_full.assign(Or(*from_host_fifos_full))
      write_fifos_full.name = "write_fifos_full"

      # Assign the outputs.
      ports.axil_out = axil_out_type(axil_data_width)({
          "awready": awready,
          "wready": wready,
          "arready": 1,
          "rvalid": rvalid,
          "rdata": rdata,
          "rresp": 0,
          "bvalid": reg_write_happened,
          "bresp": 0
      })

  @module
  class top:
    ap_clk = Clock()
    ap_resetn = Input(types.i1)

    # AXI4-Lite slave interface
    s_axi_control_AWVALID = Input(types.i1)
    s_axi_control_AWREADY = Output(types.i1)
    s_axi_control_AWADDR = Input(types.int(32))
    s_axi_control_WVALID = Input(types.i1)
    s_axi_control_WREADY = Output(types.i1)
    s_axi_control_WDATA = Input(types.int(axil_data_width))
    s_axi_control_WSTRB = Input(types.int(axil_data_width // 8))
    s_axi_control_ARVALID = Input(types.i1)
    s_axi_control_ARREADY = Output(types.i1)
    s_axi_control_ARADDR = Input(types.int(axil_addr_width))
    s_axi_control_RVALID = Output(types.i1)
    s_axi_control_RREADY = Input(types.i1)
    s_axi_control_RDATA = Output(types.int(axil_data_width))
    s_axi_control_RRESP = Output(types.i2)
    s_axi_control_BVALID = Output(types.i1)
    s_axi_control_BREADY = Input(types.i1)
    s_axi_control_BRESP = Output(types.i2)

    @generator
    def construct(ports):

      axil_in_sig = axil_in_type(axil_addr_width, axil_data_width)({
          "awvalid": ports.s_axi_control_AWVALID,
          "awaddr": ports.s_axi_control_AWADDR[0:24],
          "wvalid": ports.s_axi_control_WVALID,
          "wdata": ports.s_axi_control_WDATA,
          "wstrb": ports.s_axi_control_WSTRB,
          "arvalid": ports.s_axi_control_ARVALID,
          "araddr": ports.s_axi_control_ARADDR,
          "rready": ports.s_axi_control_RREADY,
          "bready": ports.s_axi_control_BREADY,
      })

      rst = ~ports.ap_resetn

      print("WARNING: this XRT bridge is still a work-in-progress! Use at your"
            " own risk!")
      xrt = XrtService(clk=ports.ap_clk, rst=rst, axil_in=axil_in_sig)

      axil_out = xrt.axil_out

      # AXI-Lite control
      ports.s_axi_control_AWREADY = axil_out['awready']
      ports.s_axi_control_WREADY = axil_out['wready']
      ports.s_axi_control_ARREADY = axil_out['arready']
      ports.s_axi_control_RVALID = axil_out['rvalid']
      ports.s_axi_control_RDATA = axil_out['rdata']
      ports.s_axi_control_RRESP = axil_out['rresp']
      ports.s_axi_control_BVALID = axil_out['bvalid']
      ports.s_axi_control_BRESP = axil_out['bresp']

      # Splice in the user's code
      # NOTE: the clock is `ports.ap_clk`
      #       and reset is `ports.ap_resetn` which is active low
      user_module(clk=ports.ap_clk, rst=rst)

      # Copy additional sources
      sys: System = System.current()
      sys.add_packaging_step(top.package)

    @staticmethod
    def package(sys: System):
      from jinja2 import Environment, FileSystemLoader, StrictUndefined

      sv_sources = glob.glob(str(__dir__ / '*.sv'))
      tcl_sources = glob.glob(str(__dir__ / '*.tcl'))
      for source in sv_sources + tcl_sources:
        shutil.copy(source, sys.hw_output_dir)

      env = Environment(loader=FileSystemLoader(str(__dir__)),
                        undefined=StrictUndefined)
      template = env.get_template("Makefile.xrt.j2")
      dst_makefile = sys.output_directory / "Makefile.xrt"
      dst_makefile.open("w").write(template.render(system_name=sys.name))

      runtime_dir = sys.output_directory / "runtime" / sys.name
      so_sources = glob.glob(str(__dir__ / '*.so'))
      for so in so_sources:
        shutil.copy(so, runtime_dir)
      shutil.copy(__dir__ / "xrt_api.py", runtime_dir / "xrt.py")
      shutil.copy(__dir__ / "EsiXrtPython.cpp",
                  sys.sys_runtime_output_dir / "EsiXrtPython.cpp")

  return top
