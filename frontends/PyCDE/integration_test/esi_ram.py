# REQUIRES: esi-runtime, esi-cosim, rtl-sim
# RUN: rm -rf %t
# RUN: mkdir %t && cd %t
# RUN: %PYTHON% %s %t 2>&1
# RUN: esi-cosim-runner.py --tmpdir %t --exec %S/test_software/esi_ram.py `ls %t/hw/*.sv | grep -v driver.sv`

import pycde
from pycde import (AppID, Clock, Input, Module, generator)
from pycde.esi import DeclareRandomAccessMemory, ServiceDecl
from pycde.bsp import cosim
from pycde.types import Bits

import sys

RamI64x8 = DeclareRandomAccessMemory(Bits(64), 8)
WriteType = RamI64x8.write.type.write


@ServiceDecl
class MemComms:
  write = ServiceDecl.From(RamI64x8.write.type)
  read = ServiceDecl.From(RamI64x8.read.type)


class MemWriter(Module):
  """Write to address 3 the contents of address 2."""

  clk = Clock()
  rst = Input(Bits(1))

  @generator
  def construct(ports):
    read_bundle_type = RamI64x8.read.type
    address = 2
    (address_chan, ready) = read_bundle_type.address.wrap(address, True)
    read_bundle, [data_chan] = read_bundle_type.pack(address=address_chan)
    read_data, read_valid = data_chan.unwrap(True)
    RamI64x8.read(read_bundle, AppID("int_reader"))

    write_bundle_type = RamI64x8.write.type
    write_data, _ = write_bundle_type.write.wrap(
        {
            'data': read_data,
            'address': 3
        }, read_valid)
    write_bundle, [ack] = write_bundle_type.pack(write=write_data)
    RamI64x8.write(write_bundle, appid=AppID("int_writer"))


class Top(Module):
  clk = Clock()
  rst = Input(Bits(1))

  @generator
  def construct(ports):
    MemWriter(clk=ports.clk, rst=ports.rst)

    # Pass through reads and writes from the host.
    ram_write = MemComms.write(AppID("write"))
    RamI64x8.write(ram_write, AppID("ram_write"))
    ram_read = MemComms.read(AppID("read"))
    RamI64x8.read(ram_read, AppID("ram_read"))

    # Instantiate the RAM.
    RamI64x8.instantiate_builtin(appid=AppID("mem"),
                                 builtin="sv_mem",
                                 result_types=[],
                                 inputs=[ports.clk, ports.rst])


if __name__ == "__main__":
  s = pycde.System([cosim.CosimBSP(Top)],
                   name="ESIMem",
                   output_directory=sys.argv[1])
  s.compile()
  s.package()
