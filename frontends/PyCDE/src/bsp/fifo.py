import math
from ..module import modparams, generator, Input, Output, Module
from ..types import types
from ..dialects import hw

# See UG974 for additional usage information


# Single clock FIFO
@modparams
def xpm_fifo_sync(CASCADE_HEIGHT: int = 0,
                  DOUT_RESET_VALUE: str = "0",
                  ECC_MODE: str = "no_ecc",
                  FIFO_MEMORY_TYPE: str = "auto",
                  FIFO_READ_LATENCY: int = 1,
                  FIFO_WRITE_DEPTH: int = 2048,
                  FULL_RESET_VALUE: int = 1,
                  PROG_EMPTY_THRESH: int = 10,
                  PROG_FULL_THRESH: int = 10,
                  RD_DATA_COUNT_WIDTH: int = 1,
                  READ_DATA_WIDTH: int = 32,
                  READ_MODE: str = "std",
                  SIM_ASSERT_CHK: int = 0,
                  USE_ADV_FEATURES: str = "0707",
                  WAKEUP_TIME: int = 0,
                  WR_DATA_COUNT_WIDTH: int = 1,
                  WRITE_DATA_WIDTH: int = 32):

  class xpm_fifo_sync(Module):
    almost_empty = Output(types.i1)
    almost_full = Output(types.i1)
    data_valid = Output(types.i1)
    dbiterr = Output(types.i1)
    din = Input(types.int(WRITE_DATA_WIDTH))
    dout = Output(types.int(READ_DATA_WIDTH))
    empty = Output(types.i1)
    full = Output(types.i1)
    injectdbiterr = Input(types.i1)
    injectsbiterr = Input(types.i1)
    overflow = Output(types.i1)
    prog_empty = Output(types.i1)
    prog_full = Output(types.i1)
    rd_data_count = Output(types.int(RD_DATA_COUNT_WIDTH))
    rd_en = Input(types.i1)
    rd_rst_busy = Output(types.i1)
    rst = Input(types.i1)
    sbiterr = Output(types.i1)
    sleep = Input(types.i1)
    underflow = Output(types.i1)
    wr_ack = Output(types.i1)
    wr_clk = Input(types.i1)
    wr_data_count = Output(types.int(WR_DATA_COUNT_WIDTH))
    wr_en = Input(types.i1)
    wr_rst_busy = Output(types.i1)

  return xpm_fifo_sync


# Simplified Single Clock FIFO
@modparams
def SimpleXilinxFifo(type=types.int(32),
                     depth: int = 2048,
                     lookahead: bool = False,
                     almost_full: int = 5,
                     almost_empty: int = 5,
                     ecc: bool = False):

  # force depth to the next power of 2 if not already a power of 2
  depth = 2**math.ceil(math.log2(depth))

  width = type.bitwidth

  class SimpleFifo(Module):
    clk = Input(types.i1)
    rst = Input(types.i1)
    rd_data = Output(type)
    wr_data = Input(type)
    rd_en = Input(types.i1)
    wr_en = Input(types.i1)
    almost_full = Output(types.i1)
    almost_empty = Output(types.i1)
    full = Output(types.i1)
    empty = Output(types.i1)

    @generator
    def generate(mod):
      zero = hw.ConstantOp(types.i1, 0)
      ecc_mode = "en_ecc" if ecc else "no_ecc"
      rd_latency = 0 if lookahead else 1
      rd_mode = "fwft" if lookahead else "std"

      write_bits = hw.BitcastOp(types.int(width), mod.wr_data)

      fifo = xpm_fifo_sync(
          ECC_MODE=ecc_mode,
          FIFO_READ_LATENCY=rd_latency,
          READ_MODE=rd_mode,
          FIFO_WRITE_DEPTH=depth,
          READ_DATA_WIDTH=width,
          WRITE_DATA_WIDTH=width,
          USE_ADV_FEATURES="0202",
          PROG_EMPTY_THRESH=almost_full,
          PROG_FULL_THRESH=almost_empty,
      )(rst=mod.rst,
        wr_clk=mod.clk,
        din=write_bits,
        wr_en=mod.wr_en,
        rd_en=mod.rd_en,
        injectdbiterr=zero,
        injectsbiterr=zero,
        sleep=zero)

      mod.rd_data = hw.BitcastOp(type, fifo.dout)
      mod.almost_full = fifo.prog_full
      mod.almost_empty = fifo.prog_empty
      mod.full = fifo.full
      mod.empty = fifo.empty

  return SimpleFifo
