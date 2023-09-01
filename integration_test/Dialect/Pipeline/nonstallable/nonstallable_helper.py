import cocotb
from cocotb.triggers import Timer
import cocotb.clock

v = 1

async def clock(dut):
  global v
  dut.clock.value = 0
  await Timer(1, units='ns')
  v += 1
  dut.arg0.value = v
  dut.clock.value = 1
  await Timer(1, units='ns')


async def initDut(dut):
  """
  Initializes a dut by adding a clock, setting initial valid and ready flags,
  and performing a reset.
  """
  # Reset
  dut.reset.value = 1
  await clock(dut)
  dut.reset.value = 0
  await clock(dut)


async def nonstallable_test(dut, nStages, nonStallableStageIdxs):
  """
  Runs a test of a non-stallable pipeline.

  Provided the number of stages in the pipeline and the indices of the non-stallable
  stages, the test will:

  1. for NStallCycles : range(0 to nStages - 1):
    1. fill the pipeline with valid tokens
    2. raise the stall signal
    3. wait for NStallCycles
    4. While doing so, check that len(nonStallableStageIdxs) dut.done assertions
        occur
    5. deassert stall
    6. check that bubbles exit the pipeline in the order that nonStallableStageIdxs
       occured in the pipeline.
  """
  numNonstallableStages = len(nonStallableStageIdxs)
  dut.go.value = 0
  dut.stall.value = 0
  dut.arg0.value = 42
  await initDut(dut)
  dut.go.value = 1


  for nStallCycles in range(1, nStages*2):
    dut.stall.value = 0
    # Fill the pipeline with valid tokens
    for i in range(2 * nStages):
      await clock(dut)

    nBufferedTokensExpected = min(numNonstallableStages, nStallCycles)
    nBubblesExpected = nBufferedTokensExpected

    # Raise the stall signal. We now expect that numNonStallableStages dut.done
    # assertions will occur in a row.
    dut.stall.value = 1
    for stallCycle in range(nStallCycles):
      if nBufferedTokensExpected > 0:
        assert dut.done == 1, f"expected dut.done to be asserted in stall cycle {stallCycle}"
        nBufferedTokensExpected -= 1
      else:
        assert dut.done == 0, f"expected dut.done to be deasserted in stall cycle {stallCycle}"
      await clock(dut)

