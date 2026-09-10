import cocotb
from cocotb.triggers import Timer
import cocotb.clock

# Value that will be adjusted and assigned to the dut input argument, every
# clock cycle. This is mainly to assist manual verification of the VCD trace/
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


async def nonstallable_test(dut, stageStallability):
  """
  Runs a test of a non-stallable pipeline.

  Provided the number of stages in the pipeline and the indices of the non-stallable
  stages, the test will:

  1. for NStallCycles : range(0 to nStages - 1):
    1. for fillCycles : range(0 to nStages - 1):
      1. fill the pipeline with $fillCycles valid tokens
      2. raise the stall signal
      3. wait for NStallCycles
      4. While doing so, check that len(nonStallableStageIdxs) dut.done assertions
          occur
      5. deassert stall
      6. check that the expected amount of bubbles exit the pipeline
  """
  nStages = len(stageStallability)
  numNonstallableStages = sum(
      [1 if not stallable else 0 for stallable in stageStallability])

  for nStallCycles in range(0, nStages * 2):
    for fillCycles in range(0, nStages + 1):
      print(f"nStallCycles: {nStallCycles}, fillCycles: {fillCycles}")
      # Reset the dut
      dut.go.value = 0
      dut.stall.value = 0
      dut.arg0.value = 42
      await initDut(dut)
      dut.stall.value = 0

      # Fill the pipeline with fillCycles valid tokens
      dut.go.value = 1
      for i in range(fillCycles):
        await clock(dut)
      dut.go.value = 0

      nBufferedTokensExpected = min(numNonstallableStages, nStallCycles,
                                    fillCycles)
      nBubblesExpected = nBufferedTokensExpected

      # Raise the stall signal. We now expect that numNonStallableStages dut.done
      # assertions will occur in a row.
      sequence = []
      dut.stall.value = 1
      for cycle in range(nStallCycles + nStages):
        if cycle > nStallCycles:
          dut.stall.value = 0

        await Timer(1, units='ns')
        sequence.append(int(dut.done))
        await clock(dut)

      # Check that exactly fill_cycles tokens exited the pipeline
      nTokensExited = sum(sequence)
      assert nTokensExited == fillCycles, f"Expected {fillCycles} tokens to exit the pipeline, but {nTokensExited} did"
