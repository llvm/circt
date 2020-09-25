//===- signals-runtime-wrappers.cpp - Runtime library implementation ------===//
//
// This file implements the runtime library used in LLHD simulation.
//
//===----------------------------------------------------------------------===//

#include "signals-runtime-wrappers.h"

#include "llvm/ADT/ArrayRef.h"

using namespace llvm;
using namespace circt::llhd::sim;

//===----------------------------------------------------------------------===//
// Runtime interface
//===----------------------------------------------------------------------===//

int alloc_signal(State *state, int index, char *owner, uint8_t *value,
                 int64_t size) {
  assert(state && "alloc_signal: state not found");
  std::string sOwner(owner);

  return state->addSignalData(index, sOwner, value, size);
}

void alloc_proc(State *state, char *owner, ProcState *procState) {
  assert(state && "alloc_proc: state not found");
  std::string sOwner(owner);
  state->addProcPtr(sOwner, procState);
}

void drive_signal(State *state, SignalDetail *detail, uint8_t *value,
                  uint64_t width, int time, int delta, int eps) {
  assert(state && "drive_signal: state not found");

  auto globalIndex = detail->globalIndex;
  auto offset = detail->offset;

  int size = llvm::divideCeil(width + offset, 8);

  APInt drive(width,
              ArrayRef<uint64_t>(reinterpret_cast<uint64_t *>(value), size));

  Time sTime(time, delta, eps);

  int bitOffset =
      (detail->value - state->signals[globalIndex].value) * 8 + offset;

  // Spawn a new event.
  state->pushQueue(sTime, globalIndex, bitOffset, drive);
}

void llhd_suspend(State *state, ProcState *procState, int time, int delta,
                  int eps) {
  std::string instS(procState->inst);
  // Add a new scheduled wake up if a time is specified.
  if (time || delta || eps) {
    Time sTime(time, delta, eps);
    state->pushQueue(sTime, instS);
  }
}
