//===- signals-runtime-wrappers.cpp - Runtime library implementation ------===//
//
// This file implements the runtime library used in LLHD simulation.
//
//===----------------------------------------------------------------------===//

#include "signals-runtime-wrappers.h"

#include "llvm/ADT/ArrayRef.h"

using namespace llvm;
using namespace mlir::llhd::sim;

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

SignalDetail *probe_signal(State *state, int index) {
  assert(state && "probe_signal: state not found");
  auto &sig = state->signals[index];
  return &sig.detail;
}

void drive_signal(State *state, int index, uint8_t *value, uint64_t width,
                  int time, int delta, int eps) {
  assert(state && "drive_signal: state not found");

  APInt drive(width, ArrayRef<uint64_t>(reinterpret_cast<uint64_t *>(value),
                                        state->signals[index].size));

  Time sTime(time, delta, eps);

  // Track back origin signal.
  int originIdx = index;
  while (state->signals[originIdx].origin >= 0) {
    originIdx = state->signals[originIdx].origin;
  }

  int bitOffset = (state->signals[index].detail.value -
                   state->signals[originIdx].detail.value) *
                      8 +
                  state->signals[index].detail.offset;

  // Spawn a new event.
  state->pushQueue(sTime, originIdx, bitOffset, drive);
}

int add_subsignal(mlir::llhd::sim::State *state, int origin, uint8_t *ptr,
                  uint64_t len, uint64_t offset) {
  int size = llvm::divideCeil(len + offset, 8);
  state->signals.push_back(Signal(origin, ptr, size, offset));
  return (state->signals.size() - 1);
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
