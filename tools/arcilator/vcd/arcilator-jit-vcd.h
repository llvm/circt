//===- xxx.h - Utils for debugging MLIR execution ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ARC_JIT_VCD_H
#define MLIR_ARC_JIT_VCD_H

#include <cstdint>

namespace arcilator {
// Common Tracer Library Structs

// Dynamic tracer state. Prepended to the simulation state.
struct ArcTracerState {
  // The current trace buffer.
  void *buffer;
  // The current size of the trace buffer.
  uint64_t size;
  // The current trace buffer's capacity.
  uint64_t capacity;
  // (Currently Unused)
  uint64_t runSteps;
  // The 'user' pointer returned by `initModel`.
  void *user;
  // The simulation state
  uint8_t simulationState[];
};

// Static Model Information
struct ArcTraceModelInfo {
  // Number of signal taps in the model
  uint64_t numTraceTaps;
  // Name of the model
  char *modelName;
  // Consecutive signal names and aliases
  char *signalNames;
  // Offset to the first character of the _next_ signal
  // name in the `signalNames` array per tap.
  uint64_t *tapNameOffsets;
  // Type descriptor array per tap.
  // At the moment, just the bitwidth.
  uint32_t *typeDescriptors;
  // Per tap ffset in bytes of the signal in the simulation state.
  uint64_t *stateOffsets;
};

// Library callback functions
struct ArcTraceLibrary {
  // Called after state allcoation.
  // Must return a poiter to the "user" struct.
  void *(*initModel)(const ArcTraceModelInfo *);
  // Called before each evaluation step.
  void (*step)(ArcTracerState *);
  // Called when the trace buffer hits its capacity.
  // Passes the pointer to the old buffer and the
  // size to which it has been filled.
  // Must return a pointer to a new buffer of at least
  // the old buffers capacity.
  void *(*swapBuffer)(void *, uint64_t, void *);
  // Called before state deallocation. Must free any
  // allocated model resources.
  void (*closeModel)(ArcTracerState *);
};

static_assert(sizeof(void *) == 8);
static_assert(sizeof(ArcTraceModelInfo) == 6 * 8);
static_assert(sizeof(ArcTracerState) == 5 * 8);

namespace vcd {

const arcilator::ArcTraceLibrary *getVcdTraceLibrary();

} // namespace vcd
} // namespace arcilator

#endif // MLIR_ARC_JIT_VCD_H
