//===- signals-runtime-wrappers.h - Simulation runtime library --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the runtime library used in LLHD simulation.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_SIMULATOR_SIGNALS_RUNTIME_WRAPPERS_H
#define CIRCT_DIALECT_LLHD_SIMULATOR_SIGNALS_RUNTIME_WRAPPERS_H

#include "circt/Dialect/LLHD/Simulator/State.h"

extern "C" {

using namespace circt::llhd::sim;

//===----------------------------------------------------------------------===//
// Runtime interfaces
//===----------------------------------------------------------------------===//

/// Allocate a new signal. The index of the new signal in the state's list of
/// signals is returned.
/// @param state Simulation context
/// @param index Signal index in global signal container
/// @param value Actual signal value
/// @param size  The size of signal in bytes
int allocSignal(State *state, int index, char *owner,
                uint8_t *value, int64_t size);

/// Add offset and size information for the elements of an array signal.
void addSigArrayElements(State *state, unsigned index,
                         unsigned size, unsigned numElements);

/// Add offset and size information for one element of a struct signal. Elements
/// are assumed to be added (by calling this function) in sequential order, from
/// first to last.
void addSigStructElement(State *state, unsigned index,
                         unsigned offset, unsigned size);

/// Add allocated constructs to a process instance.
void allocProc(State *state, char *owner, ProcState *procState);

/// Add allocated entity state to the given instance.
void allocEntity(State *state, char *owner, uint8_t *entityState);

/// Drive a value onto a signal.
void driveSignal(State *state, SignalDetail *index, uint8_t *value,
                 uint64_t width, int time, int delta, int eps);

/// Suspend a process.
void llhdSuspend(State *state, ProcState *procState,
                 int time, int delta, int eps);
}

#endif // CIRCT_DIALECT_LLHD_SIMULATOR_SIGNALS_RUNTIME_WRAPPERS_H
