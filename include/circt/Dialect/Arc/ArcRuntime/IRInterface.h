//===- IRInterface.h - ArcRuntime internal API ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares the runtime functions called from the MLIR model.
//
// This file defines the runtime's internal API. Changes to the internal API
// must be reflected in the lowering passes of the MLIR model.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCRUNTIME_IRINTERFACE_H
#define CIRCT_DIALECT_ARC_ARCRUNTIME_IRINTERFACE_H

// NOLINTBEGIN(readability-identifier-naming)

#ifndef ARC_IR_EXPORT
#ifndef ARC_RUNTIME_JIT_BIND

// Marco definition for the shipped version of the library:
// Symbols have to be visible to allow linking of the precompiled hardware
// model against the static or dynamic runtime library.

#ifdef _WIN32
#define ARC_IR_EXPORT extern "C" __declspec(dllexport)
#else
#define ARC_IR_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#else // #ifndef ARC_RUNTIME_JIT_BIND

// Marco definition for the JIT runner version of the library:
// Symbols are deliberately hidden in the executable's symbol table and
// instead explicitly bound at runtime. This avoids issues due to linker GC
// and internalization.

#ifdef _WIN32
#define ARC_IR_EXPORT extern "C"
#else
#define ARC_IR_EXPORT extern "C" __attribute__((visibility("hidden")))
#endif

#endif // #ifndef ARC_RUNTIME_JIT_BIND
#endif // #ifndef ARC_IR_EXPORT

#include "circt/Dialect/Arc/ArcRuntime/Common.h"

#include <stdint.h>

/// Allocate and initialize the state for a new instance of the given hardware
/// model.
///
/// This function must allocate an `ArcState` struct with at least
/// `model->numStateBytes` bytes provided for the `modelState` array.
/// It must return the pointer to the zero initialized model state which is
/// required to be 16-byte-aligned.
///
/// `args` is a zero terminated string containing implementation specific
/// options for the new instance or `null`.
ARC_IR_EXPORT uint8_t *
arcRuntimeIR_allocInstance(const ArcRuntimeModelInfo *model, const char *args);

/// Destroy and deallocate the state of a model instance.
///
/// This function is responsible for releasing all resources that previously
/// have been allocated by `arcRuntimeIR_allocInstance`.
ARC_IR_EXPORT void arcRuntimeIR_deleteInstance(uint8_t *modelState);

/// Pre-Eval hook of the runtime library.
///
/// Simulation drivers must call this once before every invocation of the
/// model's `eval` function.
ARC_IR_EXPORT void arcRuntimeIR_onEval(uint8_t *modelState);

// NOLINTEND(readability-identifier-naming)
#endif // CIRCT_DIALECT_ARC_ARCRUNTIME_IRINTERFACE_H
