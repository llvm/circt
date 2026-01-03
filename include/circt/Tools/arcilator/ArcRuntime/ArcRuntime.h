//===- ArcRuntime.h - ArcRuntime public API -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ArcRuntime's public API.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_ARCILATOR_ARCRUNTIME_ARCRUNTIME_H
#define CIRCT_TOOLS_ARCILATOR_ARCRUNTIME_ARCRUNTIME_H

#include "ArcRuntime/Common.h"

// `ARC_RUNTIME_ENABLE_EXPORT` must be set when compiling the runtime.
// Do not set when using (i.e., linking against) it.

#ifndef ARC_RUNTIME_EXPORT
#ifdef ARC_RUNTIME_ENABLE_EXPORT

#ifdef _WIN32
#define ARC_RUNTIME_EXPORT extern "C" __declspec(dllexport)
#else
#define ARC_RUNTIME_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#else // #ifndef ARC_RUNTIME_EXPORT

#ifdef _WIN32
#define ARC_RUNTIME_EXPORT extern "C" __declspec(dllimport)
#else
#define ARC_RUNTIME_EXPORT extern "C"
#endif

#endif // #ifdef ARC_RUNTIME_ENABLE_EXPORT
#endif // #ifndef ARC_RUNTIME_EXPORT

// Host architecture checks
#ifndef __cplusplus
#include <assert.h>
#endif

static_assert(sizeof(void *) == 8, "Unsupported architecture");
static_assert(sizeof(struct ArcState) == 16, "Unexpected ArcState size");
static_assert(sizeof(struct ArcRuntimeModelInfo) == 24,
              "Unexpected ArcRuntimeModelInfo size");

/// Allocate and initialize the state for a new instance of the given
/// hardware model.
///
/// After the end of simulation, the state must be deallocated by calling
/// `arcRuntimeDeleteInstance`.
///
/// `args` is a zero terminated string containing implementation specific
/// runtime options or `null`.
ARC_RUNTIME_EXPORT struct ArcState *
arcRuntimeAllocateInstance(const struct ArcRuntimeModelInfo *model,
                           const char *args);

/// Destroy and deallocate the state of a model instance.
ARC_RUNTIME_EXPORT void arcRuntimeDeleteInstance(struct ArcState *instance);

/// Pre-Eval hook. Must be called by the driver once before every `eval` step.
ARC_RUNTIME_EXPORT void arcRuntimeOnEval(struct ArcState *instance);

/// Return the API version of the runtime library.
ARC_RUNTIME_EXPORT uint64_t arcRuntimeGetAPIVersion();

/// Project a pointer to the model state to its ArcState container.
/// `offset` is the byte offset of the given pointer within the model state.
ARC_RUNTIME_EXPORT struct ArcState *
arcRuntimeGetStateFromModelState(uint8_t *modelState, uint64_t offset);

#endif // CIRCT_TOOLS_ARCILATOR_ARCRUNTIME_ARCRUNTIME_H
