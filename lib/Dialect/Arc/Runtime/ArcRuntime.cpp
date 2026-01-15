//===- ArcRuntime.cpp - Default implementation of the ArcRuntime-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the default implementation of a runtime library for
// arcilator simulations.
//
//===----------------------------------------------------------------------===//

#define ARC_RUNTIME_ENABLE_EXPORT

#include "circt/Dialect/Arc/Runtime/ArcRuntime.h"
#include "circt/Dialect/Arc/Runtime/Common.h"
#include "circt/Dialect/Arc/Runtime/IRInterface.h"
#include "circt/Dialect/Arc/Runtime/ModelInstance.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#ifdef ARC_RUNTIME_JIT_BIND
#define ARC_RUNTIME_JITBIND_FNDECL
#include "circt/Dialect/Arc/Runtime/JITBind.h"
#endif

#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <iostream>

using namespace circt::arc::runtime;

[[noreturn]] static void internalError(const char *message) {
  std::cerr << "[ArcRuntime] Internal Error: " << message << std::endl;
  assert(false && "ArcRuntime Internal Error");
  abort();
}

static inline ModelInstance *getModelInstance(ArcState *instance) {
  assert(instance->impl != nullptr && "Instance is null");
  return reinterpret_cast<ModelInstance *>(instance->impl);
}

ArcState *arcRuntimeAllocateInstance(const ArcRuntimeModelInfo *model,
                                     const char *args) {
  if (model->apiVersion != ARC_RUNTIME_API_VERSION) {
    internalError("API version mismatch.\nMake sure to use an ArcRuntime "
                  "release matching "
                  "the arcilator version used to build the hardware model.");
    return nullptr;
  }

  auto *statePtr = reinterpret_cast<ArcState *>(
      calloc(1, sizeof(ArcState) + model->numStateBytes));
  assert(reinterpret_cast<intptr_t>(&statePtr->modelState[0]) % 16 == 0 &&
         "Simulation state must be 16 byte aligned");
  statePtr->impl = new ModelInstance(model, args, statePtr);
  statePtr->magic = ARC_RUNTIME_MAGIC;
  return statePtr;
}

void arcRuntimeDeleteInstance(ArcState *instance) {
  if (instance->impl)
    delete reinterpret_cast<ModelInstance *>(instance->impl);
  free(instance);
}

uint64_t arcRuntimeGetAPIVersion() { return ARC_RUNTIME_API_VERSION; }

void arcRuntimeOnEval(ArcState *instance) {
  getModelInstance(instance)->onEval();
}

ArcState *arcRuntimeGetStateFromModelState(uint8_t *modelState,
                                           uint64_t offset) {
  if (!modelState)
    internalError("State pointer is null");
  uint8_t *modPtr = static_cast<uint8_t *>(modelState) - offset;
  ArcState *statePtr = reinterpret_cast<ArcState *>(modPtr - sizeof(ArcState));
  if (statePtr->magic != ARC_RUNTIME_MAGIC)
    internalError("Incorrect magic number for state");
  return statePtr;
}

// --- IR Exports ---

uint8_t *arcRuntimeIR_allocInstance(const ArcRuntimeModelInfo *model,
                                    const char *args) {
  ArcState *statePtr = arcRuntimeAllocateInstance(model, args);
  return statePtr->modelState;
}

void arcRuntimeIR_onEval(uint8_t *modelState) {
  arcRuntimeOnEval(arcRuntimeGetStateFromModelState(modelState, 0));
}

void arcRuntimeIR_deleteInstance(uint8_t *modelState) {
  arcRuntimeDeleteInstance(arcRuntimeGetStateFromModelState(modelState, 0));
}

void arcRuntimeIR_format(const FmtDescriptor *fmt, ...) {
  va_list args;
  va_start(args, fmt);

  llvm::raw_ostream &os = llvm::outs();
  while (fmt->action != FmtDescriptor::Action_End) {
    switch (fmt->action) {
    case FmtDescriptor::Action_Literal: {
      llvm::StringRef s(va_arg(args, const char *), fmt->literal.width);
      os << s;
      break;
    }
    case FmtDescriptor::Action_Int: {
      uint8_t *bytes = va_arg(args, uint8_t *);
      int64_t numWords = llvm::divideCeil(fmt->intFmt.bitwidth, 64);
      std::vector<uint64_t> words(numWords, 0);
      std::memcpy(words.data(), bytes,
                  llvm::divideCeil(fmt->intFmt.bitwidth, 8));
      llvm::APInt apInt(fmt->intFmt.bitwidth, words);

      os << apInt;
      break;
    }
    case FmtDescriptor::Action_Char:
      os << static_cast<char>(va_arg(args, int));
      break;
    case FmtDescriptor::Action_End:
      break;
    }
    fmt++;
  }

  va_end(args);
}

#ifdef ARC_RUNTIME_JIT_BIND
namespace circt::arc::runtime {

static const APICallbacks apiCallbacksGlobal{
    &arcRuntimeIR_allocInstance, &arcRuntimeIR_deleteInstance,
    &arcRuntimeIR_onEval, &arcRuntimeIR_format};

const APICallbacks &getArcRuntimeAPICallbacks() { return apiCallbacksGlobal; }

} // namespace circt::arc::runtime
#endif
