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
#include "circt/Dialect/Arc/Runtime/FmtDescriptor.h"
#include "circt/Dialect/Arc/Runtime/IRInterface.h"
#include "circt/Dialect/Arc/Runtime/Internal.h"
#include "circt/Dialect/Arc/Runtime/ModelInstance.h"
#include "circt/Support/FormatInteger.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#ifdef ARC_RUNTIME_JIT_BIND
#define ARC_RUNTIME_JITBIND_FNDECL
#include "circt/Dialect/Arc/Runtime/JITBind.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <iostream>

using namespace circt::arc::runtime;

static inline impl::ModelInstance *getModelInstance(const ArcState *instance) {
  assert(instance->impl != nullptr && "Instance is null");
  return reinterpret_cast<impl::ModelInstance *>(instance->impl);
}

ArcState *arcRuntimeAllocateInstance(const ArcRuntimeModelInfo *model,
                                     const char *args) {
  if (model->apiVersion != ARC_RUNTIME_API_VERSION) {
    impl::fatalError("API version mismatch.\nMake sure to use an ArcRuntime "
                     "release matching "
                     "the arcilator version used to build the hardware model.");
    return nullptr;
  }

  auto *statePtr = reinterpret_cast<ArcState *>(
      calloc(1, sizeof(ArcState) + model->numStateBytes));
  assert(reinterpret_cast<intptr_t>(&statePtr->modelState[0]) % 16 == 0 &&
         "Simulation state must be 16 byte aligned");
  statePtr->impl = new impl::ModelInstance(model, args, statePtr);
  statePtr->magic = ARC_RUNTIME_MAGIC;
  return statePtr;
}

void arcRuntimeDeleteInstance(ArcState *instance) {
  if (instance->impl)
    delete reinterpret_cast<impl::ModelInstance *>(instance->impl);
  free(instance);
}

uint64_t arcRuntimeGetAPIVersion() { return ARC_RUNTIME_API_VERSION; }

void arcRuntimeOnEval(ArcState *instance) {
  getModelInstance(instance)->onEval(instance);
}

void arcRuntimeOnInitialized(ArcState *instance) {
  getModelInstance(instance)->onInitialized(instance);
}

ArcState *arcRuntimeGetStateFromModelState(uint8_t *modelState,
                                           uint64_t offset) {
  if (!modelState)
    impl::fatalError("State pointer is null");
  uint8_t *modPtr = static_cast<uint8_t *>(modelState) - offset;
  ArcState *statePtr = reinterpret_cast<ArcState *>(modPtr - sizeof(ArcState));
  if (statePtr->magic != ARC_RUNTIME_MAGIC)
    impl::fatalError("Incorrect magic number for state");
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

void arcRuntimeIR_onInitialized(uint8_t *modelState) {
  arcRuntimeOnInitialized(arcRuntimeGetStateFromModelState(modelState, 0));
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
      std::string_view s(va_arg(args, const char *), fmt->literal.width);
      os << s;
      break;
    }
    case FmtDescriptor::Action_LiteralSmall: {
      std::string_view s(fmt->smallLiteral.data);
      os << s;
      break;
    }
    case FmtDescriptor::Action_Int: {
      uint64_t *words = va_arg(args, uint64_t *);
      int64_t numWords = llvm::divideCeil(fmt->intFmt.bitwidth, 64);
      llvm::APInt apInt(fmt->intFmt.bitwidth, llvm::ArrayRef(words, numWords));
      std::optional<int32_t> specifierWidth;
      if (fmt->intFmt.specifierWidth >= 0)
        specifierWidth = fmt->intFmt.specifierWidth;
      circt::formatInteger(os, apInt, fmt->intFmt.radix,
                           fmt->intFmt.isUpperCase, fmt->intFmt.isLeftAligned,
                           fmt->intFmt.paddingChar, specifierWidth,
                           fmt->intFmt.isSigned);
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

double arcRuntimeIR_realHypot(double lhs, double rhs) {
  return std::hypot(lhs, rhs);
}

uint64_t *arcRuntimeIR_swapTraceBuffer(const uint8_t *modelState) {
  auto *modPtr = static_cast<const uint8_t *>(modelState);
  auto *statePtr =
      reinterpret_cast<const ArcState *>(modPtr - sizeof(ArcState));
  if (statePtr->magic != ARC_RUNTIME_MAGIC)
    impl::fatalError("Incorrect magic number for state");
  return getModelInstance(statePtr)->swapTraceBuffer();
}

#ifdef ARC_RUNTIME_JIT_BIND
namespace circt::arc::runtime {

static const APICallbacks apiCallbacksGlobal{
    &arcRuntimeIR_allocInstance,  &arcRuntimeIR_deleteInstance,
    &arcRuntimeIR_onEval,         &arcRuntimeIR_onInitialized,
    &arcRuntimeIR_format,         &arcRuntimeIR_realHypot,
    &arcRuntimeIR_swapTraceBuffer};

const APICallbacks &getArcRuntimeAPICallbacks() { return apiCallbacksGlobal; }

} // namespace circt::arc::runtime
#endif
