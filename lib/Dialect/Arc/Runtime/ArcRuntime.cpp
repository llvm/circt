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
#include "circt/Dialect/Arc/Runtime/String.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
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

// Emits an integer to `os` using the given format specifiers.
//
// Note that this is a copy of formatIntegersByRadix from SimOps.cpp.
static void formatIntegersByRadix(llvm::raw_ostream &os, int width, int radix,
                                  const llvm::APInt &value, bool isUpperCase,
                                  bool isLeftAligned, char paddingChar,
                                  int specifierWidth, bool isSigned) {
  llvm::SmallVector<char, 32> strBuf;
  value.toString(strBuf, radix, isSigned, false, isUpperCase);
  int strBufSize = static_cast<int>(strBuf.size());

  int padWidth;
  switch (radix) {
  case 2:
    padWidth = width;
    break;
  case 8:
    padWidth = (width + 2) / 3;
    break;
  case 16:
    padWidth = (width + 3) / 4;
    break;
  default:
    padWidth = width;
    break;
  }

  int numSpaces = 0;
  if (specifierWidth >= 0 &&
      (specifierWidth > std::max(padWidth, strBufSize))) {
    numSpaces = std::max(0, specifierWidth - std::max(padWidth, strBufSize));
  }

  llvm::SmallVector<char, 1> spacePadding(numSpaces, ' ');

  padWidth = padWidth > strBufSize ? padWidth - strBufSize : 0;

  llvm::SmallVector<char, 32> padding(padWidth, paddingChar);

  if (isLeftAligned) {
    os << padding << strBuf << spacePadding;
  } else {
    os << spacePadding << padding << strBuf;
  }
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
      formatIntegersByRadix(os, fmt->intFmt.bitwidth, fmt->intFmt.radix, apInt,
                            fmt->intFmt.isUpperCase, fmt->intFmt.isLeftAligned,
                            fmt->intFmt.paddingChar, fmt->intFmt.specifierWidth,
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

uint64_t *arcRuntimeIR_swapTraceBuffer(const uint8_t *modelState) {
  auto *modPtr = static_cast<const uint8_t *>(modelState);
  auto *statePtr =
      reinterpret_cast<const ArcState *>(modPtr - sizeof(ArcState));
  if (statePtr->magic != ARC_RUNTIME_MAGIC)
    impl::fatalError("Incorrect magic number for state");
  return getModelInstance(statePtr)->swapTraceBuffer();
}

void arcRuntimeIR_stringInit(DynamicString *str, const char *initialValue,
                             int64_t initialSize) {
  if (!str || !initialValue) {
    impl::fatalError("Invalid string or initial value");
  }
  str->size = initialSize;
  str->data = new char[initialSize];
  std::memcpy(str->data, initialValue, initialSize);
}

void arcRuntimeIR_stringConcat(DynamicString *outStr, ...) {
  if (!outStr)
    impl::fatalError("Invalid output string or string list");
  va_list args;
  va_start(args, outStr);
  uint64_t totalSize = 0;
  while (true) {
    auto strPtr = va_arg(args, const DynamicString *);
    if ((!strPtr || (!strPtr->data)))
      break;
    totalSize += strPtr->size;
  }
  va_end(args);

  outStr->size = totalSize;
  outStr->data = new char[totalSize];
  char *current = outStr->data;

  va_start(args, outStr);
  while (true) {
    auto strPtr = va_arg(args, const DynamicString *);
    if ((!strPtr || (!strPtr->data)))
      break;
    std::memcpy(current, strPtr->data, strPtr->size);
    current += strPtr->size;
  }
  va_end(args);
}

#ifdef ARC_RUNTIME_JIT_BIND
namespace circt::arc::runtime {

static const APICallbacks apiCallbacksGlobal{
    &arcRuntimeIR_allocInstance, &arcRuntimeIR_deleteInstance,
    &arcRuntimeIR_onEval,        &arcRuntimeIR_onInitialized,
    &arcRuntimeIR_format,        &arcRuntimeIR_swapTraceBuffer,
    &arcRuntimeIR_stringInit,    &arcRuntimeIR_stringConcat};

const APICallbacks &getArcRuntimeAPICallbacks() { return apiCallbacksGlobal; }

} // namespace circt::arc::runtime
#endif
