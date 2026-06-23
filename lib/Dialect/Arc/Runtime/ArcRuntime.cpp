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

#include <algorithm>
#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

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

namespace {

uint64_t pow10u64(uint32_t exp) {
  static const uint64_t kPow10[] = {
      1ull,
      10ull,
      100ull,
      1000ull,
      10000ull,
      100000ull,
      1000000ull,
      10000000ull,
      100000000ull,
      1000000000ull,
      10000000000ull,
      100000000000ull,
      1000000000000ull,
      10000000000000ull,
      100000000000000ull,
      1000000000000000ull,
      10000000000000000ull,
      100000000000000000ull,
      1000000000000000000ull,
  };
  if (exp >= (sizeof(kPow10) / sizeof(kPow10[0])))
    return 0;
  return kPow10[exp];
}

bool getBit(const llvm::APInt &value, unsigned bit) {
  return value.extractBits(1, bit).getBoolValue();
}

uint32_t getBits(const llvm::APInt &value, unsigned start, unsigned width) {
  return value.extractBits(width, start).getZExtValue();
}

void trimLeadingZeros(std::string &str) {
  if (str.empty())
    return;
  auto firstNonZero = str.find_first_not_of('0');
  if (firstNonZero == std::string::npos) {
    if (str.size() > 1)
      str.erase(0, str.size() - 1);
    return;
  }
  if (firstNonZero > 0)
    str.erase(0, firstNonZero);
}

void emitPadded(llvm::raw_ostream &os, llvm::StringRef str, bool isLeftAligned,
                char paddingChar, int32_t minWidth) {
  if (minWidth < 0)
    minWidth = 0;
  if (paddingChar == '\0')
    paddingChar = ' ';
  int32_t padCount = 0;
  if (static_cast<int32_t>(str.size()) < minWidth)
    padCount = minWidth - static_cast<int32_t>(str.size());

  if (!isLeftAligned)
    for (int32_t i = 0; i < padCount; ++i)
      os << paddingChar;
  os << str;
  if (isLeftAligned)
    for (int32_t i = 0; i < padCount; ++i)
      os << paddingChar;
}

void formatExactIntegerByRadix(llvm::raw_ostream &os, int width, int radix,
                               const llvm::APInt &value, bool isUpperCase,
                               bool isLeftAligned, char paddingChar,
                               int specifierWidth, bool isSigned) {
  llvm::SmallVector<char, 32> strBuf;
  value.toString(strBuf, radix, isSigned && radix == 10, false, isUpperCase);
  std::string out(strBuf.begin(), strBuf.end());
  emitPadded(os, out, isLeftAligned, paddingChar, specifierWidth);
}

void formatFVIntegerByRadix(llvm::raw_ostream &os, int width, int radix,
                            const llvm::APInt &value,
                            const llvm::APInt &unknown, bool isUpperCase,
                            bool isLeftAligned, char paddingChar,
                            int specifierWidth, bool isSigned) {
  if (unknown.isZero()) {
    formatExactIntegerByRadix(os, width, radix, value, isUpperCase,
                              isLeftAligned, paddingChar, specifierWidth,
                              isSigned);
    return;
  }

  std::string out;
  switch (radix) {
  case 2:
    out.reserve(width);
    for (int b = width; b > 0; --b) {
      auto bit = static_cast<unsigned>(b - 1);
      if (!getBit(unknown, bit)) {
        out.push_back(getBit(value, bit) ? '1' : '0');
        continue;
      }
      bool isZ = getBit(value, bit);
      out.push_back(isZ ? (isUpperCase ? 'Z' : 'z')
                        : (isUpperCase ? 'X' : 'x'));
    }
    break;
  case 8:
  case 16: {
    unsigned groupBits = radix == 8 ? 3 : 4;
    unsigned digits = llvm::divideCeil(width, static_cast<int>(groupBits));
    out.reserve(digits);
    for (unsigned d = digits; d > 0; --d) {
      unsigned startBit = (d - 1) * groupBits;
      unsigned bitsThisDigit =
          std::min(groupBits, width > static_cast<int>(startBit)
                                  ? static_cast<unsigned>(width) - startBit
                                  : 0u);
      if (bitsThisDigit == 0) {
        out.push_back('0');
        continue;
      }

      uint32_t digitUnknown = getBits(unknown, startBit, bitsThisDigit);
      if (digitUnknown == 0) {
        uint32_t digitVal = getBits(value, startBit, bitsThisDigit);
        if (digitVal < 10)
          out.push_back(static_cast<char>('0' + digitVal));
        else
          out.push_back(
              static_cast<char>((isUpperCase ? 'A' : 'a') + (digitVal - 10)));
        continue;
      }

      uint32_t mask = (1u << bitsThisDigit) - 1u;
      uint32_t digitVal = getBits(value, startBit, bitsThisDigit) & mask;
      if (digitUnknown == mask) {
        if (digitVal == mask) {
          out.push_back(isUpperCase ? 'Z' : 'z');
          continue;
        }
        if (digitVal == 0) {
          out.push_back(isUpperCase ? 'X' : 'x');
          continue;
        }
      }
      out.push_back(isUpperCase ? 'X' : 'x');
    }
    break;
  }
  case 10:
    out.push_back(unknown.isAllOnes() && value.isAllOnes()
                      ? (isUpperCase ? 'Z' : 'z')
                      : (isUpperCase ? 'X' : 'x'));
    break;
  default:
    out = "<unsupported base>";
    break;
  }

  trimLeadingZeros(out);
  emitPadded(os, out, isLeftAligned, paddingChar, specifierWidth);
}

int32_t timeformatUnit = -15;
int32_t timeformatPrecision = 0;
std::string timeformatSuffix;
int32_t timeformatMinWidth = 20;

void formatTime(llvm::raw_ostream &os, int64_t timeFs, int32_t widthOverride) {
  int32_t unit = timeformatUnit;
  int32_t precision = std::clamp(timeformatPrecision, 0, 18);
  int32_t fieldWidth = widthOverride >= 0 ? widthOverride : timeformatMinWidth;
  if (fieldWidth < 0)
    fieldWidth = 0;

  bool neg = timeFs < 0;
  uint64_t absFs = neg ? static_cast<uint64_t>(-(timeFs + 1)) + 1
                       : static_cast<uint64_t>(timeFs);

  uint32_t unitPow = static_cast<uint32_t>(unit + 15);
  uint64_t scale = pow10u64(static_cast<uint32_t>(precision));
  if (scale == 0)
    scale = 1;

  uint64_t scaled;
  if (static_cast<uint32_t>(precision) >= unitPow) {
    uint64_t factor = pow10u64(static_cast<uint32_t>(precision) - unitPow);
    if (factor == 0)
      factor = 1;
    scaled = absFs * factor;
  } else {
    uint64_t divisor = pow10u64(unitPow - static_cast<uint32_t>(precision));
    if (divisor == 0)
      divisor = 1;
    scaled = (absFs + divisor / 2) / divisor;
  }

  uint64_t intPart = precision == 0 ? scaled : (scaled / scale);
  uint64_t fracPart = precision == 0 ? 0 : (scaled % scale);

  std::string num;
  if (neg)
    num.push_back('-');
  num.append(std::to_string(intPart));
  if (precision > 0) {
    num.push_back('.');
    auto fracStr = std::to_string(fracPart);
    if (fracStr.size() < static_cast<size_t>(precision))
      num.append(static_cast<size_t>(precision) - fracStr.size(), '0');
    num.append(fracStr);
  }

  auto formattedWidth = num.size() + timeformatSuffix.size();
  if (fieldWidth > 0 && static_cast<int32_t>(formattedWidth) < fieldWidth)
    os.indent(fieldWidth - static_cast<int32_t>(formattedWidth));
  os << num << timeformatSuffix;
}

void formatString(llvm::raw_ostream &os, const char *value,
                  const FmtDescriptor::StringFmt &fmt) {
  llvm::StringRef str(value ? value : "");
  emitPadded(os, str, fmt.isLeftAligned, fmt.paddingChar, fmt.specifierWidth);
}

void formatReal(llvm::raw_ostream &os, double value,
                const FmtDescriptor::RealFmt &fmt) {
  int fieldWidth = fmt.fieldWidth > 0 ? fmt.fieldWidth : 0;
  int fracDigits = fmt.fracDigits >= 0 ? fmt.fracDigits : 6;
  char conversion = fmt.format;
  if (conversion != 'e' && conversion != 'f' && conversion != 'g')
    conversion = 'g';

  char spec[32];
  if (fmt.isLeftAligned)
    std::snprintf(spec, sizeof(spec), "%%-%d.%d%c", fieldWidth, fracDigits,
                  conversion);
  else if (fieldWidth > 0)
    std::snprintf(spec, sizeof(spec), "%%%d.%d%c", fieldWidth, fracDigits,
                  conversion);
  else
    std::snprintf(spec, sizeof(spec), "%%.%d%c", fracDigits, conversion);

  int numChars = std::snprintf(nullptr, 0, spec, value);
  if (numChars <= 0)
    return;
  std::string buffer(static_cast<size_t>(numChars) + 1, '\0');
  std::snprintf(buffer.data(), buffer.size(), spec, value);
  os << std::string_view(buffer.data(), static_cast<size_t>(numChars));
}

void formatDescriptors(llvm::raw_ostream &os, const FmtDescriptor *fmt,
                       va_list args) {
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
    case FmtDescriptor::Action_IntExact: {
      uint64_t *words = va_arg(args, uint64_t *);
      int64_t numWords = llvm::divideCeil(fmt->intFmt.bitwidth, 64);
      llvm::APInt apInt(fmt->intFmt.bitwidth, llvm::ArrayRef(words, numWords));
      formatExactIntegerByRadix(
          os, fmt->intFmt.bitwidth, fmt->intFmt.radix, apInt,
          fmt->intFmt.isUpperCase, fmt->intFmt.isLeftAligned,
          fmt->intFmt.paddingChar, fmt->intFmt.specifierWidth,
          fmt->intFmt.isSigned);
      break;
    }
    case FmtDescriptor::Action_FVInt: {
      uint64_t *valueWords = va_arg(args, uint64_t *);
      uint64_t *unknownWords = va_arg(args, uint64_t *);
      int64_t numWords = llvm::divideCeil(fmt->intFmt.bitwidth, 64);
      llvm::APInt value(fmt->intFmt.bitwidth,
                        llvm::ArrayRef(valueWords, numWords));
      llvm::APInt unknown(fmt->intFmt.bitwidth,
                          llvm::ArrayRef(unknownWords, numWords));
      formatFVIntegerByRadix(os, fmt->intFmt.bitwidth, fmt->intFmt.radix, value,
                             unknown, fmt->intFmt.isUpperCase,
                             fmt->intFmt.isLeftAligned, fmt->intFmt.paddingChar,
                             fmt->intFmt.specifierWidth, fmt->intFmt.isSigned);
      break;
    }
    case FmtDescriptor::Action_Char: {
      char c = static_cast<char>(va_arg(args, int));
      llvm::StringRef str(&c, 1);
      emitPadded(os, str, fmt->stringFmt.isLeftAligned,
                 fmt->stringFmt.paddingChar, fmt->stringFmt.specifierWidth);
      break;
    }
    case FmtDescriptor::Action_Time:
      formatTime(os, va_arg(args, int64_t), fmt->timeFmt.widthOverride);
      break;
    case FmtDescriptor::Action_String:
      formatString(os, va_arg(args, const char *), fmt->stringFmt);
      break;
    case FmtDescriptor::Action_Real:
      formatReal(os, va_arg(args, double), fmt->realFmt);
      break;
    case FmtDescriptor::Action_End:
      break;
    }
    fmt++;
  }
}

} // namespace

void arcRuntimeIR_setTimeFormat(int32_t unit, int32_t precision,
                                const char *suffix, int32_t minFieldWidth) {
  timeformatUnit = std::clamp(unit, -15, 0);
  timeformatPrecision = std::clamp(precision, 0, 18);
  timeformatSuffix = suffix ? suffix : "";
  timeformatMinWidth = std::max(minFieldWidth, 0);
}

void arcRuntimeIR_format(const FmtDescriptor *fmt, ...) {
  va_list args;
  va_start(args, fmt);

  formatDescriptors(llvm::outs(), fmt, args);

  va_end(args);
}

const char *arcRuntimeIR_formatToString(const FmtDescriptor *fmt, ...) {
  va_list args;
  va_start(args, fmt);

  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  formatDescriptors(os, fmt, args);
  os.flush();

  va_end(args);

  char *out = static_cast<char *>(std::malloc(buffer.size() + 1));
  if (!out)
    return "";
  std::memcpy(out, buffer.data(), buffer.size());
  out[buffer.size()] = '\0';
  return out;
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
    &arcRuntimeIR_allocInstance, &arcRuntimeIR_deleteInstance,
    &arcRuntimeIR_onEval,        &arcRuntimeIR_onInitialized,
    &arcRuntimeIR_format,        &arcRuntimeIR_formatToString,
    &arcRuntimeIR_setTimeFormat, &arcRuntimeIR_swapTraceBuffer};

const APICallbacks &getArcRuntimeAPICallbacks() { return apiCallbacksGlobal; }

} // namespace circt::arc::runtime
#endif
