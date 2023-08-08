//===- HWRLELogic.cpp - Run-length encoded logic --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWRLELogic.h"

#include "llvm/ADT/Hashing.h"

using namespace circt;
using namespace hw;
using namespace logicdigits;

using namespace llvm;

// Compute the number of bytes required for encoding and the mask of contained
// digits
RLELogic::AnalyzePassResult RLELogic::analyze(ArrayRef<LogicDigit> digits) {
  RLELogic::AnalyzePassResult result = {};
  LogicDigit prevDigit = LogicDigit::Invalid;
  uint8_t runLength = 0;
  for (auto digit : digits) {
    if (!isValidLogicDigit(digit))
      return {};

    if (digit == prevDigit) {
      runLength++;
      if (runLength > 16) {
        result.requiredBytes++;
        result.lastDigitTrim++;
        runLength = 1;
      }
    } else {
      result.requiredBytes++;
      result.lastDigitTrim = 0;
      result.foundDigitsMask |= makeDigitMask(digit);
      runLength = 1;
      prevDigit = digit;
    }
  }
  // The last digit will be encoded only once
  result.requiredBytes -= result.lastDigitTrim;
  return result;
}

RLELogic RLELogic::encode(ArrayRef<LogicDigit> digits,
                          std::optional<AnalyzePassResult> analysisResult) {
  if (digits.empty())
    return RLELogic(InvalidTag::ZeroLength);

  // Run analysis pass if not already done
  if (!analysisResult.has_value())
    analysisResult = analyze(digits);

  if (!analysisResult->isValid())
    return RLELogic(InvalidTag::InvalidDigit);

  // Allocate heap memory if required or point to a temporary 8-byte stack
  // buffer
  RunLengthCode *rlePtr;
  RawType rleBuffer = 0;
  if (isSelfContained(analysisResult->requiredBytes))
    rlePtr = reinterpret_cast<RunLengthCode *>(&rleBuffer);
  else
    rlePtr = new RunLengthCode[analysisResult->requiredBytes];

  LogicDigit prevDigit = digits.front();
  uint8_t runLength = 1;
  assert(containsAny(analysisResult->foundDigitsMask, prevDigit) &&
         "digit did not occur during analysis");
  SizeType byteCount = 1;

  for (auto digit : digits.drop_front()) {
    assert(containsAny(analysisResult->foundDigitsMask, digit) &&
           "digit did not occur during analysis");

    if (byteCount == analysisResult->requiredBytes)
      break;

    if (digit == prevDigit) {
      // Current digit same as previous, increase run length
      if (runLength == 16) {
        rlePtr[byteCount - 1] = toCode(digit, runLength);
        byteCount++;
        runLength = 1;
      } else {
        runLength++;
      }
    } else {
      // Digit changed
      rlePtr[byteCount - 1] = toCode(prevDigit, runLength);
      byteCount++;
      runLength = 1;
      prevDigit = digit;
    }
  }
  // Encode the most significant digit precisely once, it is implicitly repeated
  // indefinitely.
  rlePtr[byteCount - 1] = toCode(prevDigit, 1);

  // Make sure we did not run out of buffer space before we reached the final
  // digit
  assert(byteCount == analysisResult->requiredBytes &&
         "encoding length does not match analysis result");

  // Hand over the buffer (pointer) to a newly created instance
  if (isSelfContained(byteCount))
    return RLELogic(rleBuffer, byteCount, analysisResult->foundDigitsMask);
  return RLELogic((RawType)rlePtr, byteCount, analysisResult->foundDigitsMask);
}

void RLELogic::seek(SizeType digitSkip, Offset &offset) const {
  assert(isValid() && "cannot seek on invalid value");
  const auto *ptr = getCodePointer();
  assert(offset.bytes < byteCount && "byte offset exceeds bounds");
  assert(((offset.bytes == byteCount - 1) ||
          (offset.runLength < getRunLength(ptr[offset.bytes]))) &&
         "run-length offset exceeds bounds");

  while ((offset.bytes < byteCount - 1) && (digitSkip > 0)) {
    // Fast-forward over the encoded bytes
    auto runLengthDiff = getRunLength(ptr[offset.bytes]) - offset.runLength;
    if (runLengthDiff > digitSkip) {
      offset.runLength += digitSkip;
      return;
    }
    offset.runLength = 0;
    offset.bytes++;
    digitSkip -= runLengthDiff;
  }

  if (offset.bytes == byteCount - 1) {
    // We ran into the most significant encoded digit
    offset.runLength += digitSkip;
  }
}
