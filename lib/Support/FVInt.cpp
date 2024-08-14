//===- FVInt.cpp - Four-valued integer --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/FVInt.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringExtras.h"

#define DEBUG_TYPE "fvint"

using namespace circt;

std::optional<FVInt> FVInt::tryFromString(StringRef str, unsigned radix) {
  assert(radix == 2 || radix == 8 || radix == 10 || radix == 16);
  if (str.empty())
    return {};

  // Overestimate the number of bits that will be needed to hold all digits.
  unsigned radixLog2 = 0;
  for (unsigned r = radix - 1; r > 0; r >>= 1)
    ++radixLog2;
  bool radixIsPow2 = (radix == (1U << radixLog2));

  // Parse the string.
  auto result = FVInt::getZero(str.size() * radixLog2);
  while (!str.empty()) {
    unsigned digit = llvm::toLower(str[0]);
    str = str.drop_front();

    // Handle X and Z digits.
    if (digit == 'x' || digit == 'z') {
      if (!radixIsPow2)
        return {};
      result <<= radixLog2;
      result.unknown.setLowBits(radixLog2);
      if (digit == 'z')
        result.value.setLowBits(radixLog2);
      continue;
    }

    // Determine the value of the current digit.
    if (digit >= '0' && digit <= '9')
      digit = digit - '0';
    else if (digit >= 'a' && digit <= 'z')
      digit = digit - 'a' + 10;
    else
      return {};
    if (digit >= radix)
      return {};

    // Add the digit to the result.
    if (radixIsPow2) {
      result <<= radixLog2;
      result.value |= digit;
    } else {
      result.value *= radix;
      result.value += digit;
    }
  }

  return result;
}

bool FVInt::tryToString(SmallVectorImpl<char> &str, unsigned radix,
                        bool uppercase) const {
  size_t strBaseLen = str.size();
  assert(radix == 2 || radix == 8 || radix == 10 || radix == 16);

  // Determine if the radix is a power of two.
  unsigned radixLog2 = 0;
  for (unsigned r = radix - 1; r > 0; r >>= 1)
    ++radixLog2;
  bool radixIsPow2 = (radix == (1U << radixLog2));
  unsigned radixMask = (1U << radixLog2) - 1;

  // If the number has no X or Z bits, take the easy route and print the `APInt`
  // directly.
  if (!hasUnknown()) {
    value.toString(str, radix, /*Signed=*/false, /*formatAsCLiteral=*/false,
                   uppercase);
    return true;
  }

  // We can only print with non-power-of-two radices if there are no X or Z
  // bits. So at this point we require radix be a power of two.
  if (!radixIsPow2)
    return false;

  // Otherwise chop off digits at the bottom and print them to the string. This
  // prints the digits in reverse order, with the least significant digit as the
  // first character.
  APInt value = this->value;
  APInt unknown = this->unknown;

  char chrA = uppercase ? 'A' : 'a';
  char chrX = uppercase ? 'X' : 'x';
  char chrZ = uppercase ? 'Z' : 'z';

  while (!value.isZero() || !unknown.isZero()) {
    unsigned digitValue = value.getRawData()[0] & radixMask;
    unsigned digitUnknown = unknown.getRawData()[0] & radixMask;
    unsigned shiftAmount = std::min(radixLog2, getBitWidth());
    value.lshrInPlace(shiftAmount);
    unknown.lshrInPlace(shiftAmount);

    // Handle unknown bits. Since we only get to print a single X or Z character
    // to the string, either all bits in the digit have to be X, or all have to
    // be Z. But we cannot represent the case where X, Z and 0/1 bits are mixed.
    if (digitUnknown != 0) {
      if (digitUnknown != radixMask ||
          (digitValue != 0 && digitValue != radixMask)) {
        str.resize(strBaseLen);
        return false;
      }
      str.push_back(digitValue == 0 ? chrX : chrZ);
      continue;
    }

    // Handle known bits.
    if (digitValue < 10)
      str.push_back(digitValue + '0');
    else
      str.push_back(digitValue - 10 + chrA);
  }

  // Reverse the digits.
  std::reverse(str.begin() + strBaseLen, str.end());
  return true;
}

void FVInt::print(raw_ostream &os) const {
  SmallString<32> buffer;
  if (!tryToString(buffer))
    if (!tryToString(buffer, 16))
      tryToString(buffer, 2);
  os << buffer;
}

llvm::hash_code circt::hash_value(const FVInt &a) {
  return llvm::hash_combine(a.getRawValue(), a.getRawUnknown());
}

void circt::printFVInt(AsmPrinter &p, const FVInt &value) {
  SmallString<32> buffer;
  if (value.getBitWidth() > 1 && value.isNegative() &&
      (-value).tryToString(buffer)) {
    p << "-" << buffer;
  } else if (value.tryToString(buffer)) {
    p << buffer;
  } else if (value.tryToString(buffer, 16)) {
    p << "h" << buffer;
  } else {
    value.tryToString(buffer, 2);
    p << "b" << buffer;
  }
}

ParseResult circt::parseFVInt(AsmParser &p, FVInt &result) {
  // Parse the value as either a keyword (`b[01XZ]+` for binary or
  // `h[0-9A-FXZ]+` for hexadecimal), or an integer value (for decimal).
  FVInt value;
  StringRef strValue;
  auto valueLoc = p.getCurrentLocation();
  if (succeeded(p.parseOptionalKeyword(&strValue))) {
    // Determine the radix based on the `b` or `h` prefix.
    unsigned base = 0;
    if (strValue.consume_front("b")) {
      base = 2;
    } else if (strValue.consume_front("h")) {
      base = 16;
    } else {
      return p.emitError(valueLoc) << "expected `b` or `h` prefix";
    }

    // Parse the value.
    auto parsedValue = FVInt::tryFromString(strValue, base);
    if (!parsedValue) {
      return p.emitError(valueLoc)
             << "expected base-" << base << " four-valued integer";
    }

    // Add a zero bit at the top to ensure the value reads as positive.
    result = parsedValue->zext(parsedValue->getBitWidth() + 1);
  } else {
    APInt intValue;
    if (p.parseInteger(intValue))
      return failure();
    result = std::move(intValue);
  }
  return success();
}
