//===- HWLogicDigits.h - Primitives for 9-valued logic ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the binary encoding for multi-valued logic primitives and
// operations. The set of valid digits is equivalent to the 9-valued logic
// of the IEEE 1164 standard. However, their specific semantics may differ
// depending on the context.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_LOGIC_DIGITS_H
#define CIRCT_DIALECT_HW_LOGIC_DIGITS_H

#include <stdint.h>

namespace circt {
namespace hw {
namespace logicdigits {

/// Logic digit primitives
enum LogicDigit : uint8_t {
  Invalid = 0, // Guard value, not an actual digit
  LD_U = 1,
  LD_X = 2,
  LD_0 = 3,
  LD_1 = 4,
  LD_Z = 5,
  LD_W = 6,
  LD_L = 7,
  LD_H = 8,
  LD_DC = 9 // Don't-care '-'
};

/// LUT type for unary logic operations
typedef LogicDigit UnaryLogicLUT[10];
/// LUT type for binary logic operations
typedef UnaryLogicLUT BinaryLogicLUT[10];

/// Returns true if the given digit is a valid 9-valued logic digit
constexpr bool isValidLogicDigit(LogicDigit digit) {
  return (digit != LogicDigit::Invalid) && (digit <= LogicDigit::LD_DC);
}

/// Convert a (uppercase) character to its matching 9-valued logic digit
static constexpr LogicDigit charToLogicDigit(char c) {
  switch (c) {
  case 'U':
    return LogicDigit::LD_U;
  case 'X':
    return LogicDigit::LD_X;
  case '0':
    return LogicDigit::LD_0;
  case '1':
    return LogicDigit::LD_1;
  case 'Z':
    return LogicDigit::LD_Z;
  case 'W':
    return LogicDigit::LD_W;
  case 'L':
    return LogicDigit::LD_L;
  case 'H':
    return LogicDigit::LD_H;
  case '-':
    return LogicDigit::LD_DC;
  default:
    return LogicDigit::Invalid;
  }
};

/// Convert a binary-encoded 9-valued logic digit to a human readable character
static constexpr char logicDigitToChar(LogicDigit ld) {
  switch (ld) {
  case Invalid:
    return '#';
  case LD_U:
    return 'U';
  case LD_X:
    return 'X';
  case LD_0:
    return '0';
  case LD_1:
    return '1';
  case LD_Z:
    return 'Z';
  case LD_W:
    return 'W';
  case LD_L:
    return 'L';
  case LD_H:
    return 'H';
  case LD_DC:
    return '-';
  default:
    return '?';
  }
}

// --------------
// IEEE1164 LUTs
// --------------

/// IEEE1164 XOR operation
static constexpr BinaryLogicLUT LUT_IEEE1164_XOR = {
    {Invalid, Invalid, Invalid, Invalid, Invalid, Invalid, Invalid, Invalid,
     Invalid, Invalid},
    {Invalid, LD_U, LD_U, LD_U, LD_U, LD_U, LD_U, LD_U, LD_U, LD_U},
    {Invalid, LD_U, LD_X, LD_X, LD_X, LD_X, LD_X, LD_X, LD_X, LD_X},
    {Invalid, LD_U, LD_X, LD_0, LD_1, LD_X, LD_X, LD_0, LD_1, LD_X},
    {Invalid, LD_U, LD_X, LD_1, LD_0, LD_X, LD_X, LD_1, LD_0, LD_X},
    {Invalid, LD_U, LD_X, LD_X, LD_X, LD_X, LD_X, LD_X, LD_X, LD_X},
    {Invalid, LD_U, LD_X, LD_X, LD_X, LD_X, LD_X, LD_X, LD_X, LD_X},
    {Invalid, LD_U, LD_X, LD_0, LD_1, LD_X, LD_X, LD_0, LD_1, LD_X},
    {Invalid, LD_U, LD_X, LD_1, LD_0, LD_X, LD_X, LD_1, LD_0, LD_X},
    {Invalid, LD_U, LD_X, LD_X, LD_X, LD_X, LD_X, LD_X, LD_X, LD_X}};

// TODO: Add other operations

} // namespace logicdigits
} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_LOGIC_DIGITS_H
