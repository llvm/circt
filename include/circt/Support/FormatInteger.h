//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Integer formatting shared between the Sim dialect's constant folder and the
// Arc runtime. Keeping it in a header-only utility lets both produce identical
// output without the Sim dialect depending on the Arc runtime or vice versa.
//
// The formatting follows the SystemVerilog `%d`/`%h`/`%o`/`%b` family. By
// default a value is padded to the *natural* width of its type -- the number
// of digits required to print the type's largest value -- so values of the
// same type line up. An explicit field width overrides this natural width; a
// field width of 0 disables padding and prints the value in its minimal
// representation, matching SystemVerilog's `%0d` and friends.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_FORMATINTEGER_H
#define CIRCT_SUPPORT_FORMATINTEGER_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>

namespace circt {

/// Number of decimal digits required to print the largest value of an integer
/// type of the given bit width.
inline unsigned getDecimalDigitWidth(unsigned bits, bool isSigned) {
  if (bits == 0)
    return 1;
  if (bits == 1)
    return isSigned ? 2 : 1;
  if (isSigned)
    bits--;
  // Should be precise up until bits = 13301; log(2) / log(10) + epsilon.
  const double baseConversionFactor = 0.30103;
  unsigned digits = std::ceil(bits * baseConversionFactor);
  return isSigned ? digits + 1 : digits;
}

/// Natural field width of an integer type printed in the given radix: the
/// number of digits required to print the type's largest value. `radix` must be
/// one of {2, 8, 10, 16}.
inline unsigned getNaturalIntegerWidth(unsigned bits, unsigned radix,
                                       bool isSigned) {
  switch (radix) {
  case 2:
    return bits;
  case 8:
    return llvm::divideCeil(bits, 3);
  case 16:
    return llvm::divideCeil(bits, 4);
  default: // radix 10
    return getDecimalDigitWidth(bits, isSigned);
  }
}

/// Format `value` in the given `radix` and emit it to `os`, padded to a field
/// width. The field width defaults to the natural width of the value's type and
/// is overridden by `specifierWidth` when present; a `specifierWidth` of 0
/// therefore disables padding. Right-aligned output (the default) is padded on
/// the leading side with `paddingChar`, matching the SystemVerilog `%d`/`%h`
/// family. Left-aligned output is padded on the trailing side with spaces,
/// matching C `printf`, whose zero-pad flag is ignored under left
/// justification; SystemVerilog's `$display` has no left-aligned form. `radix`
/// must be one of {2, 8, 10, 16}.
inline void formatInteger(llvm::raw_ostream &os, const llvm::APInt &value,
                          unsigned radix, bool isUpperCase, bool isLeftAligned,
                          char paddingChar,
                          std::optional<int32_t> specifierWidth,
                          bool isSigned) {
  llvm::SmallVector<char, 32> digits;
  value.toString(digits, radix, isSigned, /*formatAsCLiteral=*/false,
                 isUpperCase);

  unsigned fieldWidth =
      specifierWidth.has_value() && *specifierWidth >= 0
          ? static_cast<unsigned>(*specifierWidth)
          : getNaturalIntegerWidth(value.getBitWidth(), radix, isSigned);
  unsigned padWidth =
      fieldWidth > digits.size() ? fieldWidth - digits.size() : 0;

  if (isLeftAligned) {
    llvm::SmallVector<char, 32> padding(padWidth, ' ');
    os << digits << padding;
  } else {
    llvm::SmallVector<char, 32> padding(padWidth, paddingChar);
    os << padding << digits;
  }
}

} // namespace circt

#endif // CIRCT_SUPPORT_FORMATINTEGER_H
