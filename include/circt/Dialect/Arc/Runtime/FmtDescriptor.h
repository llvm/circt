//===- FmtDescriptor.h - Format descriptor for the ArcRuntime ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares FmtDescriptor, used by arcRuntimeFormat.
//
// This struct is created during compilation and serialized into the generated
// LLVM IR. It is treated as opaque by the generated LLVM IR, and therefore can
// use implementation-defined layout and padding if needed, as long as the
// compiler used during compilation is that same as that used when compiling
// the runtime.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_RUNTIME_FMTDESCRIPTOR_H
#define CIRCT_DIALECT_ARC_RUNTIME_FMTDESCRIPTOR_H

#include <cassert>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>

namespace circt {
namespace arc {
namespace runtime {

/// A format descriptor, to be given to arcRuntimeFormat.
///
/// arcRuntimeFormat takes an array of FmtDescriptor and a variadic argument
/// list. Each FmtDescriptor describes how to format the corresponding
/// argument. The array is terminated by a FmtDescriptor with action Action_End.
struct FmtDescriptor {
  /// Default construction creates an end of string descriptor.
  FmtDescriptor() {
    std::memset(this, 0, sizeof(*this));
    action = Action_End;
  }

  /// Creates a literal string descriptor.
  ///
  /// width: The width of the literal string in characters.
  ///
  /// The string itself will be passed as a variadic argument (const char*).
  static FmtDescriptor createLiteral(int64_t width) {
    FmtDescriptor d;
    d.action = Action_Literal;
    d.literal.width = width;
    return d;
  }

  /// Creates a small literal string descriptor.
  static FmtDescriptor createSmallLiteral(std::string_view str) {
    assert(str.size() < sizeof(FmtDescriptor::smallLiteral.data));
    FmtDescriptor d;
    d.action = Action_LiteralSmall;
    std::strncpy(d.smallLiteral.data, str.data(),
                 sizeof(d.smallLiteral.data) - 1);
    d.smallLiteral.data[sizeof(d.smallLiteral.data) - 1] = '\0';
    return d;
  }

  /// Creates an integer descriptor.
  ///
  /// bitwidth: The bitwidth of the integer value.
  /// radix: The radix to use for formatting. Must be one of {2, 8, 10, 16}.
  /// isLeftAligned: Whether the value is left aligned.
  /// specifierWidth: The minumum width of the output in characters.
  /// isUpperCase: Whether to use uppercase hex letters.
  /// isSigned: Whether to treat the value as signed.
  ///
  /// The integer value will be passed as a variadic argument by *pointer*.
  static FmtDescriptor createInt(int32_t bitwidth, int8_t radix,
                                 bool isLeftAligned, int32_t specifierWidth,
                                 char paddingChar, bool isUpperCase,
                                 bool isSigned) {
    FmtDescriptor d;
    d.action = Action_Int;
    d.intFmt.bitwidth = bitwidth;
    d.intFmt.radix = radix;
    d.intFmt.isLeftAligned = isLeftAligned;
    d.intFmt.specifierWidth = specifierWidth;
    d.intFmt.paddingChar = paddingChar;
    d.intFmt.isUpperCase = isUpperCase;
    d.intFmt.isSigned = isSigned;
    return d;
  }

  /// Creates an integer descriptor with exact minimum-width semantics.
  ///
  /// Unlike `createInt`, this does not apply the natural binary/octal/hex
  /// width of the value type before applying `specifierWidth`.
  static FmtDescriptor createExactInt(int32_t bitwidth, int8_t radix,
                                      bool isLeftAligned,
                                      int32_t specifierWidth, char paddingChar,
                                      bool isUpperCase, bool isSigned) {
    FmtDescriptor d = createInt(bitwidth, radix, isLeftAligned, specifierWidth,
                                paddingChar, isUpperCase, isSigned);
    d.action = Action_IntExact;
    return d;
  }

  /// Creates a four-valued integer descriptor. The value and unknown masks will
  /// be passed as consecutive variadic arguments by pointer.
  static FmtDescriptor createFVInt(int32_t bitwidth, int8_t radix,
                                   bool isLeftAligned, int32_t specifierWidth,
                                   char paddingChar, bool isUpperCase,
                                   bool isSigned) {
    FmtDescriptor d = createInt(bitwidth, radix, isLeftAligned, specifierWidth,
                                paddingChar, isUpperCase, isSigned);
    d.action = Action_FVInt;
    return d;
  }

  /// Creates a char descriptor.
  ///
  /// The character value will be passed as a variadic argument by value.
  static FmtDescriptor createChar(bool isLeftAligned = false,
                                  char paddingChar = ' ',
                                  int32_t specifierWidth = -1) {
    FmtDescriptor d;
    d.action = Action_Char;
    d.stringFmt.specifierWidth = specifierWidth;
    d.stringFmt.paddingChar = paddingChar;
    d.stringFmt.isLeftAligned = isLeftAligned;
    return d;
  }

  /// Creates a time descriptor.
  ///
  /// The time value will be passed as a variadic argument by value in
  /// femtoseconds. A negative widthOverride uses the current `$timeformat`
  /// minimum field width.
  static FmtDescriptor createTime(int32_t widthOverride) {
    FmtDescriptor d;
    d.action = Action_Time;
    d.timeFmt.widthOverride = widthOverride;
    return d;
  }

  /// Creates a dynamic string descriptor.
  ///
  /// The string value will be passed as a variadic argument by pointer to a
  /// NUL-terminated string.
  static FmtDescriptor createString(bool isLeftAligned, char paddingChar,
                                    int32_t specifierWidth) {
    FmtDescriptor d;
    d.action = Action_String;
    d.stringFmt.specifierWidth = specifierWidth;
    d.stringFmt.paddingChar = paddingChar;
    d.stringFmt.isLeftAligned = isLeftAligned;
    return d;
  }

  /// Creates a real descriptor.
  ///
  /// format: The printf-style conversion character to use. Must be one of
  /// {'e', 'f', 'g'}.
  /// isLeftAligned: Whether the value is left aligned.
  /// fieldWidth: The minimum width of the output in characters.
  /// fracDigits: The precision to use for the conversion.
  ///
  /// The real value will be passed as a variadic argument by value as a double.
  static FmtDescriptor createReal(char format, bool isLeftAligned,
                                  int32_t fieldWidth, int32_t fracDigits) {
    FmtDescriptor d;
    d.action = Action_Real;
    d.realFmt.fieldWidth = fieldWidth;
    d.realFmt.fracDigits = fracDigits;
    d.realFmt.format = format;
    d.realFmt.isLeftAligned = isLeftAligned;
    return d;
  }

  /// The action to take for this descriptor.
  ///
  /// We use uint64_t to ensure that the descriptor is always 16 bytes in size
  /// with zero padding.
  enum Action : uint64_t {
    /// End of the format string, no action to take.
    Action_End = 0,
    /// Prints a literal string.
    Action_Literal,
    /// Prints a literal string (small string optimization).
    Action_LiteralSmall,
    /// Prints an integer.
    Action_Int,
    /// Prints an integer with exact minimum-width semantics.
    Action_IntExact,
    /// Prints a four-valued integer.
    Action_FVInt,
    /// Prints a character (%c).
    Action_Char,
    /// Prints a time value (%t).
    Action_Time,
    /// Prints a dynamic string (%s).
    Action_String,
    /// Prints a real (%e, %f, or %g).
    Action_Real,
  };
  Action action;

  /// Integer formatting options.
  struct IntFmt {
    /// The bitwidth of the integer value.
    int16_t bitwidth;
    /// The minumum width of the output in characters.
    int16_t specifierWidth;
    /// The radix to use for formatting. Must be one of {2, 8, 10, 16}.
    int8_t radix;
    /// Padding character (NUL if no padding is desired).
    char paddingChar;
    /// Whether the value is left aligned.
    bool isLeftAligned;
    /// Whether to use uppercase hex letters.
    bool isUpperCase;
    /// Whether to treat the value as signed.
    bool isSigned;
  };

  /// Literal string formatting options.
  struct LiteralFmt {
    /// The width of the literal string in characters. Note that the string
    /// itself is passed as a variadic argument, and may contain NUL characters.
    int64_t width;
  };

  /// Literal string (small string optimization).
  struct SmallLiteral {
    /// NUL-terminated string.
    char data[8];
  };

  /// Time formatting options.
  struct TimeFmt {
    /// Optional minimum field width override, or -1 to use `$timeformat`.
    int32_t widthOverride;
    /// Reserved to keep the descriptor payload at 8 bytes.
    int32_t reserved;
  };

  /// Dynamic string formatting options.
  struct StringFmt {
    /// The minimum width of the output in characters.
    int16_t specifierWidth;
    /// The padding character to use if width requires padding.
    char paddingChar;
    /// Whether the value is left aligned.
    bool isLeftAligned;
  };

  /// Real formatting options.
  struct RealFmt {
    /// The minimum width of the output in characters.
    int16_t fieldWidth;
    /// The number of fractional digits or significant digits to print.
    int16_t fracDigits;
    /// The printf-style conversion character to use: 'e', 'f', or 'g'.
    char format;
    /// Whether the value is left aligned.
    bool isLeftAligned;
  };

  union {
    LiteralFmt literal;
    IntFmt intFmt;
    SmallLiteral smallLiteral;
    TimeFmt timeFmt;
    StringFmt stringFmt;
    RealFmt realFmt;
  };
};

static_assert(std::is_standard_layout_v<FmtDescriptor>,
              "FmtDescriptor must be standard layout");

} // namespace runtime
} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_RUNTIME_FMTDESCRIPTOR_H
