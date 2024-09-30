//===- FormatStrings.cpp - Verilog format string conversion ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/text/SFormat.h"

using namespace mlir;
using namespace circt;
using namespace ImportVerilog;
using moore::IntAlign;
using moore::IntFormat;
using moore::IntPadding;
using slang::SFormat::FormatOptions;

namespace {
struct FormatStringParser {
  Context &context;
  OpBuilder &builder;
  /// The remaining arguments to be parsed.
  ArrayRef<const slang::ast::Expression *> arguments;
  /// The current location to use for ops and diagnostics.
  Location loc;
  /// The default format for integer arguments not covered by a format string
  /// literal.
  IntFormat defaultFormat;
  /// The interpolated string fragments that will be concatenated using a
  /// `moore.fmt.concat` op.
  SmallVector<Value> fragments;

  FormatStringParser(Context &context,
                     ArrayRef<const slang::ast::Expression *> arguments,
                     Location loc, IntFormat defaultFormat)
      : context(context), builder(context.builder), arguments(arguments),
        loc(loc), defaultFormat(defaultFormat) {}

  /// Entry point to the format string parser.
  FailureOr<Value> parse(bool appendNewline) {
    while (!arguments.empty()) {
      const auto &arg = *arguments[0];
      arguments = arguments.drop_front();
      if (arg.kind == slang::ast::ExpressionKind::EmptyArgument)
        continue;
      loc = context.convertLocation(arg.sourceRange);
      if (auto *lit = arg.as_if<slang::ast::StringLiteral>()) {
        if (failed(parseFormat(lit->getValue())))
          return failure();
      } else {
        if (failed(emitDefault(arg)))
          return failure();
      }
    }

    // Append the optional newline.
    if (appendNewline)
      emitLiteral("\n");

    // Concatenate all string fragments into one formatted string, or return an
    // empty literal if no fragments were generated.
    if (fragments.empty())
      return Value{};
    if (fragments.size() == 1)
      return fragments[0];
    return builder.create<moore::FormatConcatOp>(loc, fragments).getResult();
  }

  /// Parse a format string literal and consume and format the arguments
  /// corresponding to the format specifiers it contains.
  LogicalResult parseFormat(StringRef format) {
    bool anyFailure = false;
    auto onText = [&](auto text) {
      if (anyFailure)
        return;
      emitLiteral(text);
    };
    auto onArg = [&](auto specifier, auto offset, auto len,
                     const auto &options) {
      if (anyFailure)
        return;
      if (failed(emitArgument(specifier, format.substr(offset, len), options)))
        anyFailure = true;
    };
    auto onError = [&](auto, auto, auto, auto) {
      assert(false && "Slang should have already reported all errors");
    };
    slang::SFormat::parse(format, onText, onArg, onError);
    return failure(anyFailure);
  }

  /// Emit a string literal that requires no additional formatting.
  void emitLiteral(StringRef literal) {
    fragments.push_back(builder.create<moore::FormatLiteralOp>(loc, literal));
  }

  /// Consume the next argument from the list and emit it according to the given
  /// format specifier.
  LogicalResult emitArgument(char specifier, StringRef fullSpecifier,
                             const FormatOptions &options) {
    auto specifierLower = std::tolower(specifier);

    // Special handling for format specifiers that consume no argument.
    if (specifierLower == 'm' || specifierLower == 'l')
      return mlir::emitError(loc)
             << "unsupported format specifier `" << fullSpecifier << "`";

    // Consume the next argument, which will provide the value to be
    // formatted.
    assert(!arguments.empty() && "Slang guarantees correct arg count");
    const auto &arg = *arguments[0];
    arguments = arguments.drop_front();
    auto argLoc = context.convertLocation(arg.sourceRange);

    // Handle the different formatting options.
    // See IEEE 1800-2017 § 21.2.1.2 "Format specifications".
    switch (specifierLower) {
    case 'b':
      return emitInteger(arg, options, IntFormat::Binary);
    case 'o':
      return emitInteger(arg, options, IntFormat::Octal);
    case 'd':
      return emitInteger(arg, options, IntFormat::Decimal);
    case 'h':
    case 'x':
      return emitInteger(arg, options,
                         std::isupper(specifier) ? IntFormat::HexUpper
                                                 : IntFormat::HexLower);

    case 's':
      // Simplified handling for literals.
      if (auto *lit = arg.as_if<slang::ast::StringLiteral>()) {
        if (options.width)
          return mlir::emitError(loc)
                 << "string format specifier with width not supported";
        emitLiteral(lit->getValue());
        return success();
      }
      return mlir::emitError(argLoc)
             << "expression cannot be formatted as string";

    default:
      return mlir::emitError(loc)
             << "unsupported format specifier `" << fullSpecifier << "`";
    }
  }

  /// Emit an integer value with the given format.
  LogicalResult emitInteger(const slang::ast::Expression &arg,
                            const FormatOptions &options, IntFormat format) {
    auto value =
        context.convertToSimpleBitVector(context.convertRvalueExpression(arg));
    if (!value)
      return failure();

    // Determine the width to which the formatted integer should be padded.
    unsigned width;
    if (options.width) {
      width = *options.width;
    } else {
      width = cast<moore::IntType>(value.getType()).getWidth();
      if (format == IntFormat::Octal)
        // 3 bits per octal digit
        width = (width + 2) / 3;
      else if (format == IntFormat::HexLower || format == IntFormat::HexUpper)
        // 4 bits per hex digit
        width = (width + 3) / 4;
      else if (format == IntFormat::Decimal)
        // ca. 3.322 bits per decimal digit (ln(10)/ln(2))
        width = std::ceil(width * std::log(2) / std::log(10));
    }

    // Determine the alignment and padding.
    auto alignment = options.leftJustify ? IntAlign::Left : IntAlign::Right;
    auto padding =
        format == IntFormat::Decimal ? IntPadding::Space : IntPadding::Zero;

    fragments.push_back(builder.create<moore::FormatIntOp>(
        loc, value, format, width, alignment, padding));
    return success();
  }

  /// Emit an expression argument with the appropriate default formatting.
  LogicalResult emitDefault(const slang::ast::Expression &expr) {
    FormatOptions options;
    return emitInteger(expr, options, defaultFormat);
  }
};
} // namespace

FailureOr<Value> Context::convertFormatString(
    slang::span<const slang::ast::Expression *const> arguments, Location loc,
    IntFormat defaultFormat, bool appendNewline) {
  FormatStringParser parser(*this, ArrayRef(arguments.data(), arguments.size()),
                            loc, defaultFormat);
  return parser.parse(appendNewline);
}
