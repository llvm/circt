//===- ScanStrings.cpp - Scan format string conversion --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"

using namespace mlir;
using namespace circt;
using namespace ImportVerilog;
using moore::IntFormat;

namespace {

//===----------------------------------------------------------------------===//
// Scan format string parser
//===----------------------------------------------------------------------===//

static bool parseScanFormat(StringRef format,
                            function_ref<void(StringRef text)> onText,
                            function_ref<void(char specifier, bool suppress,
                                              std::optional<uint32_t> maxWidth)>
                                onArg,
                            function_ref<void(StringRef msg)> onError) {

  size_t i = 0;
  size_t textStart = 0;

  while (i < format.size()) {
    if (format[i] != '%') {
      ++i;
      continue;
    }

    if (i > textStart)
      onText(format.substr(textStart, i - textStart));

    ++i; // consume '%'
    if (i >= format.size()) {
      onError("unexpected end of format string after '%'");
      return false;
    }

    if (format[i] == '%') {
      onText("%");
      ++i;
      textStart = i;
      continue;
    }

    // Optional assignment suppression '*'.
    bool suppress = false;
    if (format[i] == '*') {
      suppress = true;
      ++i;
      if (i >= format.size()) {
        onError("unexpected end of format string after '%*'");
        return false;
      }
    }

    // Optional maximum field width.
    std::optional<uint32_t> maxWidth;
    if (std::isdigit(format[i])) {
      uint32_t w = 0;
      while (i < format.size() && std::isdigit(format[i]))
        w = w * 10 + (format[i++] - '0');
      maxWidth = w;
      if (w == 0) {
        onError("field width must be at least 1");
        return false;
      }
    }

    if (i >= format.size()) {
      onError("unexpected end of format string: missing conversion specifier");
      return false;
    }

    char specifier = format[i++];
    onArg(specifier, suppress, maxWidth);
    textStart = i;
  }

  // Flush any trailing literal text.
  if (textStart < format.size())
    onText(format.substr(textStart));

  return true;
}

struct ScanStringParser {
  Context &context;
  OpBuilder &builder;
  /// Remaining destination arguments (lvalue expressions to write into).
  /// Suppressed assignments do not consume from this list.
  ArrayRef<const slang::ast::Expression *> destinations;
  /// Current location for ops and diagnostics.
  Location loc;
  /// Accumulated scan format fragments.
  SmallVector<Value> fragments;

  ScanStringParser(Context &context,
                   ArrayRef<const slang::ast::Expression *> destinations,
                   Location loc)
      : context(context), builder(context.builder), destinations(destinations),
        loc(loc) {}

  /// Parse the format string and produce a ScanStringType value combining
  /// all fragments. Returns failure if any error occurs.
  FailureOr<Value> parse(StringRef formatStr) {
    bool anyFailure = false;

    auto onText = [&](StringRef text) {
      if (!anyFailure)
        emitLiteral(text);
    };

    auto onArg = [&](char specifier, bool suppress,
                     std::optional<uint32_t> maxWidth) {
      if (anyFailure)
        return;
      if (failed(emitScanArg(specifier, suppress, maxWidth)))
        anyFailure = true;
    };

    auto onError = [&](StringRef msg) {
      if (!anyFailure) {
        mlir::emitError(loc) << "scan format string error: " << msg;
        anyFailure = true;
      }
    };

    if (!parseScanFormat(formatStr, onText, onArg, onError) || anyFailure)
      return failure();

    if (fragments.empty())
      return Value{};
    if (fragments.size() == 1)
      return fragments[0];
    return moore::ScanConcatOp::create(builder, loc, fragments).getResult();
  }

  /// Emit a literal text fragment that must match verbatim in the input.
  void emitLiteral(StringRef text) {
    if (text.empty())
      return;
    fragments.push_back(moore::ScanLiteralOp::create(builder, loc, text));
  }

  /// Resolve a destination expression to an lvalue reference value.
  FailureOr<Value> resolveDest(const slang::ast::Expression &destExpr) {
    const auto *expr = &destExpr;
    if (auto assign = expr->as_if<slang::ast::AssignmentExpression>())
      expr = &assign->left();
    auto lhs = context.convertLvalueExpression(*expr);
    if (!lhs)
      return failure();
    return lhs;
  }

  /// Consume the next destination argument (if not suppressed) and emit a
  /// scan fragment op.
  LogicalResult emitScanArg(char specifier, bool suppress,
                            std::optional<uint32_t> maxWidth) {
    auto specifierLower = std::tolower(specifier);

    // Hierarchical path specifier (%m), no argument consumed.
    if (specifierLower == 'm') {
      fragments.push_back(moore::ScanHierPathMatchOp::create(builder, loc));
      return success();
    }

    // All other specifiers consume a destination argument (if not suppressed).
    Value dest;
    if (!suppress) {
      if (destinations.empty())
        return mlir::emitError(loc)
               << "too few arguments for scan format specifier '%" << specifier
               << "'";
      auto destOrFailure = resolveDest(*destinations.front());
      destinations = destinations.drop_front();
      if (failed(destOrFailure))
        return failure();
      dest = *destOrFailure;
    }

    IntegerAttr maxWidthAttr = nullptr;
    if (maxWidth)
      maxWidthAttr = builder.getI32IntegerAttr(*maxWidth);

    switch (specifierLower) {
    // Integer specifiers (%b, %o, %d, %h, %x)
    case 'b':
      return emitScanInt(dest, IntFormat::Binary, maxWidthAttr);
    case 'o':
      return emitScanInt(dest, IntFormat::Octal, maxWidthAttr);
    case 'd':
      return emitScanInt(dest, IntFormat::Decimal, maxWidthAttr);
    case 'h':
    case 'x':
      return emitScanInt(dest,
                         std::isupper(specifier) ? IntFormat::HexUpper
                                                 : IntFormat::HexLower,
                         maxWidthAttr);

    // Floating-point specifiers (%f, %e, %g)
    case 'f':
    case 'e':
    case 'g':
      return emitScanReal(dest, maxWidthAttr);

    // Time specifier (%t)
    case 't':
      return emitScanTime(dest, maxWidthAttr);

    // Character specifier (%c)
    case 'c':
      return emitScanChar(dest);

    // String specifier (%s)
    case 's':
      return emitScanStr(dest, maxWidthAttr);

    // Unformatted binary (%u, %z)
    case 'u':
      return emitScanUnformatted(dest, /*fourValue=*/false);
    case 'z':
      return emitScanUnformatted(dest, /*fourValue=*/true);

    default:
      return mlir::emitError(loc)
             << "unknown scan format specifier '%" << specifier << "'";
    }
  }

  LogicalResult emitScanInt(Value dest, IntFormat format,
                            IntegerAttr maxWidth) {
    fragments.push_back(
        moore::ScanIntOp::create(builder, loc, dest, format, maxWidth));
    return success();
  }

  LogicalResult emitScanReal(Value dest, IntegerAttr maxWidth) {
    fragments.push_back(
        moore::ScanRealOp::create(builder, loc, dest, maxWidth));
    return success();
  }

  LogicalResult emitScanTime(Value dest, IntegerAttr maxWidth) {
    fragments.push_back(
        moore::ScanTimeOp::create(builder, loc, dest, maxWidth));
    return success();
  }

  LogicalResult emitScanChar(Value dest) {
    fragments.push_back(moore::ScanCharOp::create(builder, loc, dest));
    return success();
  }

  LogicalResult emitScanStr(Value dest, IntegerAttr maxWidth) {
    fragments.push_back(moore::ScanStrOp::create(builder, loc, dest, maxWidth));
    return success();
  }

  LogicalResult emitScanUnformatted(Value dest, bool fourValue) {
    fragments.push_back(
        moore::ScanUnformattedOp::create(builder, loc, dest, fourValue));
    return success();
  }
};

} // namespace

FailureOr<Value> Context::convertScanString(
    StringRef formatStr,
    std::span<const slang::ast::Expression *const> destinations, Location loc) {
  ScanStringParser parser(
      *this, ArrayRef(destinations.data(), destinations.size()), loc);
  return parser.parse(formatStr);
}
