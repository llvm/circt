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
  /// The current tail of the consuming chain.
  Value cursor;
  /// Accumulated (unwrapped destination expression, scanned value, matched)
  /// tuples to be assigned with a moore.blocking_assign by the caller.
  SmallVector<std::tuple<const slang::ast::Expression *, Value, Value>>
      assignments;

  ScanStringParser(Context &context, Value initialCursor,
                   ArrayRef<const slang::ast::Expression *> destinations,
                   Location loc)
      : context(context), builder(context.builder), destinations(destinations),
        loc(loc), cursor(initialCursor) {}

  /// Thread the consuming chain and collect (destination, value, matched)
  /// assignments tuples.
  FailureOr<Context::ScanStringResult> parse(StringRef formatStr) {
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

    return Context::ScanStringResult{cursor, std::move(assignments)};
  }

  /// Emit a literal text fragment that must match verbatim in the input.
  void emitLiteral(StringRef text) {
    if (text.empty())
      return;
    cursor =
        moore::ScanLiteralOp::create(builder, loc, cursor, text).getRemaining();
  }

  /// Strip an outer AssignmentExpression wrapper if present,
  /// returning the actual lvalue subexpression.
  static const slang::ast::Expression *
  unwrapDest(const slang::ast::Expression *expr) {
    if (auto *assign = expr->as_if<slang::ast::AssignmentExpression>())
      return &assign->left();
    return expr;
  }

  /// Consume the next destination argument (if not suppressed) and emit a
  /// scan fragment op.
  LogicalResult emitScanArg(char specifier, bool suppress,
                            std::optional<uint32_t> maxWidth) {
    auto specifierLower = std::tolower(specifier);

    // Hierarchical path specifier (%m), no argument consumed.
    if (specifierLower == 'm') {
      cursor = moore::ScanHierPathMatchOp::create(builder, loc, cursor)
                   .getRemaining();
      return success();
    }

    IntegerAttr maxWidthAttr = nullptr;
    if (maxWidth)
      maxWidthAttr = builder.getI32IntegerAttr(*maxWidth);

    const slang::ast::Expression *destExpr = nullptr;
    if (!suppress) {
      if (destinations.empty())
        return mlir::emitError(loc)
               << "too few arguments for scan format specifier '%" << specifier
               << "'";
      destExpr = unwrapDest(destinations.front());
      destinations = destinations.drop_front();
    }

    switch (specifierLower) {
    // Integer specifiers (%b, %o, %d, %h, %x)
    case 'b':
      return emitScanInt(destExpr, suppress, IntFormat::Binary, maxWidthAttr);
    case 'o':
      return emitScanInt(destExpr, suppress, IntFormat::Octal, maxWidthAttr);
    case 'd':
      return emitScanInt(destExpr, suppress, IntFormat::Decimal, maxWidthAttr);
    case 'h':
    case 'x':
      return emitScanInt(destExpr, suppress,
                         std::isupper(specifier) ? IntFormat::HexUpper
                                                 : IntFormat::HexLower,
                         maxWidthAttr);

    // Floating-point specifiers (%f, %e, %g)
    case 'f':
    case 'e':
    case 'g':
      return emitScanReal(destExpr, suppress, maxWidthAttr);

    // Time specifier (%t)
    case 't':
      return emitScanTime(destExpr, suppress, maxWidthAttr);

    // Character specifier (%c)
    case 'c':
      return emitScanChar(destExpr, suppress);

    // String specifier (%s)
    case 's':
      return emitScanStr(destExpr, suppress, maxWidthAttr);

    // Unformatted binary (%u, %z)
    case 'u':
      return emitScanUnformatted(destExpr, suppress, /*fourValue=*/false);
    case 'z':
      return emitScanUnformatted(destExpr, suppress, /*fourValue=*/true);

    default:
      return mlir::emitError(loc)
             << "unknown scan format specifier '%" << specifier << "'";
    }
  }

  LogicalResult emitScanInt(const slang::ast::Expression *destExpr,
                            bool suppress, IntFormat format,
                            IntegerAttr maxWidth) {
    auto scanStringTy = moore::ScanStringType::get(builder.getContext());

    if (suppress) {
      auto op = moore::ScanIntOp::create(builder, loc, TypeRange{scanStringTy},
                                         cursor, format, maxWidth);
      cursor = op.getRemaining();
      return success();
    }

    auto mooreIntTy =
        llvm::dyn_cast<moore::IntType>(context.convertType(*destExpr->type));
    if (!mooreIntTy)
      return mlir::emitError(loc)
             << "destination of integer scan specifier must be an integer";
    auto mlirIntTy = builder.getIntegerType(mooreIntTy.getWidth());
    auto matchedTy = moore::IntType::getInt(builder.getContext(), 1);
    auto op = moore::ScanIntOp::create(builder, loc, scanStringTy, mlirIntTy,
                                       matchedTy, cursor, format, maxWidth);
    cursor = op.getRemaining();
    auto mooreVal = moore::FromBuiltinIntOp::create(builder, loc, op.getValue())
                        .getResult();
    if (mooreIntTy.getDomain() == moore::Domain::FourValued)
      mooreVal = moore::IntToLogicOp::create(builder, loc, mooreIntTy, mooreVal)
                     .getResult();
    assignments.push_back({destExpr, mooreVal, op.getMatched()});
    return success();
  }

  LogicalResult emitScanReal(const slang::ast::Expression *destExpr,
                             bool suppress, IntegerAttr maxWidth) {
    auto scanStringTy = moore::ScanStringType::get(builder.getContext());
    if (suppress) {
      auto op = moore::ScanRealOp::create(builder, loc, TypeRange{scanStringTy},
                                          cursor, maxWidth);
      cursor = op.getRemaining();
      return success();
    }

    auto valueTy =
        llvm::dyn_cast<moore::RealType>(context.convertType(*destExpr->type));
    if (!valueTy)
      return mlir::emitError(loc)
             << "destination of real scan specifier must be a real";
    auto matchedTy = moore::IntType::getInt(builder.getContext(), 1);
    auto op = moore::ScanRealOp::create(
        builder, loc, TypeRange{scanStringTy, valueTy, matchedTy}, cursor,
        maxWidth);
    cursor = op.getRemaining();
    assignments.push_back({destExpr, op.getValue(), op.getMatched()});
    return success();
  }

  LogicalResult emitScanTime(const slang::ast::Expression *destExpr,
                             bool suppress, IntegerAttr maxWidth) {
    auto scanStringTy = moore::ScanStringType::get(builder.getContext());
    if (suppress) {
      auto op = moore::ScanTimeOp::create(builder, loc, TypeRange{scanStringTy},
                                          cursor, maxWidth);
      cursor = op.getRemaining();
      return success();
    }
    auto valueTy = moore::TimeType::get(builder.getContext());
    auto matchedTy = moore::IntType::getInt(builder.getContext(), 1);
    auto op = moore::ScanTimeOp::create(
        builder, loc, TypeRange{scanStringTy, valueTy, matchedTy}, cursor,
        maxWidth);
    cursor = op.getRemaining();
    assignments.push_back({destExpr, op.getValue(), op.getMatched()});
    return success();
  }

  LogicalResult emitScanChar(const slang::ast::Expression *destExpr,
                             bool suppress) {
    auto scanStringTy = moore::ScanStringType::get(builder.getContext());
    if (suppress) {
      auto op = moore::ScanCharOp::create(builder, loc, TypeRange{scanStringTy},
                                          cursor);
      cursor = op.getRemaining();
      return success();
    }
    auto mooreIntTy =
        llvm::dyn_cast<moore::IntType>(context.convertType(*destExpr->type));
    if (!mooreIntTy)
      return mlir::emitError(loc)
             << "destination of char scan specifier must be an integer";
    auto mlirIntTy = builder.getIntegerType(mooreIntTy.getWidth());
    auto matchedTy = moore::IntType::getInt(builder.getContext(), 1);
    auto op = moore::ScanCharOp::create(
        builder, loc, TypeRange{scanStringTy, mlirIntTy, matchedTy}, cursor);
    cursor = op.getRemaining();
    auto mooreVal = moore::FromBuiltinIntOp::create(builder, loc, op.getValue())
                        .getResult();
    if (mooreIntTy.getDomain() == moore::Domain::FourValued)
      mooreVal = moore::IntToLogicOp::create(builder, loc, mooreIntTy, mooreVal)
                     .getResult();
    assignments.push_back({destExpr, mooreVal, op.getMatched()});
    return success();
  }

  LogicalResult emitScanStr(const slang::ast::Expression *destExpr,
                            bool suppress, IntegerAttr maxWidth) {
    auto scanStringTy = moore::ScanStringType::get(builder.getContext());
    if (suppress) {
      auto op = moore::ScanStrOp::create(builder, loc, TypeRange{scanStringTy},
                                         cursor, maxWidth);
      cursor = op.getRemaining();
      return success();
    }
    auto stringTy = moore::StringType::get(builder.getContext());
    auto matchedTy = moore::IntType::getInt(builder.getContext(), 1);
    auto op = moore::ScanStrOp::create(
        builder, loc, TypeRange{scanStringTy, stringTy, matchedTy}, cursor,
        maxWidth);
    cursor = op.getRemaining();
    Value val = op.getValue();
    auto destTy = context.convertType(*destExpr->type);
    if (auto intTy = llvm::dyn_cast<moore::IntType>(destTy)) {
      val =
          moore::StringToIntOp::create(builder, loc, intTy.getTwoValued(), val)
              .getResult();
      if (intTy.getDomain() == moore::Domain::FourValued)
        val = moore::IntToLogicOp::create(builder, loc, intTy, val).getResult();
    } else if (!llvm::isa<moore::StringType>(destTy)) {
      return mlir::emitError(loc)
             << "destination of string scan specifier must be a string or "
                "integer type";
    }
    assignments.push_back({destExpr, val, op.getMatched()});
    return success();
  }

  LogicalResult emitScanUnformatted(const slang::ast::Expression *destExpr,
                                    bool suppress, bool fourValue) {
    auto scanStringTy = moore::ScanStringType::get(builder.getContext());

    if (suppress) {
      auto op = moore::ScanUnformattedOp::create(
          builder, loc, TypeRange{scanStringTy}, cursor, fourValue);
      cursor = op.getRemaining();
      return success();
    }
    auto mooreIntTy =
        llvm::dyn_cast<moore::IntType>(context.convertType(*destExpr->type));
    if (!mooreIntTy)
      return mlir::emitError(loc)
             << "destination of unformatted scan specifier must be an integer";

    auto mlirIntTy = builder.getIntegerType(mooreIntTy.getWidth());
    auto matchedTy = moore::IntType::getInt(builder.getContext(), 1);
    auto op = moore::ScanUnformattedOp::create(
        builder, loc, TypeRange{scanStringTy, mlirIntTy, matchedTy}, cursor,
        fourValue);
    cursor = op.getRemaining();
    auto mooreVal = moore::FromBuiltinIntOp::create(builder, loc, op.getValue())
                        .getResult();
    if (mooreIntTy.getDomain() == moore::Domain::FourValued)
      mooreVal = moore::IntToLogicOp::create(builder, loc, mooreIntTy, mooreVal)
                     .getResult();
    assignments.push_back({destExpr, mooreVal, op.getMatched()});
    return success();
  }
};

} // namespace

FailureOr<Context::ScanStringResult> Context::convertScanString(
    StringRef formatStr, Value initialCursor,
    std::span<const slang::ast::Expression *const> destinations, Location loc) {
  ScanStringParser parser(*this, initialCursor,
                          ArrayRef(destinations.data(), destinations.size()),
                          loc);
  return parser.parse(formatStr);
}
