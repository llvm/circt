//===- SystemCTypes.cpp - Implement the SystemC types ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemC dialect type system.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SystemC/SystemCTypes.h"
#include "circt/Dialect/SystemC/SystemCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::systemc;

Type systemc::getSignalBaseType(Type type) {
  return TypeSwitch<Type, Type>(type)
      .Case<InputType, OutputType, InOutType, SignalType>(
          [](auto ty) { return ty.getBaseType(); })
      .Default([](auto ty) { return Type(); });
}

namespace circt {
namespace systemc {
namespace detail {
bool operator==(const PortInfo &a, const PortInfo &b) {
  return a.name == b.name && a.type == b.type;
}
// NOLINTNEXTLINE(readability-identifier-naming)
llvm::hash_code hash_value(const PortInfo &pi) {
  return llvm::hash_combine(pi.name, pi.type);
}
} // namespace detail
} // namespace systemc
} // namespace circt

//===----------------------------------------------------------------------===//
// ModuleType
//===----------------------------------------------------------------------===//

/// Parses a systemc::ModuleType of the format:
/// !systemc.module<moduleName(portName1: type1, portName2: type2)>
Type ModuleType::parse(AsmParser &odsParser) {
  if (odsParser.parseLess())
    return Type();

  StringRef moduleName;
  if (odsParser.parseKeyword(&moduleName))
    return Type();

  SmallVector<ModuleType::PortInfo> ports;

  if (odsParser.parseCommaSeparatedList(
          AsmParser::Delimiter::Paren, [&]() -> ParseResult {
            StringRef name;
            PortInfo port;
            if (odsParser.parseKeyword(&name) || odsParser.parseColon() ||
                odsParser.parseType(port.type))
              return failure();

            port.name = StringAttr::get(odsParser.getContext(), name);
            ports.push_back(port);

            return success();
          }))
    return Type();

  if (odsParser.parseGreater())
    return Type();

  return ModuleType::getChecked(
      UnknownLoc::get(odsParser.getContext()), odsParser.getContext(),
      StringAttr::get(odsParser.getContext(), moduleName), ports);
}

/// Prints a systemc::ModuleType in the format:
/// !systemc.module<moduleName(portName1: type1, portName2: type2)>
void ModuleType::print(AsmPrinter &odsPrinter) const {
  odsPrinter << '<' << getModuleName().getValue() << '(';
  llvm::interleaveComma(getPorts(), odsPrinter, [&](auto port) {
    odsPrinter << port.name.getValue() << ": " << port.type;
  });
  odsPrinter << ")>";
}

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/SystemC/SystemCTypes.cpp.inc"

void SystemCDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/SystemC/SystemCTypes.cpp.inc"
      >();
  addTypes<UIntType, IntType, BigIntType, BigUIntType, UIntBaseType,
           IntBaseType, SignedType, UnsignedType, BitVectorBaseType,
           BitVectorType>();
}

namespace circt {
namespace systemc {
namespace detail {

/// Integer Type Storage and Uniquing.
struct IntegerWidthStorage : public TypeStorage {
  IntegerWidthStorage(unsigned width) : width(width) {}

  /// The hash key used for uniquing.
  using KeyTy = unsigned;

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  bool operator==(const KeyTy &key) const { return KeyTy(width) == key; }

  static IntegerWidthStorage *construct(mlir::TypeStorageAllocator &allocator,
                                        KeyTy key) {
    return new (allocator.allocate<IntegerWidthStorage>())
        IntegerWidthStorage(key);
  }

  unsigned width : 30;
};

} // namespace detail
} // namespace systemc
} // namespace circt

IntBaseType IntBaseType::get(MLIRContext *context) {
  return Base::get(context);
}

IntType IntType::get(MLIRContext *context, unsigned width) {
  return Base::get(context, width);
}

unsigned IntType::getWidth() { return getImpl()->width; }

UIntBaseType UIntBaseType::get(MLIRContext *context) {
  return Base::get(context);
}

UIntType UIntType::get(MLIRContext *context, unsigned width) {
  return Base::get(context, width);
}
unsigned UIntType::getWidth() { return getImpl()->width; }

SignedType SignedType::get(MLIRContext *context) { return Base::get(context); }

BigIntType BigIntType::get(MLIRContext *context, unsigned width) {
  return Base::get(context, width);
}
unsigned BigIntType::getWidth() { return getImpl()->width; }

UnsignedType UnsignedType::get(MLIRContext *context) {
  return Base::get(context);
}

BigUIntType BigUIntType::get(MLIRContext *context, unsigned width) {
  return Base::get(context, width);
}

unsigned BigUIntType::getWidth() { return getImpl()->width; }

BitVectorBaseType BitVectorBaseType::get(MLIRContext *context) {
  return Base::get(context);
}

BitVectorType BitVectorType::get(MLIRContext *context, unsigned width) {
  return Base::get(context, width);
}

unsigned BitVectorType::getWidth() { return getImpl()->width; }

LogicVectorBaseType LogicVectorBaseType::get(MLIRContext *context) {
  return Base::get(context);
}

LogicVectorType LogicVectorType::get(MLIRContext *context, unsigned width) {
  return Base::get(context, width);
}

unsigned LogicVectorType::getWidth() { return getImpl()->width; }

static mlir::OptionalParseResult customTypeParser(DialectAsmParser &parser,
                                                  StringRef mnemonic,
                                                  llvm::SMLoc loc, Type &type) {
  auto *ctxt = parser.getContext();

  type = llvm::StringSwitch<Type>(mnemonic)
             .Case("int_base", IntBaseType::get(ctxt))
             .Case("uint_base", UIntBaseType::get(ctxt))
             .Case("signed", SignedType::get(ctxt))
             .Case("unsigned", UnsignedType::get(ctxt))
             .Case("bv_base", BitVectorBaseType::get(ctxt))
             .Default(Type());

  if (type)
    return success();

  unsigned width;
  if (parser.parseLess() || parser.parseInteger(width) || parser.parseGreater())
    return failure();

  type = llvm::StringSwitch<Type>(mnemonic)
             .Case("int", IntType::get(ctxt, width))
             .Case("uint", UIntType::get(ctxt, width))
             .Case("bigint", BigIntType::get(ctxt, width))
             .Case("biguint", BigUIntType::get(ctxt, width))
             .Case("bv", BitVectorType::get(ctxt, width))
             .Default(Type());

  if (type)
    return success();

  return failure();
}

static LogicalResult customTypePrinter(Type type, DialectAsmPrinter &printer) {
  return TypeSwitch<Type, LogicalResult>(type)
      .Case<IntType, UIntType, BigIntType, BigUIntType, BitVectorType>(
          [&](auto ty) {
            printer << ty.getMnemonic() << "<" << ty.getWidth() << ">";
            return success();
          })
      .Case<IntBaseType, UIntBaseType, SignedType, UnsignedType,
            BitVectorBaseType>([&](auto ty) {
        printer << ty.getMnemonic();
        return success();
      })
      .Default([](auto ty) { return failure(); });
}

/// Parse a type registered with this dialect.
static ParseResult parseSystemCType(DialectAsmParser &parser, Type &type) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  StringRef mnemonic;
  mlir::OptionalParseResult result =
      generatedTypeParser(parser, &mnemonic, type);
  if (result.has_value())
    return result.value();

  result = customTypeParser(parser, mnemonic, loc, type);
  if (result.has_value())
    return result.value();

  parser.emitError(loc) << "unknown type `" << mnemonic
                        << "` in dialect `moore`";
  return failure();
}

/// Print a type registered with this dialect.
static void printSystemCType(Type type, DialectAsmPrinter &printer) {
  if (succeeded(generatedTypePrinter(type, printer)))
    return;
  if (succeeded(customTypePrinter(type, printer)))
    return;
  assert(false && "no printer for unknown `moore` dialect type");
}

/// Parse a type registered with this dialect.
Type SystemCDialect::parseType(DialectAsmParser &parser) const {
  Type type;
  if (parseSystemCType(parser, type))
    return {};
  return type;
}

/// Print a type registered with this dialect.
void SystemCDialect::printType(Type type, DialectAsmPrinter &printer) const {
  printSystemCType(type, printer);
}