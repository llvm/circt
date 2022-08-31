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
}
