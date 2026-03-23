//===- SimTypes.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/SimTypes.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

// Include the generated enum definitions (DPIDirection).
#include "circt/Dialect/Sim/SimEnums.h.inc"

using namespace circt;
using namespace sim;
using namespace mlir;

//===----------------------------------------------------------------------===//
// DPIDirection helpers
//===----------------------------------------------------------------------===//

StringRef sim::stringifyDPIDirectionKeyword(DPIDirection dir) {
  switch (dir) {
  case DPIDirection::Input:
    return "input";
  case DPIDirection::Output:
    return "output";
  case DPIDirection::InOut:
    return "inout";
  case DPIDirection::Return:
    return "return";
  case DPIDirection::Ref:
    return "ref";
  }
  llvm_unreachable("unknown DPIDirection");
}

std::optional<DPIDirection> sim::parseDPIDirectionKeyword(StringRef keyword) {
  return llvm::StringSwitch<std::optional<DPIDirection>>(keyword)
      .Case("input", DPIDirection::Input)
      .Case("output", DPIDirection::Output)
      .Case("inout", DPIDirection::InOut)
      .Case("return", DPIDirection::Return)
      .Case("ref", DPIDirection::Ref)
      .Default(std::nullopt);
}

bool sim::isCallOperandDir(DPIDirection dir) {
  return dir == DPIDirection::Input || dir == DPIDirection::InOut ||
         dir == DPIDirection::Ref;
}

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Sim/SimTypes.cpp.inc"

void SimDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Sim/SimTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DPIModuleType storage
//===----------------------------------------------------------------------===//

sim::detail::DPIModuleTypeStorage::DPIModuleTypeStorage(
    ArrayRef<DPIPort> dpiPorts)
    : ports(dpiPorts) {
  for (auto [idx, p] : llvm::enumerate(ports)) {
    if (p.dir == DPIDirection::Input || p.dir == DPIDirection::InOut ||
        p.dir == DPIDirection::Ref)
      inputToAbs.push_back(idx);
    if (p.dir == DPIDirection::Output || p.dir == DPIDirection::InOut ||
        p.dir == DPIDirection::Return)
      resultToAbs.push_back(idx);
  }
}

//===----------------------------------------------------------------------===//
// DPIModuleType methods
//===----------------------------------------------------------------------===//

ArrayRef<DPIPort> DPIModuleType::getPorts() const {
  return getImpl()->getPorts();
}

size_t DPIModuleType::getNumPorts() const { return getImpl()->ports.size(); }

SmallVector<DPIPort> DPIModuleType::getInputPorts() const {
  SmallVector<DPIPort> result;
  for (auto idx : getImpl()->inputToAbs)
    result.push_back(getImpl()->ports[idx]);
  return result;
}

SmallVector<DPIPort> DPIModuleType::getResultPorts() const {
  SmallVector<DPIPort> result;
  for (auto idx : getImpl()->resultToAbs)
    result.push_back(getImpl()->ports[idx]);
  return result;
}

const DPIPort *DPIModuleType::getReturnPort() const {
  auto &ports = getImpl()->ports;
  if (!ports.empty() && ports.back().dir == DPIDirection::Return)
    return &ports.back();
  return nullptr;
}

FunctionType DPIModuleType::getFunctionType() const {
  SmallVector<Type> inputs, results;
  for (auto &port : getImpl()->ports) {
    if (port.dir == DPIDirection::Input || port.dir == DPIDirection::InOut ||
        port.dir == DPIDirection::Ref)
      inputs.push_back(port.type);
    if (port.dir == DPIDirection::Output || port.dir == DPIDirection::InOut ||
        port.dir == DPIDirection::Return)
      results.push_back(port.type);
  }
  return FunctionType::get(getContext(), inputs, results);
}

hw::ModuleType DPIModuleType::getHWModuleType() const {
  SmallVector<hw::ModulePort> hwPorts;
  for (auto &port : getImpl()->ports) {
    hw::ModulePort::Direction hwDir;
    switch (port.dir) {
    case DPIDirection::Input:
    case DPIDirection::Ref:
      hwDir = hw::ModulePort::Direction::Input;
      break;
    case DPIDirection::Output:
    case DPIDirection::Return:
      hwDir = hw::ModulePort::Direction::Output;
      break;
    case DPIDirection::InOut:
      hwDir = hw::ModulePort::Direction::InOut;
      break;
    }
    hwPorts.push_back({port.name, port.type, hwDir});
  }
  return hw::ModuleType::get(getContext(), hwPorts);
}

LogicalResult
DPIModuleType::verify(function_ref<InFlightDiagnostic()> emitError) const {
  auto dpiPorts = getPorts();
  unsigned returnCount = 0;
  for (auto [i, port] : llvm::enumerate(dpiPorts)) {
    if (port.dir == DPIDirection::Return) {
      ++returnCount;
      if (i != dpiPorts.size() - 1)
        return emitError() << "'return' port must be the last port";
    }
  }
  if (returnCount > 1)
    return emitError() << "must have at most one 'return' port";
  return success();
}

//===----------------------------------------------------------------------===//
// DPIModuleType parser/printer
//===----------------------------------------------------------------------===//

/// Parse: !sim.dpi_modty<input "a" : i32, output "b" : i32>
Type DPIModuleType::parse(AsmParser &parser) {
  SmallVector<DPIPort> ports;
  if (parser.parseLess())
    return {};

  // Handle empty port list.
  if (succeeded(parser.parseOptionalGreater()))
    return get(parser.getContext(), ports);

  auto parseOnePort = [&]() -> ParseResult {
    StringRef dirKeyword;
    if (parser.parseKeyword(&dirKeyword))
      return failure();
    auto dir = parseDPIDirectionKeyword(dirKeyword);
    if (!dir) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected DPI port direction keyword");
      return failure();
    }
    std::string name;
    if (parser.parseString(&name))
      return failure();
    Type type;
    if (parser.parseColonType(type))
      return failure();
    ports.push_back({StringAttr::get(parser.getContext(), name), type, *dir});
    return success();
  };

  if (parser.parseCommaSeparatedList(parseOnePort) || parser.parseGreater())
    return {};
  return get(parser.getContext(), ports);
}

/// Print: !sim.dpi_modty<input "a" : i32, output "b" : i32>
void DPIModuleType::print(AsmPrinter &printer) const {
  printer << '<';
  llvm::interleaveComma(getPorts(), printer, [&](const DPIPort &port) {
    printer << stringifyDPIDirectionKeyword(port.dir) << ' ';
    printer << '"' << port.name.getValue() << '"';
    printer << " : ";
    printer.printType(port.type);
  });
  printer << '>';
}
