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
    return "in";
  case DPIDirection::Output:
    return "out";
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
      .Case("in", DPIDirection::Input)
      .Case("out", DPIDirection::Output)
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
// DPIFunctionType storage
//===----------------------------------------------------------------------===//

sim::detail::DPIFunctionTypeStorage::DPIFunctionTypeStorage(
    ArrayRef<DPIArgument> dpiArgs)
    : arguments(dpiArgs) {
  for (auto [idx, a] : llvm::enumerate(arguments)) {
    if (a.dir == DPIDirection::Input || a.dir == DPIDirection::InOut ||
        a.dir == DPIDirection::Ref)
      inputToAbs.push_back(idx);
    if (a.dir == DPIDirection::Output || a.dir == DPIDirection::InOut ||
        a.dir == DPIDirection::Return)
      resultToAbs.push_back(idx);
  }
}

sim::detail::DPIFunctionTypeStorage *
sim::detail::DPIFunctionTypeStorage::construct(TypeStorageAllocator &allocator,
                                               const KeyTy &key) {
  auto *storage = new (allocator.allocate<DPIFunctionTypeStorage>())
      DPIFunctionTypeStorage(key);
  // Compute and cache the MLIR FunctionType.
  SmallVector<Type> inputs, results;
  for (auto &arg : storage->arguments) {
    if (arg.dir == DPIDirection::Input || arg.dir == DPIDirection::InOut ||
        arg.dir == DPIDirection::Ref)
      inputs.push_back(arg.type);
    if (arg.dir == DPIDirection::Output || arg.dir == DPIDirection::InOut ||
        arg.dir == DPIDirection::Return)
      results.push_back(arg.type);
  }
  MLIRContext *ctx = nullptr;
  if (!storage->arguments.empty()) {
    ctx = storage->arguments[0].type.getContext();
    assert(ctx && "DPIArgument types must have a valid MLIRContext");
  }
  if (ctx)
    storage->cachedFuncType = FunctionType::get(ctx, inputs, results);
  return storage;
}

//===----------------------------------------------------------------------===//
// DPIFunctionType methods
//===----------------------------------------------------------------------===//

ArrayRef<DPIArgument> DPIFunctionType::getArguments() const {
  return getImpl()->getArguments();
}

size_t DPIFunctionType::getNumArguments() const {
  return getImpl()->arguments.size();
}

SmallVector<DPIArgument> DPIFunctionType::getInputArguments() const {
  SmallVector<DPIArgument> result;
  for (auto idx : getImpl()->inputToAbs)
    result.push_back(getImpl()->arguments[idx]);
  return result;
}

SmallVector<DPIArgument> DPIFunctionType::getResultArguments() const {
  SmallVector<DPIArgument> result;
  for (auto idx : getImpl()->resultToAbs)
    result.push_back(getImpl()->arguments[idx]);
  return result;
}

const DPIArgument *DPIFunctionType::getReturnArgument() const {
  auto &args = getImpl()->arguments;
  if (!args.empty() && args.back().dir == DPIDirection::Return)
    return &args.back();
  return nullptr;
}

FunctionType DPIFunctionType::getFunctionType() const {
  auto cached = getImpl()->getCachedFunctionType();
  if (cached)
    return cached;
  // Empty argument list — context unavailable during storage construction.
  return FunctionType::get(getContext(), {}, {});
}

LogicalResult
DPIFunctionType::verify(function_ref<InFlightDiagnostic()> emitError) const {
  auto dpiArgs = getArguments();
  unsigned returnCount = 0;
  for (auto [i, arg] : llvm::enumerate(dpiArgs)) {
    if (arg.dir == DPIDirection::Return) {
      ++returnCount;
      if (i != dpiArgs.size() - 1)
        return emitError() << "'return' argument must be the last argument";
    }
  }
  if (returnCount > 1)
    return emitError() << "must have at most one 'return' argument";
  return success();
}

//===----------------------------------------------------------------------===//
// DPIFunctionType parser/printer
//===----------------------------------------------------------------------===//

/// Parse: !sim.dpi_functy<input "a" : i32, output "b" : i32>
Type DPIFunctionType::parse(AsmParser &parser) {
  SmallVector<DPIArgument> args;
  if (parser.parseLess())
    return {};

  // Handle empty argument list.
  if (succeeded(parser.parseOptionalGreater()))
    return get(parser.getContext(), args);

  auto parseOneArg = [&]() -> ParseResult {
    StringRef dirKeyword;
    if (parser.parseKeyword(&dirKeyword))
      return failure();
    auto dir = parseDPIDirectionKeyword(dirKeyword);
    if (!dir) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected DPI argument direction keyword");
      return failure();
    }
    std::string name;
    if (parser.parseString(&name))
      return failure();
    Type type;
    if (parser.parseColonType(type))
      return failure();
    args.push_back({StringAttr::get(parser.getContext(), name), type, *dir});
    return success();
  };

  if (parser.parseCommaSeparatedList(parseOneArg) || parser.parseGreater())
    return {};
  return get(parser.getContext(), args);
}

/// Print: !sim.dpi_functy<input "a" : i32, output "b" : i32>
void DPIFunctionType::print(AsmPrinter &printer) const {
  printer << '<';
  llvm::interleaveComma(getArguments(), printer, [&](const DPIArgument &arg) {
    printer << stringifyDPIDirectionKeyword(arg.dir) << ' ';
    printer << '"' << arg.name.getValue() << '"';
    printer << " : ";
    printer.printType(arg.type);
  });
  printer << '>';
}
