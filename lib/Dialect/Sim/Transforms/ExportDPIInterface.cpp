//===- ExportDPIInterface.cpp - Export DPI function interfaces --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/ExportDPIInterface.h"
#include "circt/Dialect/SV/DPITypeInfo.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace circt;

namespace {

llvm::json::Object exportType(const sv::DPITypeInfo &type) {
  return llvm::json::Object{{"width", type.width}, {"signed", type.isSigned}};
}

bool isCIdentifier(StringRef name) {
  return !name.empty() &&
         (llvm::isAlpha(name.front()) || name.front() == '_') &&
         llvm::all_of(name.drop_front(),
                      [](char c) { return llvm::isAlnum(c) || c == '_'; });
}

} // namespace

LogicalResult sim::exportDPIInterface(mlir::ModuleOp module,
                                      llvm::raw_ostream &output) {
  llvm::json::Array functions;
  llvm::StringSet<> functionNames;
  for (auto func : module.getOps<sim::DPIFuncOp>()) {
    auto name = func.getVerilogName().value_or(func.getSymName());
    if (!isCIdentifier(name) || !sv::isNameValid(name, false)) {
      func.emitError() << "DPI function name \"" << name
                       << "\" is not a valid C/SV identifier";
      return failure();
    }
    if (!functionNames.insert(name).second) {
      func.emitError() << "duplicate DPI function name \"" << name << "\"";
      return failure();
    }
    llvm::json::Array args;
    for (const auto &arg : func.getDpiFunctionType().getArguments()) {
      if (arg.dir == sim::DPIDirection::Ref) {
        func.emitError("IEEE 1800 DPI does not allow ref arguments");
        return failure();
      }
      auto type = sv::resolveDPIType(arg.type);
      if (failed(type) ||
          (arg.dir == sim::DPIDirection::Return && !type->isValidReturn())) {
        func.emitError() << "unsupported DPI schema argument " << arg.name
                         << " of type " << arg.type << " with direction "
                         << sim::stringifyDPIDirectionKeyword(arg.dir);
        return failure();
      }
      auto argument = exportType(*type);
      argument["name"] = arg.name.getValue().str();
      argument["direction"] = sim::stringifyDPIDirectionKeyword(arg.dir).str();
      args.push_back(std::move(argument));
    }
    functions.push_back(llvm::json::Object{{"function", name.str()},
                                           {"arguments", std::move(args)}});
  }
  if (functions.empty()) {
    module.emitError("no sim.func.dpi operations to export");
    return failure();
  }

  llvm::json::Object schema{{"dpi_functions", std::move(functions)}};
  output << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(schema)));
  return success();
}
