//===- BLIFEmitter.cpp - BLIF dialect to .blif emitter --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a .blif file emitter.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/BLIF/BLIFEmitter.h"
#include "circt/Dialect/BLIF/BLIFDialect.h"
#include "circt/Dialect/BLIF/BLIFOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "export-blif"

using namespace circt;
using namespace blif;

//===----------------------------------------------------------------------===//
// Emitter
//===----------------------------------------------------------------------===//

// NOLINTBEGIN(misc-no-recursion)
namespace {

/// An emitter for FIRRTL dialect operations to .fir output.
struct Emitter {
  Emitter(llvm::raw_ostream &os) : os(os) {}
  LogicalResult finalize();

  // Circuit/module emission
  void emitModel(ModelOp op);
  void emitModuleParameters(Operation *op, ArrayAttr parameters);

  // Statement emission
  void emitCommand(LogicGateOp op);
  void emitCommand(LatchGateOp op);

private:
  /// Emit an error and remark that emission failed.
  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    encounteredError = true;
    return op->emitError(message);
  }

  /// Emit an error and remark that emission failed.
  InFlightDiagnostic emitOpError(Operation *op, const Twine &message) {
    encounteredError = true;
    return op->emitOpError(message);
  }

private:
  /// Pretty printer.
  llvm::raw_ostream &os;

  /// Whether we have encountered any errors during emission.
  bool encounteredError = false;

  /// The names used to emit values already encountered. Anything that gets a
  /// name in the output FIR is listed here, such that future expressions can
  /// reference it.
  DenseMap<Value, StringRef> valueNames;
  StringSet<> valueNamesStorage;
  int nameCounter = 0;

  /// Return the name used during emission of a `Value`, or none if the value
  /// has not yet been emitted or it was emitted inline.
  StringRef lookupEmittedName(Value value) {
    auto it = valueNames.find(value);
    if (it != valueNames.end())
      return {it->second};
    std::string name = "v" + std::to_string(nameCounter++);
    while (valueNamesStorage.count(name)) {
      name = "v" + std::to_string(nameCounter++);
    }
    addValueName(value, name);
    return valueNames.find(value)->second;
  }

  void addValueName(Value value, StringAttr attr) {
    valueNames.insert({value, attr.getValue()});
  }
  void addValueName(Value value, StringRef str) {
    auto it = valueNamesStorage.insert(str);
    valueNames.insert({value, it.first->getKey()});
  }
};
} // namespace

LogicalResult Emitter::finalize() { return failure(encounteredError); }

/// Emit an entire circuit.
void Emitter::emitModel(ModelOp op) {
  os << ".model " << op.getSymName() << "\n";
  auto mType = op.getModuleType();
  auto clockMask = op.getClocks();

  int inIdx = 0;
  int outIdx = 0;
  auto &body = op.getBody().front();
  auto term = body.getTerminator();
  for (auto [idx, port] : llvm::enumerate(mType.getPorts())) {
    switch (port.dir) {
    case hw::ModulePort::Direction::Input:
      if (clockMask.extractBits(idx, 1).isZero())
        os << ".input " << port.name.getValue() << "\n";
      else
        os << ".clock " << port.name.getValue() << "\n";
      addValueName(body.getArgument(inIdx++), port.name);
      break;
    case hw::ModulePort::Direction::Output:
      os << ".output " << port.name.getValue() << "\n";
      addValueName(term->getOperand(outIdx++), port.name);
      break;
    case hw::ModulePort::Direction::InOut:
      emitOpError(op, "inout ports are not expected");
      break;
    }
  }

  // Emit the model body.
  for (auto &bodyOp : body) {
    if (encounteredError)
      return;
    TypeSwitch<Operation *>(&bodyOp)
        .Case<LogicGateOp>([&](auto op) { emitCommand(op); })
        .Case<LatchGateOp>([&](auto op) { emitCommand(op); })
        .Case<OutputOp>([&](auto op) {})
        .Default([&](auto op) {
          emitOpError(op, "not supported for emission inside model definition");
        });
  }

  os << ".end\n";
}

void Emitter::emitCommand(LogicGateOp op) {
  os << ".names";
  for (auto input : op.getInputs())
    os << " " << lookupEmittedName(input);
  os << " " << lookupEmittedName(op.getResult()) << "\n";

  for (auto row : op.getFunc()) {
    const char *names[] = {"0", "1", "-"};
    for (auto input : ArrayRef(row).drop_back())
      os << names[input];
    if (row.size() > 1)
      os << " ";
    os << names[row.back()] << "\n";
  }
}

void Emitter::emitCommand(LatchGateOp op) {
  os << ".latch";
  os << " " << lookupEmittedName(op.getInput());
  os << " " << lookupEmittedName(op.getOutput());
  bool emitClk = true;
  switch (op.getMode()) {
  case LatchModeEnum::Unspecified:
    emitClk = false;
    break;
  case LatchModeEnum::FallingEdge:
    os << " fe ";
    break;
  case LatchModeEnum::RisingEdge:
    os << " re ";
    break;
  case LatchModeEnum::ActiveHigh:
    os << " ah ";
    break;
  case LatchModeEnum::ActiveLow:
    os << " al ";
    break;
  case LatchModeEnum::Asynchronous:
    os << " as ";
    break;
  }
  if (emitClk) {
    if (op.getClock())
      os << lookupEmittedName(op.getClock());
    else
      os << "NIL";
  }
  if (op.getInitVal() != 3)
    os << " " << op.getInitVal();
  os << "\n";
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

// Emit the specified BLIF circuit into the given output stream.
mlir::LogicalResult circt::blif::exportBLIFFile(mlir::ModuleOp module,
                                                llvm::raw_ostream &os) {
  Emitter emitter(os);
  for (auto &op : *module.getBody()) {
    if (auto circuitOp = dyn_cast<ModelOp>(op))
      emitter.emitModel(circuitOp);
  }
  return emitter.finalize();
}

void circt::blif::registerToBLIFFileTranslation() {
  static mlir::TranslateFromMLIRRegistration toBLIF(
      "export-blif", "emit BLIF dialect operations to .blif output",
      [](ModuleOp module, llvm::raw_ostream &os) {
        return exportBLIFFile(module, os);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<hw::HWDialect>();
        registry.insert<blif::BLIFDialect>();
      });
}
