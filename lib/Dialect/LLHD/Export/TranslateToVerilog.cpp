//===- TranslateToVerilog.cpp - Verilog Printer ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main LLHD to Verilog Printer implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/Translation/TranslateToVerilog.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "llvm/Support/FormattedStream.h"

using namespace mlir;
using namespace circt;

namespace {

class VerilogPrinter {
public:
  VerilogPrinter(llvm::formatted_raw_ostream &output) : out(output) {}

  LogicalResult printModule(ModuleOp op);
  LogicalResult printOperation(Operation *op, unsigned indentAmount = 0);

private:
  LogicalResult printType(Type type);
  LogicalResult printUnaryOp(Operation *op, StringRef opSymbol,
                             unsigned indentAmount = 0);
  LogicalResult printBinaryOp(Operation *op, StringRef opSymbol,
                              unsigned indentAmount = 0);
  LogicalResult printSignedBinaryOp(Operation *op, StringRef opSymbol,
                                    unsigned indentAmount = 0);

  /// Prints a SSA value. In case no mapping to a name exists yet, a new one is
  /// added.
  raw_ostream &printVariableName(Value value);

  void printNewInstantiationName() { out << "inst_" << instCount++; }

  /// Adds an alias for an existing SSA value. In case doesn't exist, it just
  /// adds the alias as a new value.
  void addAliasVariable(Value alias, Value existing);

  unsigned instCount = 0;
  llvm::formatted_raw_ostream &out;
  unsigned nextValueNum = 0;
  DenseMap<Value, unsigned> mapValueToName;
  DenseMap<Value, unsigned> timeValueMap;
};

LogicalResult VerilogPrinter::printModule(ModuleOp module) {
  WalkResult result = module.walk([this](llhd::EntityOp entity) -> WalkResult {
    // An EntityOp always has a single block
    Block &entryBlock = entity.body().front();

    // Print the module signature
    out << "module _" << entity.getName();
    if (!entryBlock.args_empty()) {
      out << "(";
      for (unsigned int i = 0, e = entryBlock.getNumArguments(); i < e; ++i) {
        out << (i > 0 ? ", " : "") << (i < entity.ins() ? "input " : "output ");
        (void)printType(entryBlock.getArgument(i).getType());
        out << " ";
        printVariableName(entryBlock.getArgument(i));
      }
      out << ")";
    }
    out << ";\n";

    // Print the operations within the entity
    for (auto iter = entryBlock.begin();
         iter != entryBlock.end() && !dyn_cast<llhd::TerminatorOp>(iter);
         ++iter) {
      if (failed(printOperation(&(*iter), 4))) {
        return emitError(iter->getLoc(), "Operation not supported!");
      }
    }

    out << "endmodule\n";
    // Reset variable name counter as variables only have to be unique within a
    // module
    nextValueNum = 0;
    return WalkResult::advance();
  });
  // if printing of a single operation failed, fail the whole translation
  return failure(result.wasInterrupted());
}

LogicalResult VerilogPrinter::printBinaryOp(Operation *inst, StringRef opSymbol,
                                            unsigned indentAmount) {
  // Check that the operation is indeed a binary operation
  if (inst->getNumOperands() != 2) {
    return emitError(inst->getLoc(),
                     "This operation does not have two operands!");
  }
  if (inst->getNumResults() != 1) {
    return emitError(inst->getLoc(),
                     "This operation does not have one result!");
  }

  // Print the operation
  out.PadToColumn(indentAmount);
  out << "wire ";
  if (failed(printType(inst->getResult(0).getType())))
    return failure();
  out << " ";
  printVariableName(inst->getResult(0)) << " = ";
  printVariableName(inst->getOperand(0)) << " " << opSymbol << " ";
  printVariableName(inst->getOperand(1)) << ";\n";
  return success();
}

LogicalResult VerilogPrinter::printSignedBinaryOp(Operation *inst,
                                                  StringRef opSymbol,
                                                  unsigned indentAmount) {
  // Note: the wire is not declared as signed, because if you have the result
  // of two signed operation as an input to an unsigned operation, it would
  // perform the signed version (sometimes this is not a problem because it's
  // the same, but e.g. for modulo it would make a difference). Alternatively
  // we could use $unsigned at every operation where it makes a difference,
  // but that would look a lot uglier, we could also track which variable is
  // signed and which unsigned and only do the conversion when necessary, but
  // that is more effort. Also, because we have the explicit $signed at every
  // signed operation, this isn't a problem for further signed operations.
  //
  // Check that the operation is indeed a binary operation
  if (inst->getNumOperands() != 2) {
    return emitError(inst->getLoc(),
                     "This operation does not have two operands!");
  }
  if (inst->getNumResults() != 1) {
    return emitError(inst->getLoc(),
                     "This operation does not have one result!");
  }

  // Print the operation
  out.PadToColumn(indentAmount);
  out << "wire ";
  if (failed(printType(inst->getResult(0).getType())))
    return failure();
  out << " ";
  printVariableName(inst->getResult(0)) << " = $signed(";
  printVariableName(inst->getOperand(0)) << ") " << opSymbol << " $signed(";
  printVariableName(inst->getOperand(1)) << ");\n";
  return success();
}

LogicalResult VerilogPrinter::printUnaryOp(Operation *inst, StringRef opSymbol,
                                           unsigned indentAmount) {
  // Check that the operation is indeed a unary operation
  if (inst->getNumOperands() != 1) {
    return emitError(inst->getLoc(),
                     "This operation does not have exactly one operand!");
  }
  if (inst->getNumResults() != 1) {
    return emitError(inst->getLoc(),
                     "This operation does not have one result!");
  }

  // Print the operation
  out.PadToColumn(indentAmount);
  out << "wire ";
  if (failed(printType(inst->getResult(0).getType())))
    return failure();
  out << " ";
  printVariableName(inst->getResult(0)) << " = " << opSymbol;
  printVariableName(inst->getOperand(0)) << ";\n";
  return success();
}

LogicalResult VerilogPrinter::printOperation(Operation *inst,
                                             unsigned indentAmount) {
  if (auto op = dyn_cast<llhd::ConstOp>(inst)) {
    if (IntegerAttr intAttr = op.value().dyn_cast<IntegerAttr>()) {
      // Integer constant
      out.PadToColumn(indentAmount);
      out << "wire ";
      if (failed(printType(inst->getResult(0).getType())))
        return failure();
      out << " ";
      printVariableName(inst->getResult(0))
          << " = " << op.getResult().getType().getIntOrFloatBitWidth() << "'d"
          << intAttr.getValue().getZExtValue() << ";\n";
      return success();
    }
    if (llhd::TimeAttr timeAttr = op.value().dyn_cast<llhd::TimeAttr>()) {
      // Time Constant
      if (timeAttr.getTime() == 0 && timeAttr.getDelta() != 1) {
        return emitError(
            op.getLoc(),
            "Not possible to translate a time attribute with 0 real "
            "time and non-1 delta.");
      }
      // Track time value for future use
      timeValueMap.insert(
          std::make_pair(inst->getResult(0), timeAttr.getTime()));
      return success();
    }
    return failure();
  }
  if (auto op = dyn_cast<llhd::SigOp>(inst)) {
    out.PadToColumn(indentAmount);
    out << "var ";
    if (failed(printType(inst->getResult(0).getType())))
      return failure();
    out << " ";
    printVariableName(inst->getResult(0)) << " = ";
    printVariableName(op.init()) << ";\n";
    return success();
  }
  if (auto op = dyn_cast<llhd::PrbOp>(inst)) {
    // Prb is a nop, it just defines an alias for an already existing wire
    addAliasVariable(inst->getResult(0), op.signal());
    return success();
  }
  if (auto op = dyn_cast<llhd::DrvOp>(inst)) {
    out.PadToColumn(indentAmount);
    out << "assign ";
    printVariableName(op.signal())
        << " = #(" << timeValueMap.lookup(op.time()) << "ns) ";
    if (op.enable()) {
      printVariableName(op.enable()) << " ? ";
      printVariableName(op.value()) << " : ";
      printVariableName(op.signal()) << ";\n";
    } else {
      printVariableName(op.value()) << ";\n";
    }
    return success();
  }
  if (auto op = dyn_cast<llhd::AndOp>(inst)) {
    return printBinaryOp(inst, "&", indentAmount);
  }
  if (auto op = dyn_cast<llhd::OrOp>(inst)) {
    return printBinaryOp(inst, "|", indentAmount);
  }
  if (auto op = dyn_cast<llhd::XorOp>(inst)) {
    return printBinaryOp(inst, "^", indentAmount);
  }
  if (auto op = dyn_cast<llhd::NotOp>(inst)) {
    return printUnaryOp(inst, "~", indentAmount);
  }
  if (auto op = dyn_cast<llhd::ShlOp>(inst)) {
    unsigned baseWidth = inst->getOperand(0).getType().getIntOrFloatBitWidth();
    unsigned hiddenWidth =
        inst->getOperand(1).getType().getIntOrFloatBitWidth();
    unsigned combinedWidth = baseWidth + hiddenWidth;

    out.PadToColumn(indentAmount);
    out << "wire [" << (combinedWidth - 1) << ":0] ";
    printVariableName(inst->getResult(0)) << "tmp0 = {";
    printVariableName(inst->getOperand(0)) << ", ";
    printVariableName(inst->getOperand(1)) << "};\n";

    out.PadToColumn(indentAmount);
    out << "wire [" << (combinedWidth - 1) << ":0] ";
    printVariableName(inst->getResult(0)) << "tmp1 = ";
    printVariableName(inst->getResult(0)) << "tmp0 << ";
    printVariableName(inst->getOperand(2)) << ";\n";

    out.PadToColumn(indentAmount);
    out << "wire ";
    if (failed(printType(inst->getResult(0).getType())))
      return failure();
    out << " ";
    printVariableName(inst->getResult(0)) << " = ";
    printVariableName(inst->getResult(0))
        << "tmp1[" << (combinedWidth - 1) << ':' << hiddenWidth << "];\n";

    return success();
  }
  if (auto op = dyn_cast<llhd::ShrOp>(inst)) {
    unsigned baseWidth = inst->getOperand(0).getType().getIntOrFloatBitWidth();
    unsigned hiddenWidth =
        inst->getOperand(1).getType().getIntOrFloatBitWidth();
    unsigned combinedWidth = baseWidth + hiddenWidth;

    out.PadToColumn(indentAmount);
    out << "wire [" << (combinedWidth - 1) << ":0] ";
    printVariableName(inst->getResult(0)) << "tmp0 = {";
    printVariableName(inst->getOperand(1)) << ", ";
    printVariableName(inst->getOperand(0)) << "};\n";

    out.PadToColumn(indentAmount);
    out << "wire [" << (combinedWidth - 1) << ":0] ";
    printVariableName(inst->getResult(0)) << "tmp1 = ";
    printVariableName(inst->getResult(0)) << "tmp0 << ";
    printVariableName(inst->getOperand(2)) << ";\n";

    out.PadToColumn(indentAmount);
    out << "wire ";
    if (failed(printType(inst->getResult(0).getType())))
      return failure();
    out << " ";
    printVariableName(inst->getResult(0)) << " = ";
    printVariableName(inst->getResult(0))
        << "tmp1[" << (baseWidth - 1) << ":0];\n";

    return success();
  }
  if (auto op = dyn_cast<llhd::NegOp>(inst)) {
    return printUnaryOp(inst, "-", indentAmount);
  }
  if (auto op = dyn_cast<AddIOp>(inst)) {
    return printBinaryOp(inst, "+", indentAmount);
  }
  if (auto op = dyn_cast<SubIOp>(inst)) {
    return printBinaryOp(inst, "-", indentAmount);
  }
  if (auto op = dyn_cast<MulIOp>(inst)) {
    return printBinaryOp(inst, "*", indentAmount);
  }
  if (auto op = dyn_cast<UnsignedDivIOp>(inst)) {
    return printBinaryOp(inst, "/", indentAmount);
  }
  if (auto op = dyn_cast<SignedDivIOp>(inst)) {
    return printSignedBinaryOp(inst, "/", indentAmount);
  }
  if (auto op = dyn_cast<UnsignedRemIOp>(inst)) {
    // % in Verilog is the remainder in LLHD semantics
    return printBinaryOp(inst, "%", indentAmount);
  }
  if (auto op = dyn_cast<SignedRemIOp>(inst)) {
    // % in Verilog is the remainder in LLHD semantics
    return printSignedBinaryOp(inst, "%", indentAmount);
  }
  if (auto op = dyn_cast<llhd::SModOp>(inst)) {
    return emitError(op.getLoc(),
                     "Signed modulo operation is not yet supported!");
  }
  if (auto op = dyn_cast<CmpIOp>(inst)) {
    switch (op.getPredicate()) {
    case mlir::CmpIPredicate::eq:
      return printBinaryOp(inst, "==", indentAmount);
    case mlir::CmpIPredicate::ne:
      return printBinaryOp(inst, "!=", indentAmount);
    case mlir::CmpIPredicate::sge:
      return printSignedBinaryOp(inst, ">=", indentAmount);
    case mlir::CmpIPredicate::sgt:
      return printSignedBinaryOp(inst, ">", indentAmount);
    case mlir::CmpIPredicate::sle:
      return printSignedBinaryOp(inst, "<=", indentAmount);
    case mlir::CmpIPredicate::slt:
      return printSignedBinaryOp(inst, "<", indentAmount);
    case mlir::CmpIPredicate::uge:
      return printBinaryOp(inst, ">=", indentAmount);
    case mlir::CmpIPredicate::ugt:
      return printBinaryOp(inst, ">", indentAmount);
    case mlir::CmpIPredicate::ule:
      return printBinaryOp(inst, "<=", indentAmount);
    case mlir::CmpIPredicate::ult:
      return printBinaryOp(inst, "<", indentAmount);
    }
    return failure();
  }
  if (auto op = dyn_cast<llhd::InstOp>(inst)) {
    out.PadToColumn(indentAmount);
    out << "_" << op.callee() << " ";
    printNewInstantiationName();
    if (op.inputs().size() > 0 || op.outputs().size() > 0)
      out << " (";
    unsigned counter = 0;
    for (Value arg : op.inputs()) {
      if (counter++ > 0)
        out << ", ";
      printVariableName(arg);
    }
    for (Value arg : op.outputs()) {
      if (counter++ > 0)
        out << ", ";
      printVariableName(arg);
    }
    if (op.inputs().size() > 0 || op.outputs().size() > 0)
      out << ")";
    out << ";\n";
    return success();
  }
  // TODO: insert structural operations here
  return failure();
}

LogicalResult VerilogPrinter::printType(Type type) {
  if (type.isIntOrFloat()) {
    unsigned int width = type.getIntOrFloatBitWidth();
    if (width != 1)
      out << '[' << (width - 1) << ":0]";
    return success();
  }
  if (llhd::SigType sig = type.dyn_cast<llhd::SigType>()) {
    return printType(sig.getUnderlyingType());
  }
  return failure();
}

raw_ostream &VerilogPrinter::printVariableName(Value value) {
  if (!mapValueToName.count(value)) {
    mapValueToName.insert(std::make_pair(value, nextValueNum));
    nextValueNum++;
  }
  return out << '_' << mapValueToName.lookup(value);
}

void VerilogPrinter::addAliasVariable(Value alias, Value existing) {
  if (!mapValueToName.count(existing)) {
    mapValueToName.insert(std::make_pair(alias, nextValueNum));
    nextValueNum++;
  } else {
    mapValueToName.insert(
        std::make_pair(alias, mapValueToName.lookup(existing)));
  }
}

} // anonymous namespace

LogicalResult circt::llhd::exportVerilog(ModuleOp module, raw_ostream &os) {
  llvm::formatted_raw_ostream out(os);
  VerilogPrinter printer(out);
  return printer.printModule(module);
}

void circt::llhd::registerToVerilogTranslation() {
  TranslateFromMLIRRegistration registration(
      "export-llhd-verilog", exportVerilog, [](DialectRegistry &registry) {
        registry.insert<mlir::StandardOpsDialect, llhd::LLHDDialect>();
      });
}
