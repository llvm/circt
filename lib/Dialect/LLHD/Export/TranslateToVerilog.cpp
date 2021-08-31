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
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
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
  LogicalResult printVariadicOp(Operation *op, StringRef opSymbol,
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

LogicalResult VerilogPrinter::printVariadicOp(Operation *inst,
                                              StringRef opSymbol,
                                              unsigned indentAmount) {
  // Check that the operation is indeed a binary operation
  if (inst->getNumOperands() < 1) {
    return emitError(inst->getLoc(),
                     "This operation does not have at least one operand!");
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

  for (unsigned i = 0; i < inst->getNumOperands() - 1; i++) {
    printVariableName(inst->getOperand(i)) << " " << opSymbol << " ";
  }
  printVariableName(inst->getOperand(inst->getNumOperands() - 1)) << ";\n";

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
  if (auto op = dyn_cast<comb::AndOp>(inst)) {
    return printVariadicOp(inst, "&", indentAmount);
  }
  if (auto op = dyn_cast<comb::OrOp>(inst)) {
    return printVariadicOp(inst, "|", indentAmount);
  }
  if (auto op = dyn_cast<comb::XorOp>(inst)) {
    return printVariadicOp(inst, "^", indentAmount);
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
  if (auto op = dyn_cast<comb::AddOp>(inst))
    return printVariadicOp(inst, "+", indentAmount);
  if (auto op = dyn_cast<comb::SubOp>(inst))
    return printVariadicOp(inst, "-", indentAmount);
  if (auto op = dyn_cast<comb::MulOp>(inst))
    return printVariadicOp(inst, "*", indentAmount);
  if (auto op = dyn_cast<comb::DivUOp>(inst))
    return printVariadicOp(inst, "/", indentAmount);
  if (auto op = dyn_cast<comb::DivSOp>(inst))
    return printSignedBinaryOp(inst, "/", indentAmount);
  if (auto op = dyn_cast<comb::ModUOp>(inst))
    return printVariadicOp(inst, "%", indentAmount);
  if (auto op = dyn_cast<comb::ModSOp>(inst))
    // The result of % in Verilog takes the sign of the dividend
    return printSignedBinaryOp(inst, "%", indentAmount);
  if (auto op = dyn_cast<comb::ShlOp>(inst))
    return printVariadicOp(inst, "<<", indentAmount);
  if (auto op = dyn_cast<comb::ShrUOp>(inst))
    return printVariadicOp(inst, ">>", indentAmount);
  if (auto op = dyn_cast<comb::ShrSOp>(inst))
    // The right operand is also converted to a signed value, but in Verilog the
    // amount is always treated as unsigned.
    // TODO: would be better to not print the signed conversion of the second
    // operand.
    return printSignedBinaryOp(inst, ">>>", indentAmount);
  if (auto op = dyn_cast<comb::ICmpOp>(inst)) {
    switch (op.predicate()) {
    case comb::ICmpPredicate::eq:
      return printVariadicOp(inst, "==", indentAmount);
    case comb::ICmpPredicate::ne:
      return printVariadicOp(inst, "!=", indentAmount);
    case comb::ICmpPredicate::sge:
      return printSignedBinaryOp(inst, ">=", indentAmount);
    case comb::ICmpPredicate::sgt:
      return printSignedBinaryOp(inst, ">", indentAmount);
    case comb::ICmpPredicate::sle:
      return printSignedBinaryOp(inst, "<=", indentAmount);
    case comb::ICmpPredicate::slt:
      return printSignedBinaryOp(inst, "<", indentAmount);
    case comb::ICmpPredicate::uge:
      return printVariadicOp(inst, ">=", indentAmount);
    case comb::ICmpPredicate::ugt:
      return printVariadicOp(inst, ">", indentAmount);
    case comb::ICmpPredicate::ule:
      return printVariadicOp(inst, "<=", indentAmount);
    case comb::ICmpPredicate::ult:
      return printVariadicOp(inst, "<", indentAmount);
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
  if (auto op = dyn_cast<comb::ParityOp>(inst))
    return printUnaryOp(inst, "^", indentAmount);
  if (auto op = dyn_cast<comb::ExtractOp>(inst)) {
    out.PadToColumn(indentAmount);
    out << "wire ";
    if (failed(printType(op.result().getType())))
      return failure();
    out << " ";
    printVariableName(op.result());
    out << " = ";
    printVariableName(op.input());
    out << "["
        << (op.lowBit() + op.result().getType().getIntOrFloatBitWidth() - 1)
        << ":" << op.lowBit() << "];\n";
    return success();
  }
  if (auto op = dyn_cast<comb::SExtOp>(inst)) {
    out.PadToColumn(indentAmount);
    out << "wire ";
    if (failed(printType(op.result().getType())))
      return failure();
    out << " ";
    printVariableName(op.result());
    out << " = ";
    out << "{{"
        << (op.result().getType().getIntOrFloatBitWidth() -
            op.input().getType().getIntOrFloatBitWidth())
        << "{";
    printVariableName(op.input());
    out << "[" << (op.input().getType().getIntOrFloatBitWidth() - 1) << "]}}, ";
    printVariableName(op.input());
    out << "};\n";
    return success();
  }
  if (auto op = dyn_cast<comb::ConcatOp>(inst)) {
    out.PadToColumn(indentAmount);
    out << "wire ";
    if (failed(printType(op.result().getType())))
      return failure();
    out << " ";
    printVariableName(op.result());
    out << " = {";
    bool first = true;
    for (unsigned i = 0; i < op->getNumOperands(); i++) {
      if (!first)
        out << ", ";
      printVariableName(op.getOperand(i));
      first = false;
    }
    out << "};\n";
    return success();
  }
  if (auto op = dyn_cast<comb::MuxOp>(inst)) {
    out.PadToColumn(indentAmount);
    out << "wire ";
    if (failed(printType(op.result().getType())))
      return failure();
    out << " ";
    printVariableName(op.result());
    out << " = ";
    printVariableName(op.cond());
    out << " ? ";
    printVariableName(op.trueValue());
    out << " : ";
    printVariableName(op.falseValue());
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
        registry.insert<llhd::LLHDDialect, comb::CombDialect>();
      });
}
