#include "circt/Target/Verilog/VerilogPrinter.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

LogicalResult mlir::llhd::VerilogPrinter::printModule(ModuleOp module) {
  bool didFail = false;
  module.walk([&didFail, this](llhd::EntityOp entity) {
    // An EntityOp always has a single block
    Block &entryBlock = entity.body().front();

    // Print the module signature
    out << "module _" << entity.getName();
    if (!entryBlock.args_empty()) {
      out << "(";
      for (unsigned int i = 0; i < entryBlock.getNumArguments(); i++) {
        out << (i > 0 ? ", " : "")
            << (i < entity.ins().getZExtValue() ? "input " : "output ");
        printType(entryBlock.getArgument(i).getType());
        out << " " << getVariableName(entryBlock.getArgument(i));
      }
      out << ")";
    }
    out << ";\n";

    // Print the operations within the entity
    for (auto iter = entryBlock.begin();
         iter != entryBlock.end() && !dyn_cast<llhd::TerminatorOp>(iter);
         iter++) {
      if (failed(printOperation(&(*iter), 4))) {
        emitError(iter->getLoc(), "Operation not supported!");
        didFail = true;
        return;
      }
    }

    out << "endmodule\n";
    // Reset variable name counter as variables only have to be unique within a
    // module
    nextValueNum = 0;
  });
  // if printing of a single operation failed, fail the whole translation
  if (didFail)
    return failure();

  return success();
}

LogicalResult mlir::llhd::VerilogPrinter::printBinaryOp(Operation *inst,
                                                        StringRef opSymbol,
                                                        unsigned indentAmount) {
  // Check that the operation is indeed a binary operation
  if (inst->getNumOperands() != 2) {
    emitError(inst->getLoc(), "This operation does not have two operands!");
    return failure();
  }
  if (inst->getNumResults() != 1) {
    emitError(inst->getLoc(), "This operation does not have one result!");
    return failure();
  }

  // Print the operation
  out.PadToColumn(indentAmount);
  out << "wire ";
  if (failed(printType(inst->getResult(0).getType())))
    return failure();
  out << " " << getVariableName(inst->getResult(0)) << " = "
      << getVariableName(inst->getOperand(0)) << " " << opSymbol << " "
      << getVariableName(inst->getOperand(1)) << ";\n";
  return success();
}

LogicalResult mlir::llhd::VerilogPrinter::printSignedBinaryOp(
    Operation *inst, StringRef opSymbol, unsigned indentAmount) {
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
    emitError(inst->getLoc(), "This operation does not have two operands!");
    return failure();
  }
  if (inst->getNumResults() != 1) {
    emitError(inst->getLoc(), "This operation does not have one result!");
    return failure();
  }

  // Print the operation
  out.PadToColumn(indentAmount);
  out << "wire ";
  if (failed(printType(inst->getResult(0).getType())))
    return failure();
  out << " " << getVariableName(inst->getResult(0)) << " = $signed("
      << getVariableName(inst->getOperand(0)) << ") " << opSymbol << " $signed("
      << getVariableName(inst->getOperand(1)) << ");\n";
  return success();
}

LogicalResult mlir::llhd::VerilogPrinter::printUnaryOp(Operation *inst,
                                                       StringRef opSymbol,
                                                       unsigned indentAmount) {
  // Check that the operation is indeed a unary operation
  if (inst->getNumOperands() != 1) {
    emitError(inst->getLoc(),
              "This operation does not have exactly one operand!");
    return failure();
  }
  if (inst->getNumResults() != 1) {
    emitError(inst->getLoc(), "This operation does not have one result!");
    return failure();
  }

  // Print the operation
  out.PadToColumn(indentAmount);
  out << "wire ";
  if (failed(printType(inst->getResult(0).getType())))
    return failure();
  out << " " << getVariableName(inst->getResult(0)) << " = " << opSymbol
      << getVariableName(inst->getOperand(0)) << ";\n";
  return success();
}

LogicalResult
mlir::llhd::VerilogPrinter::printOperation(Operation *inst,
                                           unsigned indentAmount) {
  if (auto op = dyn_cast<llhd::ConstOp>(inst)) {
    if (op.value().getKind() == StandardAttributes::Integer) {
      // Integer constant
      out.PadToColumn(indentAmount);
      out << "wire ";
      if (failed(printType(inst->getResult(0).getType())))
        return failure();
      out << " " << getVariableName(inst->getResult(0)) << " = "
          << op.getResult().getType().getIntOrFloatBitWidth() << "'d"
          << op.value().cast<IntegerAttr>().getValue().getZExtValue() << ";\n";
      return success();
    } else if (llhd::TimeAttr::kindof(op.value().getKind())) {
      // Time Constant
      auto timeAttr = op.value().cast<TimeAttr>();
      if (timeAttr.getTime() == 0 && timeAttr.getDelta() != 1) {
        emitError(op.getLoc(),
                  "Not possible to translate a time attribute with 0 real "
                  "time and non-1 delta.");
        return failure();
      } else {
        // Track time value for future use
        timeValueMap.insert(
            std::make_pair(inst->getResult(0), timeAttr.getTime()));
        return success();
      }
    }
    return failure();
  }
  if (auto op = dyn_cast<llhd::SigOp>(inst)) {
    out.PadToColumn(indentAmount);
    out << "var ";
    if (failed(printType(inst->getResult(0).getType())))
      return failure();
    out << " " << getVariableName(inst->getResult(0)) << " = "
        << getVariableName(op.init()) << ";\n";
    return success();
  }
  if (auto op = dyn_cast<llhd::PrbOp>(inst)) {
    // Prb is a nop, it just defines an alias for an already existing wire
    addAliasVariable(inst->getResult(0), op.signal());
    return success();
  }
  if (auto op = dyn_cast<llhd::DrvOp>(inst)) {
    out.PadToColumn(indentAmount);
    out << "assign " << getVariableName(op.signal()) << " = #("
        << timeValueMap.lookup(op.time()) << "ns) ";
    if (op.enable()) {
      out << getVariableName(op.enable()) << " ? "
          << getVariableName(op.value()) << " : "
          << getVariableName(op.signal()) << ";\n";
    } else {
      out << getVariableName(op.value()) << ";\n";
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
    out << "wire [" << (combinedWidth - 1) << ":0] "
        << getVariableName(inst->getResult(0)) << "tmp0 = {"
        << getVariableName(inst->getOperand(0)) << ", "
        << getVariableName(inst->getOperand(1)) << "};\n";

    out.PadToColumn(indentAmount);
    out << "wire [" << (combinedWidth - 1) << ":0] "
        << getVariableName(inst->getResult(0))
        << "tmp1 = " << getVariableName(inst->getResult(0)) << "tmp0 << "
        << getVariableName(inst->getOperand(2)) << ";\n";

    out.PadToColumn(indentAmount);
    out << "wire ";
    if (failed(printType(inst->getResult(0).getType())))
      return failure();
    out << " " << getVariableName(inst->getResult(0)) << " = "
        << getVariableName(inst->getResult(0)) << "tmp1[" << (combinedWidth - 1)
        << ':' << hiddenWidth << "];\n";

    return success();
  }
  if (auto op = dyn_cast<llhd::ShrOp>(inst)) {
    unsigned baseWidth = inst->getOperand(0).getType().getIntOrFloatBitWidth();
    unsigned hiddenWidth =
        inst->getOperand(1).getType().getIntOrFloatBitWidth();
    unsigned combinedWidth = baseWidth + hiddenWidth;

    out.PadToColumn(indentAmount);
    out << "wire [" << (combinedWidth - 1) << ":0] "
        << getVariableName(inst->getResult(0)) << "tmp0 = {"
        << getVariableName(inst->getOperand(1)) << ", "
        << getVariableName(inst->getOperand(0)) << "};\n";

    out.PadToColumn(indentAmount);
    out << "wire [" << (combinedWidth - 1) << ":0] "
        << getVariableName(inst->getResult(0))
        << "tmp1 = " << getVariableName(inst->getResult(0)) << "tmp0 << "
        << getVariableName(inst->getOperand(2)) << ";\n";

    out.PadToColumn(indentAmount);
    out << "wire ";
    if (failed(printType(inst->getResult(0).getType())))
      return failure();
    out << " " << getVariableName(inst->getResult(0)) << " = "
        << getVariableName(inst->getResult(0)) << "tmp1[" << (baseWidth - 1)
        << ":0];\n";

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
    emitError(op.getLoc(), "Signed modulo operation is not yet supported!");
    return failure();
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
    out << "_" << op.callee() << " " << getNewInstantiationName();
    if (op.inputs().size() > 0 || op.outputs().size() > 0)
      out << " (";
    unsigned counter = 0;
    for (Value arg : op.inputs()) {
      if (counter++ > 0)
        out << ", ";
      out << getVariableName(arg);
    }
    for (Value arg : op.outputs()) {
      if (counter++ > 0)
        out << ", ";
      out << getVariableName(arg);
    }
    if (op.inputs().size() > 0 || op.outputs().size() > 0)
      out << ")";
    out << ";\n";
    return success();
  }
  // TODO: insert structural operations here
  return failure();
}

LogicalResult mlir::llhd::VerilogPrinter::printType(Type type) {
  if (type.isIntOrFloat()) {
    unsigned int width = type.getIntOrFloatBitWidth();
    if (width != 1)
      out << '[' << (width - 1) << ":0]";
    return success();
  } else if (llhd::SigType::kindof(type.getKind())) {
    llhd::SigType sig = type.cast<llhd::SigType>();
    return printType(sig.getUnderlyingType());
  }
  return failure();
}

Twine mlir::llhd::VerilogPrinter::getVariableName(Value value) {
  if (!mapValueToName.count(value)) {
    mapValueToName.insert(std::make_pair(value, nextValueNum));
    nextValueNum++;
  }
  return Twine("_") + Twine(mapValueToName.lookup(value));
}

void mlir::llhd::VerilogPrinter::addAliasVariable(Value alias, Value existing) {
  if (!mapValueToName.count(existing)) {
    mapValueToName.insert(std::make_pair(alias, nextValueNum));
    nextValueNum++;
  } else {
    mapValueToName.insert(
        std::make_pair(alias, mapValueToName.lookup(existing)));
  }
}
