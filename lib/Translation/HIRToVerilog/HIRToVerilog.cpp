//=========- HIRToVerilog.cpp - Verilog Printer ---------------------------===//
//
// This is the main HIR to Verilog Printer implementation.
//
//===----------------------------------------------------------------------===//

#include "Generators.h"
#include "VerilogValue.h"
#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"
#include "circt/Translation/HIRToVerilog.h"
#include "helper.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
//#include "mlir/IR/Module.h"
#include "helper.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormattedStream.h"

#include <functional>
#include <list>
#include <string>

using namespace mlir;
using namespace hir;
using namespace std;

// TODO: Printer should never fail. All the checks we are doing here should
// pass. We should implement these checks in op's verify function.
// TODO: Replace recursive function calls.
namespace {
enum RegionKind { DefOpBody, forOpBody, unrollForOpBody };
/// This class is the verilog output printer for the HIR dialect. It walks
/// throught the module and prints the verilog in the 'out' variable.
class VerilogPrinter {
public:
  VerilogPrinter(llvm::formatted_raw_ostream &output)
      : replacer(outBuffer), out(output) {}

  void printModule(ModuleOp op);
  void printDefOp(DefOp op, unsigned indentAmount = 0);

private:
  unsigned newValueNumber() { return nextValueNum++; }

  helper::StringReplacerClass replacer;

  helper::VerilogMapperClass verilogMapper;

  void region_begin() {
    verilogMapper.pushFrame();
    replacer.pushFrame();
    // outBuffer << "\n//{Region  (level = " << verilogMapper.stackLevel()
    //           << ").\n";
    // outBuffer << "//#VerilogValues = " << verilogMapper.size() << ".\n";
  }
  void region_end() {
    // outBuffer << "\n// #VerilogValues = " << verilogMapper.size() << ".\n";
    // outBuffer << "//}Region  (level = " << verilogMapper.stackLevel()
    //           << ").\n";

    // replacer should process this frame's replacements before verilogMapper
    // pops its frame.
    replacer.popFrame();
    verilogMapper.popFrame();
  }

  void printOperation(Operation *op, unsigned indentAmount = 0);

  // Individual op printers.
  void printConstantOp(hir::ConstantOp op, unsigned indentAmount = 0);
  void printIfOp(IfOp op, unsigned indentAmount = 0);
  void printForOp(ForOp op, unsigned indentAmount = 0);
  void printUnrollForOp(UnrollForOp op, unsigned indentAmount = 0);
  void printMemReadOp(MemReadOp op, unsigned indentAmount = 0);
  void printEQOp(hir::EQOp op, unsigned indentAmount = 0);
  void printNEQOp(hir::NEQOp op, unsigned indentAmount = 0);
  void printGTOp(hir::GTOp op, unsigned indentAmount = 0);
  void printLTOp(hir::LTOp op, unsigned indentAmount = 0);
  void printAndOp(hir::AndOp op, unsigned indentAmount = 0);
  void printOrOp(hir::OrOp op, unsigned indentAmount = 0);
  void printAddOp(hir::AddOp op, unsigned indentAmount = 0);
  void printSubtractOp(hir::SubtractOp op, unsigned indentAmount = 0);
  void printMemWriteOp(MemWriteOp op, unsigned indentAmount = 0);
  void printReturnOp(hir::ReturnOp op, unsigned indentAmount = 0);
  void printYieldOp(hir::YieldOp op, unsigned indentAmount = 0);
  void printWireWriteOp(hir::WireWriteOp op, unsigned indentAmount = 0);
  void printWireReadOp(hir::WireReadOp op, unsigned indentAmount = 0);
  void printAllocOp(hir::AllocOp op, unsigned indentAmount = 0);
  void printDelayOp(hir::DelayOp op, unsigned indentAmount = 0);
  void printCallOp(hir::CallOp op, unsigned indentAmount = 0);

  // Helpers.
  void printBody(Block &block, unsigned indentAmount = 0);
  void printType(Type type);
  void printTimeOffsets(const VerilogValue *timeVar);
  SmallVector<VerilogValue *, 4> convertToVerilog(OperandRange operands);
  SmallVector<VerilogValue *, 4> registerResults(ResultRange operands);

  stringstream outBuffer;
  llvm::formatted_raw_ostream &out;
  unsigned nextValueNum = 0;
  std::stack<std::pair<RegionKind, string>> yieldTarget;
  SmallVector<std::string, 4> outputVerilogNames;
};

SmallVector<VerilogValue *, 4>
VerilogPrinter::convertToVerilog(OperandRange inputs) {
  SmallVector<VerilogValue *, 4> out;
  for (Value input : inputs) {
    out.push_back(verilogMapper.getMutable(input));
  }
  return out;
}

SmallVector<VerilogValue *, 4>
VerilogPrinter::registerResults(ResultRange results) {
  SmallVector<VerilogValue *, 4> out;
  for (Value result : results) {
    verilogMapper.insert(
        result, VerilogValue(result, "v" + to_string(newValueNumber())));
    out.push_back(verilogMapper.getMutable(result));
  }
  return out;
}

void VerilogPrinter::printMemReadOp(MemReadOp op, unsigned indentAmount) {
  Value result = op.res();
  VerilogValue vResult(result, "v" + to_string(newValueNumber()));
  verilogMapper.insert(result, vResult);

  auto addr = convertToVerilog(op.addr());
  Value mem = op.mem();
  auto shape = mem.getType().dyn_cast<hir::MemrefType>().getShape();
  auto packing = mem.getType().dyn_cast<hir::MemrefType>().getPacking();

  assert(addr.size() == shape.size());

  Value tstart = op.tstart();
  VerilogValue *vTstart = verilogMapper.getMutable(tstart);
  Value offset = op.offset();
  int delayValue = 0;
  if (offset) {
    VerilogValue *v_offset = verilogMapper.getMutable(offset);
    delayValue = offset ? v_offset->getIntegerConst() : 0;
  }
  VerilogValue *vMem = verilogMapper.getMutable(mem);
  if (!packing.empty()) {
    // Address bus assignments.
    outBuffer << "assign "
              << vMem->strMemrefAddrValid(addr, vMem->numAccess(addr)) << " = "
              << vTstart->strDelayedWire(delayValue) << ";\n";

    outBuffer << "assign "
              << vMem->strMemrefAddrInput(addr, vMem->numAccess(addr))
              << " = {";
    for (int i = packing.size() - 1; i >= 0; i--) {
      auto p = addr.size() - 1 - packing[i];
      auto addrI = addr[p];
      auto dimI = shape[p];
      if (i < packing.size() - 1)
        outBuffer << ", ";
      outBuffer << addrI->strConstOrWire(ceil(log2(dimI)));
    }
    outBuffer << "};\n";
  }

  // Read bus assignments.
  outBuffer << "wire" << vResult.strWireDecl() << " = "
            << vMem->strMemrefRdData(addr) << ";\n";
  outBuffer << "assign " + vMem->strMemrefRdEnInput(addr, vMem->numReads(addr))
            << " = " << vTstart->strDelayedWire(delayValue) << ";\n\n";

  vMem->incMemrefNumReads(addr);
}

void VerilogPrinter::printCallOp(hir::CallOp op, unsigned indentAmount) {
  ResultRange results = op.res();
  auto vResults = registerResults(results);
  StringRef callee = op.callee();
  OperandRange operands = op.operands();
  Value tstart = op.tstart();
  VerilogValue *vTstart = verilogMapper.getMutable(tstart);
  Value offset = op.offset();
  int delayValue = 0;
  if (offset) {
    delayValue = verilogMapper.get(offset).getIntegerConst();
  }
  string calleeStr = callee.str();

  for (VerilogValue *vResult : vResults) {
    outBuffer << "wire " << vResult->strWireDecl() << ";\n";
  }

  std::string strEpilogue;
  outBuffer << calleeStr << " " << calleeStr << newValueNumber() << "(";
  bool firstArg = true;
  for (VerilogValue *vResult : vResults) {
    assert(!vResult->getType().isa<hir::MemrefType>());
    if (!firstArg) {
      outBuffer << ",\n";
    }
    firstArg = false;
    outBuffer << vResult->strConstOrWire();
  }

  for (Value operand : operands) {
    if (auto memrefType = operand.getType().dyn_cast<MemrefType>()) {
      Details::PortKind port = memrefType.getPort();
      VerilogValue *v_operand = verilogMapper.getMutable(operand);
      if (memrefType.getPacking().size() > 0) {
        if (!firstArg) {
          outBuffer << ",\n";
        }
        firstArg = false;
        outBuffer << v_operand->strMemrefAddrInputIf(v_operand->maxNumAccess());
      }
      if (port == Details::PortKind::r || port == Details::PortKind::rw) {
        if (!firstArg) {
          outBuffer << ",\n";
        }
        firstArg = false;
        if (memrefType.getPacking().size() > 0) {
          outBuffer << v_operand->strMemrefRdEnInputIf(
              v_operand->maxNumReads());
        } else {
          outBuffer << "/*Ignored*/";
        }
        outBuffer << ",\n" << v_operand->strMemrefRdData();
      }
      if (port == Details::PortKind::w || port == Details::PortKind::rw) {
        if (!firstArg) {
          outBuffer << ",\n";
        }
        firstArg = false;
        outBuffer << v_operand->strMemrefWrEnInputIf(v_operand->maxNumWrites())
                  << ",\n"
                  << v_operand->strMemrefWrDataInputIf(
                         v_operand->maxNumWrites());
        strEpilogue +=
            "assign " +
            v_operand->strMemrefWrDataValidIf(v_operand->maxNumAccess()) +
            " = " + v_operand->strMemrefWrEnInputIf(v_operand->maxNumWrites()) +
            ";\n";
      }

      if (memrefType.getPacking().size() > 0) {
        strEpilogue +=
            "assign " +
            v_operand->strMemrefAddrValidIf(v_operand->maxNumAccess()) + " = ";
        if (port == Details::PortKind::rw) {
          strEpilogue +=
              v_operand->strMemrefRdEnInputIf(v_operand->maxNumReads()) + "|" +
              v_operand->strMemrefWrEnInputIf(v_operand->maxNumWrites());
        } else if (port == Details::PortKind::r) {
          strEpilogue +=
              v_operand->strMemrefRdEnInputIf(v_operand->maxNumReads());
        } else {
          strEpilogue +=
              v_operand->strMemrefWrEnInputIf(v_operand->maxNumWrites());
        }
        strEpilogue += ";\n";
      }
      if (port == Details::PortKind::r || port == Details::PortKind::rw) {
        v_operand->incMemrefNumReads();
      }
      if (port == Details::PortKind::w || port == Details::PortKind::rw) {
        v_operand->incMemrefNumWrites();
      }
    } else {
      if (!firstArg) {
        outBuffer << ",\n";
      }
      firstArg = false;
      VerilogValue *v_operand = verilogMapper.getMutable(operand);
      outBuffer << v_operand->strConstOrWire();
    }
  }
  if (!firstArg) {
    outBuffer << ",\n";
  }
  outBuffer << vTstart->strDelayedWire(delayValue);
  outBuffer << ",\n";
  outBuffer << "clk\n";

  outBuffer << ");\n" << strEpilogue;
  for (VerilogValue *vResult : vResults) {
    if (vResult->getType().isa<hir::TimeType>()) {
      printTimeOffsets(vResult);
    }
  }
}

void VerilogPrinter::printDelayOp(hir::DelayOp op, unsigned indentAmount) {
  // contract:
  // delay > 0
  // delay constant

  Value input = op.input();
  VerilogValue *v_input = verilogMapper.getMutable(input);
  // Propagate constant value.
  if (v_input->isIntegerConst()) {
    Value result = op.res();
    verilogMapper.insert(
        result, VerilogValue(result, "v" + to_string(newValueNumber())));
    VerilogValue *vResult = verilogMapper.getMutable(result);
    outBuffer << "//" << vResult->strWire() << " = "
              << v_input->strConstOrError();
    vResult->setIntegerConst(v_input->getIntegerConst());
  }
  Value delayValue = op.delay();
  VerilogValue *v_delay = verilogMapper.getMutable(delayValue);
  Value result = op.res();
  verilogMapper.insert(result,
                       VerilogValue(result, "v" + to_string(newValueNumber())));
  VerilogValue *vResult = verilogMapper.getMutable(result);
  unsigned input_bitwidth = helper::getBitWidth(input.getType());
  string str_shiftreg = "shiftreg" + to_string(newValueNumber());
  outBuffer << "reg[" << input_bitwidth - 1 << ":0]" << str_shiftreg << "["
            << v_delay->strConstOrError() << ":0] = "
            << "'{default:0};\n";

  outBuffer << "always@(*) " << str_shiftreg
            << "[0] <= " << v_input->strConstOrWire() << ";\n";
  if (v_delay->getIntegerConst() > 0) {
    outBuffer << "always@(posedge clk) " << str_shiftreg << "["
              << v_delay->strConstOrError() << ":1] <= " << str_shiftreg << "["
              << v_delay->strConstOrError(-1) << ":0];\n";
  }
  outBuffer << "wire " << vResult->strWireDecl() << " = " << str_shiftreg << "["
            << v_delay->strConstOrError() << "];\n";
  if (vResult->getType().isa<hir::TimeType>())
    printTimeOffsets(vResult);
}

void VerilogPrinter::printAllocOp(hir::AllocOp op, unsigned indentAmount) {
  Operation::result_range result = op.res();
  if (result[0].getType().isa<MemrefType>()) {
    if (result.size() == 1) {
      string str_id_result = to_string(newValueNumber());
      VerilogValue vResult(result[0], "v" + str_id_result);
      outBuffer << vResult.strMemrefInstDecl();
      verilogMapper.insert(result[0], vResult);
      string loc = replacer.insert([=]() -> string {
        return verilogMapper.getMutable(result[0])->strMemrefSelDecl();
      });
      outBuffer << loc << "\n";
    } else {
      assert(result.size() == 2);
      Value r0 = result[0];
      Value r1 = result[1];
      assert(r1.getType().isa<MemrefType>());
      // For now only support below set.
      assert(r0.getType().dyn_cast<MemrefType>().getPort() == Details::r);
      assert(r1.getType().dyn_cast<MemrefType>().getPort() == Details::w);
      for (int i = 0; i < 2; i++) {
        string str_id_result = to_string(newValueNumber());
        VerilogValue vResult(result[i], "v" + str_id_result);
        outBuffer << "//strMemrefInstDecl\n";
        outBuffer << vResult.strMemrefInstDecl();
        verilogMapper.insert(result[i], vResult);
        string loc = replacer.insert([=]() -> string {
          return verilogMapper.getMutable(result[i])->strMemrefSelDecl();
        });
        outBuffer << "//strMemrefSelDecl\n";
        outBuffer << loc << "\n";
      }
      outBuffer << "\n //Instantiate Memory.\n";
      outBuffer << gen_bram("bram_tdp_rf_rf", verilogMapper.get(r0),
                            verilogMapper.get(r1));
    }
  } else {
    assert(result.size() == 1);
    string str_id_result = to_string(newValueNumber());
    VerilogValue vResult(result[0], "v" + str_id_result);
    verilogMapper.insert(result[0], vResult);
    outBuffer << "wire " << vResult.strWireDecl() << ";\n";
  }
}

void VerilogPrinter::printWireReadOp(hir::WireReadOp op,
                                     unsigned indentAmount) {

  Value result = op.res();
  unsigned id_result = newValueNumber();
  VerilogValue vResult(result, "v" + to_string(id_result));
  verilogMapper.insert(result, vResult);

  Value wire = op.wire();
  VerilogValue *v_wire = verilogMapper.getMutable(wire);
  SmallVector<VerilogValue *, 4> addr = convertToVerilog(op.addr());
  string arrayAccessStr = "";
  for (auto idx : addr) {
    arrayAccessStr += "[" + idx->strConstOrError() + "]";
  }
  outBuffer << "wire " << vResult.strWireDecl() << " = " << v_wire->strWire()
            << arrayAccessStr << ";\n";
}

void VerilogPrinter::printWireWriteOp(hir::WireWriteOp op,
                                      unsigned indentAmount) {
  Value value = op.value();
  VerilogValue *vValue = verilogMapper.getMutable(value);
  Value wire = op.wire();
  VerilogValue *v_wire = verilogMapper.getMutable(wire);
  auto addr = convertToVerilog(op.addr());
  string arrayAccessStr = "";

  for (auto idx : addr) {
    if (idx->isIntegerConst()) {
      arrayAccessStr += "[" + to_string(idx->getIntegerConst()) + "]";
    } else {
      arrayAccessStr +=
          "[" + idx->strWire() + " /*ERROR: Expected integer constant.*/]";
    }
  }

  outBuffer << "assign " << v_wire->strWire() << arrayAccessStr << " = "
            << vValue->strConstOrWire() << ";\n";
}

void VerilogPrinter::printMemWriteOp(MemWriteOp op, unsigned indentAmount) {
  auto addr = convertToVerilog(op.addr());
  Value mem = op.mem();
  Value value = op.value();
  Value tstart = op.tstart();
  VerilogValue *vTstart = verilogMapper.getMutable(tstart);
  Value offset = op.offset();
  VerilogValue *v_offset;
  if (offset)
    v_offset = verilogMapper.getMutable(offset);

  auto shape = mem.getType().dyn_cast<hir::MemrefType>().getShape();
  auto packing = mem.getType().dyn_cast<hir::MemrefType>().getPacking();
  if (shape.size() != addr.size()) {
    emitError(op.getLoc(), "Dimension mismatch. Number of addresses is wrong.");
    assert(shape.size() == addr.size());
  }

  VerilogValue *vValue = verilogMapper.getMutable(value);
  VerilogValue *vMem = verilogMapper.getMutable(mem);

  int delayValue = offset ? v_offset->getIntegerConst() : 0;

  if (!packing.empty()) {
    // Address bus assignments.
    outBuffer << "assign "
              << vMem->strMemrefAddrValid(addr, vMem->numAccess(addr)) << " = "
              << vTstart->strDelayedWire(delayValue) << ";\n";

    outBuffer << "assign "
              << vMem->strMemrefAddrInput(addr, vMem->numAccess(addr))
              << " = {";
    for (int i = packing.size() - 1; i >= 0; i--) {
      auto p = addr.size() - 1 - packing[i];
      VerilogValue *addrI = addr[p];
      auto dimI = shape[p];
      if (i < int(packing.size()) - 1)
        outBuffer << ", ";
      outBuffer << addrI->strConstOrWire(ceil(log2(dimI)));
    }
    outBuffer << "};\n";
  }

  outBuffer << "assign "
            << vMem->strMemrefWrEnInput(addr, vMem->numWrites(addr)) << " = "
            << vTstart->strDelayedWire(delayValue) << ";\n";
  outBuffer << "assign "
            << vMem->strMemrefWrDataValid(addr, vMem->numWrites(addr)) << " = "
            << vTstart->strDelayedWire(delayValue) << ";\n";
  outBuffer << "assign "
            << vMem->strMemrefWrDataInput(addr, vMem->numWrites(addr)) << " = "
            << vValue->strConstOrWire() << ";\n\n";

  vMem->incMemrefNumWrites(addr);
}

void VerilogPrinter::printEQOp(hir::EQOp op, unsigned indentAmount) {
  Value result = op.res();
  Type resType = result.getType();
  assert(resType.isa<IntegerType>() || resType.isa<hir::ConstType>());
  unsigned id_result = newValueNumber();
  VerilogValue vResult(result, "v" + to_string(id_result));

  const VerilogValue vLOperand = verilogMapper.get(op.left());
  const VerilogValue vROperand = verilogMapper.get(op.right());
  Type resultType = result.getType();
  if (resultType.isa<hir::ConstType>()) {
    outBuffer << "//wire " << vResult.strWire() << " = "
              << vLOperand.strConstOrError()
              << " == " << vROperand.strConstOrError() << ";\n";
    vResult.setIntegerConst(
        (vLOperand.getIntegerConst() == vROperand.getIntegerConst()) ? 1 : 0);
  } else {
    outBuffer << "wire " << vResult.strWireDecl() << " = "
              << vLOperand.strConstOrWire()
              << " == " << vROperand.strConstOrWire() << ";\n";
    if (vLOperand.isIntegerConst() && vROperand.isIntegerConst()) {
      vResult.setIntegerConst(
          (vLOperand.getIntegerConst() == vROperand.getIntegerConst()) ? 1 : 0);
    }
  }
  verilogMapper.insert(result, vResult);
}

void VerilogPrinter::printNEQOp(hir::NEQOp op, unsigned indentAmount) {
  Value result = op.res();
  Type resType = result.getType();
  assert(resType.isa<IntegerType>() || resType.isa<hir::ConstType>());
  unsigned id_result = newValueNumber();
  VerilogValue vResult(result, "v" + to_string(id_result));

  const VerilogValue vLOperand = verilogMapper.get(op.left());
  const VerilogValue vROperand = verilogMapper.get(op.right());
  Type resultType = result.getType();
  if (resultType.isa<hir::ConstType>()) {
    outBuffer << "//wire " << vResult.strWire() << " = "
              << vLOperand.strConstOrError()
              << " != " << vROperand.strConstOrError() << ";\n";
    vResult.setIntegerConst(
        (vLOperand.getIntegerConst() != vROperand.getIntegerConst()) ? 1 : 0);
  } else {
    outBuffer << "wire " << vResult.strWireDecl() << " = "
              << vLOperand.strConstOrWire()
              << " != " << vROperand.strConstOrWire() << ";\n";
    if (vLOperand.isIntegerConst() && vROperand.isIntegerConst()) {
      vResult.setIntegerConst(
          (vLOperand.getIntegerConst() != vROperand.getIntegerConst()) ? 1 : 0);
    }
  }
  verilogMapper.insert(result, vResult);
}

void VerilogPrinter::printGTOp(hir::GTOp op, unsigned indentAmount) {
  Value result = op.res();
  Type resType = result.getType();
  assert(resType.isa<IntegerType>() || resType.isa<hir::ConstType>());
  unsigned id_result = newValueNumber();
  VerilogValue vResult(result, "v" + to_string(id_result));

  const VerilogValue vLOperand = verilogMapper.get(op.left());
  const VerilogValue vROperand = verilogMapper.get(op.right());
  Type resultType = result.getType();
  if (resultType.isa<hir::ConstType>()) {
    outBuffer << "//wire " << vResult.strWire() << " = "
              << vLOperand.strConstOrError() << " > "
              << vROperand.strConstOrError() << ";\n";
    vResult.setIntegerConst(
        (vLOperand.getIntegerConst() > vROperand.getIntegerConst()) ? 1 : 0);
  } else {
    outBuffer << "wire " << vResult.strWireDecl() << " = "
              << vLOperand.strConstOrWire() << " > "
              << vROperand.strConstOrWire() << ";\n";
    if (vLOperand.isIntegerConst() && vROperand.isIntegerConst()) {
      vResult.setIntegerConst(
          (vLOperand.getIntegerConst() > vROperand.getIntegerConst()) ? 1 : 0);
    }
  }
  verilogMapper.insert(result, vResult);
}

void VerilogPrinter::printLTOp(hir::LTOp op, unsigned indentAmount) {
  Value result = op.res();
  Type resType = result.getType();
  assert(resType.isa<IntegerType>() || resType.isa<hir::ConstType>());
  unsigned id_result = newValueNumber();
  VerilogValue vResult(result, "v" + to_string(id_result));

  const VerilogValue vLOperand = verilogMapper.get(op.left());
  const VerilogValue vROperand = verilogMapper.get(op.right());
  Type resultType = result.getType();
  if (resultType.isa<hir::ConstType>()) {
    outBuffer << "//wire " << vResult.strWire() << " = "
              << vLOperand.strConstOrError() << " < "
              << vROperand.strConstOrError() << ";\n";
    vResult.setIntegerConst(
        (vLOperand.getIntegerConst() < vROperand.getIntegerConst()) ? 1 : 0);
  } else {
    outBuffer << "wire " << vResult.strWireDecl() << " = "
              << vLOperand.strConstOrWire() << " < "
              << vROperand.strConstOrWire() << ";\n";
    if (vLOperand.isIntegerConst() && vROperand.isIntegerConst()) {
      vResult.setIntegerConst(
          (vLOperand.getIntegerConst() < vROperand.getIntegerConst()) ? 1 : 0);
    }
  }
  verilogMapper.insert(result, vResult);
}

void VerilogPrinter::printAndOp(hir::AndOp op, unsigned indentAmount) {
  Value result = op.res();
  Type resType = result.getType();
  assert(resType.isa<IntegerType>() || resType.isa<hir::ConstType>());
  unsigned id_result = newValueNumber();
  VerilogValue vResult(result, "v" + to_string(id_result));

  const VerilogValue vLOperand = verilogMapper.get(op.left());
  const VerilogValue vROperand = verilogMapper.get(op.right());
  Type resultType = result.getType();
  if (resultType.isa<hir::ConstType>()) {
    outBuffer << "//wire " << vResult.strWire() << " = "
              << vLOperand.strConstOrError() << " && "
              << vROperand.strConstOrError() << ";\n";
    vResult.setIntegerConst(
        (vLOperand.getIntegerConst() && vROperand.getIntegerConst()) ? 1 : 0);
  } else {
    outBuffer << "wire " << vResult.strWireDecl() << " = "
              << vLOperand.strConstOrWire() << " &&"
              << vROperand.strConstOrWire() << ";\n";
    if (vLOperand.isIntegerConst() && vROperand.isIntegerConst()) {
      vResult.setIntegerConst(
          (vLOperand.getIntegerConst() && vROperand.getIntegerConst()) ? 1 : 0);
    }
  }
  verilogMapper.insert(result, vResult);
}

void VerilogPrinter::printOrOp(hir::OrOp op, unsigned indentAmount) {
  Value result = op.res();
  Type resType = result.getType();
  assert(resType.isa<IntegerType>() || resType.isa<hir::ConstType>());
  VerilogValue vResult(result, "v" + to_string(newValueNumber()));

  const VerilogValue vLOperand = verilogMapper.get(op.left());
  const VerilogValue vROperand = verilogMapper.get(op.right());
  Type resultType = result.getType();
  if (resultType.isa<hir::ConstType>()) {
    outBuffer << "//wire " << vResult.strWire() << " = "
              << vLOperand.strConstOrError() << " || "
              << vROperand.strConstOrError() << ";\n";
    vResult.setIntegerConst(
        (vLOperand.getIntegerConst() || vROperand.getIntegerConst()) ? 1 : 0);
  } else {
    outBuffer << "wire " << vResult.strWireDecl() << " = "
              << vLOperand.strConstOrWire() << " || "
              << vROperand.strConstOrWire() << ";\n";
    if (vLOperand.isIntegerConst() && vROperand.isIntegerConst()) {
      vResult.setIntegerConst(
          (vLOperand.getIntegerConst() || vROperand.getIntegerConst()) ? 1 : 0);
    }
  }
  verilogMapper.insert(result, vResult);
}

void VerilogPrinter::printAddOp(hir::AddOp op, unsigned indentAmount) {
  // TODO: Handle signed and unsigned integers of unequal width using sign
  // extension.
  Value result = op.res();
  Type resType = result.getType();
  assert(resType.isa<IntegerType>() || resType.isa<hir::ConstType>());
  unsigned id_result = newValueNumber();
  VerilogValue vResult(result, "v" + to_string(id_result));

  const VerilogValue vLOperand = verilogMapper.get(op.left());
  const VerilogValue vROperand = verilogMapper.get(op.right());
  Type resultType = result.getType();
  if (resultType.isa<hir::ConstType>()) {
    outBuffer << "//wire " << vResult.strWire() << " = "
              << vLOperand.strConstOrError() << " + "
              << vROperand.strConstOrError() << ";\n";
    vResult.setIntegerConst(vLOperand.getIntegerConst() +
                            vROperand.getIntegerConst());
  } else {
    outBuffer << "wire " << vResult.strWireDecl() << " = "
              << vLOperand.strConstOrWire() << " + "
              << vROperand.strConstOrWire() << ";\n";
    if (vLOperand.isIntegerConst() && vROperand.isIntegerConst()) {
      vResult.setIntegerConst(vLOperand.getIntegerConst() +
                              vROperand.getIntegerConst());
    }
  }
  verilogMapper.insert(result, vResult);
}

void VerilogPrinter::printSubtractOp(hir::SubtractOp op,
                                     unsigned indentAmount) {
  // TODO: Handle signed and unsigned integers of unequal width using sign
  // extension.
  Value result = op.res();
  unsigned id_result = newValueNumber();
  VerilogValue vResult(result, "v" + to_string(id_result));

  const VerilogValue vLOperand = verilogMapper.get(op.left());
  const VerilogValue vROperand = verilogMapper.get(op.right());
  Type resultType = result.getType();
  if (resultType.isa<hir::ConstType>()) {
    outBuffer << "wire " << vResult.strWireDecl() << " = "
              << vLOperand.strConstOrError() << " - "
              << vROperand.strConstOrError() << ";\n";
    vResult.setIntegerConst(vLOperand.getIntegerConst() -
                            vROperand.getIntegerConst());
  } else {
    outBuffer << "wire " << vResult.strWireDecl() << " = "
              << vLOperand.strConstOrWire() << " - "
              << vROperand.strConstOrWire() << ";\n";
    if (vLOperand.isIntegerConst() && vROperand.isIntegerConst()) {
      vResult.setIntegerConst(vLOperand.getIntegerConst() -
                              vROperand.getIntegerConst());
    }
  }
  verilogMapper.insert(result, vResult);
}

void VerilogPrinter::printReturnOp(hir::ReturnOp op, unsigned indentAmount) {
  auto outputs = op.operands();
  if (outputs.size() != outputVerilogNames.size()) {
    assert(false);
  }
  for (unsigned i = 0; i < outputs.size(); i++) {
    Value output = outputs[i];
    string &verilogName = outputVerilogNames[i];
    outBuffer << "assign " << verilogName << " = "
              << verilogMapper.get(output).strConstOrWire() << ";\n";
  }
}

void VerilogPrinter::printConstantOp(hir::ConstantOp op,
                                     unsigned indentAmount) {
  auto result = op.res();
  unsigned id_result = newValueNumber();
  VerilogValue vResult(result, "v" + to_string(id_result));
  int value = op.value();
  vResult.setIntegerConst(value);
  verilogMapper.insert(result, vResult);

  outBuffer << "//constant " << vResult.strWireDecl();
  outBuffer << " = " << vResult.getBitWidth() << "'d" << value << ";\n";
}

void VerilogPrinter::printIfOp(IfOp op, unsigned indentAmount) {
  Value cond = op.cond();
  Value tstart = op.tstart();
  Block &if_body = op.if_region().front();
  VerilogValue *vTstartPrev = verilogMapper.getMutable(tstart);

  auto id_if = newValueNumber();
  VerilogValue vTstart = VerilogValue(tstart, "v" + to_string(id_if));
  region_begin();
  verilogMapper.insertPtr(tstart, &vTstart);
  outBuffer << "wire " << vTstart.strWire() << " = "
            << vTstartPrev->strConstOrWire() << "&&"
            << verilogMapper.get(cond).strConstOrWire() << ";\n";
  printTimeOffsets(&vTstart);
  printBody(if_body, indentAmount);
  region_end();
  verilogMapper.insertPtr(tstart, vTstartPrev);
}

void VerilogPrinter::printForOp(hir::ForOp op, unsigned indentAmount) {
  Value idx = op.getInductionVar();
  Value tloop = op.getIterTimeVar();
  Value tfinish = op.tfinish();

  auto idLoop = newValueNumber();
  yieldTarget.push(std::make_pair(forOpBody, "tloop_in" + to_string(idLoop)));

  const VerilogValue v_lb = verilogMapper.get(op.lb());
  const VerilogValue v_ub = verilogMapper.get(op.ub());
  const VerilogValue v_step = verilogMapper.get(op.step());
  VerilogValue *vTstart = verilogMapper.getMutable(op.tstart());
  const VerilogValue v_offset = verilogMapper.get(op.offset());
  unsigned delayValue = v_offset.getIntegerConst();
  assert(delayValue > 0);
  VerilogValue v_idx = VerilogValue(idx, "idx" + to_string(idLoop));
  verilogMapper.insert(idx, v_idx);
  verilogMapper.insert(tloop, VerilogValue(tloop, "tloop" + to_string(idLoop)));
  verilogMapper.insert(tfinish,
                       VerilogValue(tfinish, "tfinish" + to_string(idLoop)));
  VerilogValue *v_tloop = verilogMapper.getMutable(tloop);
  VerilogValue *v_tfinish = verilogMapper.getMutable(tfinish);

  unsigned width_lb = verilogMapper.get(op.lb()).getBitWidth();
  unsigned width_ub = verilogMapper.get(op.ub()).getBitWidth();
  unsigned width_step = verilogMapper.get(op.step()).getBitWidth();
  unsigned width_idx = verilogMapper.get(idx).getBitWidth();

  stringstream loopCounterStream;
  string loopCounterTemplate =
      "\nreg[$msb_idx:0] $v_idx ;"
      "\nreg[$msb_ub:0] ub$idLoop ;"
      "\nreg[$msb_step:0] step$idLoop ;"
      "\nwire tloop_in$idLoop;"
      "\nreg $v_tloop;"
      "\nreg $v_tfinish;"
      "\nalways@(posedge clk) begin"
      "\n if(/*tstart=*/ $vTstart) begin"
      "\n   $v_idx <= $v_lb; //lower bound."
      "\n   step$idLoop <= $v_step;"
      "\n   ub$idLoop <= $v_ub;"
      "\n   $v_tloop <= ($v_ub > $v_lb);"
      "\n   $v_tfinish <=!($v_ub > $v_lb);"
      "\n end"
      "\n else if (tloop_in$idLoop) begin"
      "\n   $v_idx <= $v_idx + step$idLoop; //increment"
      "\n   $v_tloop <= ($v_idx + step$idLoop) < ub$idLoop;"
      "\n   $v_tfinish <= !(($v_idx + step$idLoop) < ub$idLoop);"
      "\n end"
      "\n else begin"
      "\n   $v_tloop <= 1'b0;"
      "\n   $v_tfinish <= 1'b0;"
      "\n end"
      "\nend";
  loopCounterStream << loopCounterTemplate;

  string loopCounterString = loopCounterStream.str();
  helper::findAndReplaceAll(loopCounterString, "$idLoop", to_string(idLoop));
  helper::findAndReplaceAll(loopCounterString, "$msb_idx",
                            to_string(width_idx - 1));
  helper::findAndReplaceAll(loopCounterString, "$v_lb", v_lb.strConstOrWire());
  helper::findAndReplaceAll(loopCounterString, "$msb_lb",
                            to_string(width_lb - 1));
  helper::findAndReplaceAll(loopCounterString, "$v_ub", v_ub.strConstOrWire());
  helper::findAndReplaceAll(loopCounterString, "$msb_ub",
                            to_string(width_ub - 1));
  helper::findAndReplaceAll(loopCounterString, "$v_step",
                            v_step.strConstOrWire());
  helper::findAndReplaceAll(loopCounterString, "$msb_step",
                            to_string(width_step - 1));
  helper::findAndReplaceAll(loopCounterString, "$vTstart",
                            vTstart->strDelayedWire(delayValue - 1));
  helper::findAndReplaceAll(loopCounterString, "$v_tloop", v_tloop->strWire());
  helper::findAndReplaceAll(loopCounterString, "$v_tfinish",
                            v_tfinish->strWire());
  helper::findAndReplaceAll(loopCounterString, "$v_idx", v_idx.strWire());
  outBuffer << "\n//{ Loop" << idLoop << "\n";
  outBuffer << loopCounterString;
  outBuffer << "\n//Loop" << idLoop << " body\n";
  printTimeOffsets(v_tloop);
  printBody(op.getLoopBody().front(), indentAmount);
  outBuffer << "\n//} Loop" << idLoop << "\n";
  printTimeOffsets(v_tfinish);
  yieldTarget.pop();
}

void VerilogPrinter::printUnrollForOp(UnrollForOp op, unsigned indentAmount) {
  Value tlast = op.tlast(); // output
  int lb = op.lb();
  int ub = op.ub();
  int step = op.step();
  Value tstart = op.tstart();
  VerilogValue *vTstart = verilogMapper.getMutable(tstart);
  Value tloop = op.getIterTimeVar();
  Value idx = op.getInductionVar();
  verilogMapper.insertPtr(tloop, vTstart);
  auto idLoop = newValueNumber();
  verilogMapper.insert(
      idx,
      VerilogValue(
          idx,
          "idx" + to_string(idLoop))); // FIXME remove name of the VerilogValue.
  VerilogValue v_tloop;
  for (int i = lb; i < ub; i += step) {
    region_begin();
    auto id_tloop = newValueNumber();

    outBuffer << "\n//{ Unrolled body " << i
              << " of loop" + to_string(idLoop) + ".\n";
    VerilogValue *v_idx = verilogMapper.getMutable(idx);
    v_idx->setIntegerConst(i);
    outBuffer << "//DEBUG: " << v_idx->strConstOrWire() << ", expected " << i
              << "\n";
    if (i > lb) {
      printTimeOffsets(&v_tloop);
      verilogMapper.insertPtr(tloop, &v_tloop);
    }
    VerilogValue next_v_tloop =
        VerilogValue(tloop, "tloop" + to_string(id_tloop));
    yieldTarget.push(make_pair(unrollForOpBody, next_v_tloop.strWire()));

    // At first iteration, tloop -> vTstart, whose time offsets are already
    // printed.
    printBody(op.getLoopBody().front(), indentAmount);

    yieldTarget.pop();
    outBuffer << "\n//} Unrolled body " << i
              << " of loop" + to_string(idLoop) + ".\n";

    outBuffer << "//DEBUG: " << verilogMapper.getMutable(idx)->strConstOrWire()
              << ", expected " << i << "\n";
    region_end();
    v_tloop = next_v_tloop;
  }

  outBuffer << "\n//{ Assign tlast of prev UnrollForLoop\n";
  verilogMapper.insert(tlast,
                       VerilogValue(tlast, "t" + to_string(newValueNumber())));
  const VerilogValue *vTLast = verilogMapper.getMutable(tlast);
  outBuffer << "wire " << vTLast->strWireDecl() << ";\n";
  outBuffer << "assign " << vTLast->strWire() << " = "
            << verilogMapper.get(tloop).strWire() << ";\n";
  printTimeOffsets(vTLast);

  outBuffer << "\n//} Assign tlast of prev UnrollForLoop";
}

string formattedOp(Operation *op, string opName) {
  string out;
  string locStr;
  llvm::raw_string_ostream locOstream(locStr);
  op->getLoc().print(locOstream);
  out = opName + " at " + locStr + "\n";
  return out;
}

void VerilogPrinter::printOperation(Operation *inst, unsigned indentAmount) {
  if (auto op = dyn_cast<hir::ConstantOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "ConstantOp");
    return printConstantOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::CallOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "CallOp");
    return printCallOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::AllocOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "AllocOp");
    return printAllocOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::DelayOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "DelayOp");
    return printDelayOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::IfOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "IfOp");
    return printIfOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::ForOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "ForOp");
    return printForOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::UnrollForOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "UnrollForOp");
    return printUnrollForOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::ReturnOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "ReturnOp");
    return printReturnOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::MemReadOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "MemReadOp");
    return printMemReadOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::MemWriteOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "MemWriteOp");
    return printMemWriteOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::WireReadOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "WireReadOp");
    return printWireReadOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::WireWriteOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "WireWriteOp");
    return printWireWriteOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::EQOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "EQOp");
    return printEQOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::NEQOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "NEqOp");
    return printNEQOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::GTOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "GTOp");
    return printGTOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::LTOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "LTOp");
    return printLTOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::AndOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "AndOp");
    return printAndOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::OrOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "OrOp");
    return printOrOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::AddOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "AddOp");
    return printAddOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::SubtractOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "SubtractOp");
    return printSubtractOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::YieldOp>(inst)) {
    outBuffer << "\n//" << formattedOp(inst, "YieldOp");
    return printYieldOp(op, indentAmount);
  }
  if (auto op = dyn_cast<hir::TerminatorOp>(inst)) {
    outBuffer << "\n//TerminatorOp\n";
    return;
  }
  emitError(inst->getLoc(), "Unsupported Operation for codegen!");
  assert(false);
}

void VerilogPrinter::printDefOp(DefOp op, unsigned indentAmount) {
  Block &entryBlock = op.getBody().front();
  auto args = entryBlock.getArguments();
  auto resTypes = op.getType().getResults();
  // Print the module signature.
  outBuffer << "module " << op.getName().str() << "(";
  outBuffer << "\n//Outputs.\n";
  bool firstVerilogArg = true;
  for (unsigned i = 0; i < resTypes.size(); i++) {
    if (!firstVerilogArg)
      outBuffer << ",";
    outBuffer << "\n";
    firstVerilogArg = false;
    Type type = resTypes[i];
    unsigned width;
    if (type.isa<IntegerType>())
      width = type.dyn_cast<IntegerType>().getWidth();
    else if (type.isa<TimeType>())
      width = 1;
    else
      assert(false);
    std::string verilogName = "out" + std::to_string(i);
    outputVerilogNames.push_back(verilogName);
    outBuffer << "output wire[" << width - 1 << ":0] " << verilogName;
  }

  outBuffer << "\n//Inputs.\n";
  for (unsigned i = 0; i < args.size(); i++) {
    Value arg = args[i];
    Type argType = arg.getType();

    // Print verilog.

    if (!firstVerilogArg)
      outBuffer << ",";
    firstVerilogArg = false;
    outBuffer << "\n";

    if (argType.isa<hir::TimeType>()) {
      /*tstart parameter of func does not use valuenumber*/
      auto tstart = VerilogValue(arg, "tstart");
      verilogMapper.insert(arg, tstart);

      outBuffer << "//TimeType.\n";
      outBuffer << "input wire " << tstart.strWireDecl();
    } else if (argType.isa<IntegerType>()) {
      auto vIntArg = VerilogValue(arg, "v" + to_string(newValueNumber()));
      verilogMapper.insert(arg, vIntArg);

      outBuffer << "//IntegerType.\n";
      outBuffer << "input wire" << vIntArg.strWireDecl();
    } else if (argType.isa<MemrefType>()) {
      unsigned idMemrefArg = newValueNumber();
      auto vMemref_arg = VerilogValue(arg, "v" + to_string(idMemrefArg));
      verilogMapper.insert(arg, vMemref_arg);
      outBuffer << vMemref_arg.strMemrefArgDecl();
    } else {
      emitError(op.getLoc(), "Unsupported argument type!");
      assert(false);
    }

    if (i == args.size())
      outBuffer << "\n";
  }
  outBuffer << ",\n//Clock.\ninput wire clk\n);\n\n";

  region_begin();
  for (auto arg : args) {
    if (arg.getType().isa<hir::MemrefType>()) {
      string loc = replacer.insert([=]() -> string {
        return verilogMapper.getMutable(arg)->strMemrefSelDecl();
      });
      outBuffer << loc << "\n";
    }
  }

  const VerilogValue *vTstart = verilogMapper.getMutable(args.back());
  printTimeOffsets(vTstart);
  printBody(entryBlock);

  region_end();

  outBuffer << "endmodule\n";

  out << outBuffer.str();
  outBuffer = std::stringstream();
  nextValueNum = 0;
}

void VerilogPrinter::printYieldOp(hir::YieldOp op, unsigned indentAmount) {
  auto operands = op.operands();
  VerilogValue *vTstart = verilogMapper.getMutable(op.tstart());
  Value offset = op.offset();
  int delayValue = 0;
  if (offset)
    delayValue = verilogMapper.get(offset).getIntegerConst();

  if (!operands.empty())
    emitError(op.getLoc(), "YieldOp codegen does not support arguments yet!");
  auto yieldPoint = yieldTarget.top();
  if (yieldPoint.first == forOpBody) {
    assert(delayValue > 0);
    outBuffer << "assign " << yieldPoint.second << " = "
              << vTstart->strDelayedWire(delayValue - 1) << ";\n";
  } else if (yieldPoint.first == unrollForOpBody) {
    outBuffer << "wire " << yieldPoint.second << " = "
              << vTstart->strDelayedWire(delayValue) << ";\n";
  }
}

void VerilogPrinter::printTimeOffsets(const VerilogValue *timeVar) {
  outBuffer << "//printTimeOffset\n";
  string loc = replacer.insert(
      [=]() -> string { return to_string(timeVar->getMaxDelay()); });

  outBuffer << "reg " << timeVar->strDelayedWire() << "[" << loc
            << ":0] = '{default:0} ;\n";
  outBuffer << "always@(*) " << timeVar->strDelayedWire()
            << "[0] <= " << timeVar->strWire() << ";\n";
  string str_i = "i" + to_string(newValueNumber());
  outBuffer << "generate\n";
  outBuffer << "genvar " << str_i << ";\n";
  outBuffer << "\nfor(" << str_i << " = 1; " << str_i << "<= " << loc << "; "
            << str_i << "= " << str_i << " + 1) begin\n";
  outBuffer << "always@(posedge clk) begin\n";
  outBuffer << timeVar->strDelayedWire() << "[" << str_i
            << "] <= " << timeVar->strDelayedWire() << "[" << str_i << "-1];\n";
  outBuffer << "end\n";
  outBuffer << "end\n";
  outBuffer << "endgenerate\n\n";
}

void VerilogPrinter::printBody(Block &block, unsigned indentAmount) {

  region_begin();

  // Print the operations within the entity.
  for (auto iter = block.begin(); iter != block.end(); ++iter) {
    printOperation(&(*iter), 4);
  }

  region_end();
}

void VerilogPrinter::printModule(ModuleOp module) {
  outBuffer << "`default_nettype none\n";
  outBuffer << "`include \"helper.sv\"\n";
  WalkResult result = module.walk([this](DefOp defOp) -> WalkResult {
    fprintf(stderr, "DefOp found\n");
    fflush(stderr);
    printDefOp(defOp);
    return WalkResult::advance();
  });
  // if printing of a single operation failed, fail the whole translation.
}
} // namespace.

LogicalResult hir::printVerilog(ModuleOp module, raw_ostream &os) {
  llvm::formatted_raw_ostream out(os);
  out << "`default_nettype none\n";
  out << "`include \"helper.sv\"\n";
  WalkResult result = module.walk([&out](DefOp defOp) -> WalkResult {
    VerilogPrinter printer(out);
    fprintf(stderr, "DefOp found\n");
    fflush(stderr);
    printer.printDefOp(defOp);
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

void hir::registerHIRToVerilogTranslation() {
  mlir::TranslateFromMLIRRegistration registration(
      "hir-to-verilog", printVerilog,
      [](DialectRegistry &registry) { registry.insert<HIRDialect>(); });
}
