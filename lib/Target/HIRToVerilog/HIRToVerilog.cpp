//=========- HIRToVerilog.cpp - Verilog Printer ---------------------------===//
//
// This is the main HIR to Verilog Printer implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Target/HIRToVerilog/HIRToVerilog.h"
#include "Helpers.h"
#include "VerilogValue.h"
#include "circt/Dialect/HIR/HIR.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
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
string loopCounterTemplate =
    "\nwire[$msb_lb:0] lb$n_loop = $v_lb;"
    "\nwire[$msb_ub:0] ub$n_loop = $v_ub;"
    "\nwire[$msb_step:0] step$n_loop = $v_step;"
    "\nwire tstart$n_loop = $v_tstart;"
    "\nwire tloop_in$n_loop;"
    "\n"
    "\nreg[$msb_idx:0] prev_idx$n_loop= 0;"
    "\nreg[$msb_ub:0] saved_ub$n_loop= 0;"
    "\nreg[$msb_step:0] saved_step$n_loop= 0;"
    "\nwire[$msb_idx:0] idx$n_loop = tstart$n_loop? lb$n_loop "
    ": tloop_in$n_loop ? "
    "(prev_idx$n_loop+saved_step$n_loop): prev_idx$n_loop;"
    "\nwire tloop_out$n_loop = tstart$n_loop"
    "|| ((idx$n_loop < saved_ub$n_loop)?tloop_in$n_loop"
    "                          :1'b0);"
    "\nalways@(posedge clk) prev_idx$n_loop <= idx$n_loop;"
    "\nalways@(posedge clk) if (tstart$n_loop) begin"
    "\n  saved_ub$n_loop   <= ub$n_loop;"
    "\n  saved_step$n_loop <= step$n_loop;"
    "\nend"
    "\n"
    "\nwire $v_tloop = tloop_out$n_loop;";

/// This class is the verilog output printer for the HIR dialect. It walks
/// throught the module and prints the verilog in the 'out' variable.
class VerilogPrinter {
public:
  VerilogPrinter(llvm::formatted_raw_ostream &output)
      : replacer(module_out), out(output) {}

  LogicalResult printModule(ModuleOp op);

private:
  unsigned newValueNumber() { return nextValueNum++; }

  /// Handles string replacements in the generated code.
  class StringReplacerClass {
  public:
    StringReplacerClass(stringstream &module_out) : module_out(module_out) {}
    string insert(function<string()> func) {
      unsigned loc = replaceLocs.size();
      replaceLocs.push_back(make_pair(loc, func));
      return "$loc" + to_string(loc);
    }
    void push_frame() { topOfStack.push(replaceLocs.size()); }
    void pop_frame() {
      processReplacements();
      topOfStack.pop();
    }

  private:
    std::stringstream &module_out;
    void processReplacements() {
      string code = module_out.str();
      module_out.str(std::string());
      auto numEntries = replaceLocs.size() - topOfStack.top();
      for (int i = 0; i < numEntries; i++) {
        auto replacement = replaceLocs.back();
        replaceLocs.pop_back();
        unsigned loc = replacement.first;
        function<string()> getReplacementString = replacement.second;
        string replacementStr = getReplacementString();
        stringstream locSStream;
        locSStream << "$loc" << loc;
        findAndReplaceAll(code, locSStream.str(), replacementStr);
      }
      module_out << code;
      assert(replaceLocs.size() == topOfStack.top());
    }

    std::stack<unsigned> topOfStack;
    std::list<std::pair<unsigned, function<string()>>> replaceLocs;
  } replacer;

  class VerilogMapperClass {
  private:
    llvm::DenseMap<Value, VerilogValue *> mapValueToVerilogValuePtr;
    std::list<VerilogValue> verilogValueStore;
    std::stack<unsigned> topOfStack;

  public:
    void push_frame() { topOfStack.push(verilogValueStore.size()); }
    unsigned size() { return verilogValueStore.size(); }
    void pop_frame() {
      auto numDeletes = verilogValueStore.size() - topOfStack.top();
      for (int i = 0; i < numDeletes; i++) {
        verilogValueStore.pop_back();
      }
      topOfStack.pop();
    }

    VerilogValue *get_mut(Value v) {
      assert(v);
      assert(mapValueToVerilogValuePtr.find(v) !=
             mapValueToVerilogValuePtr.end());
      auto out = mapValueToVerilogValuePtr[v];
      assert(out->isInitialized());
      return out;
    }
    const VerilogValue get(Value v) { return *get_mut(v); }

    void insert_ptr(Value v, VerilogValue *vv) {
      mapValueToVerilogValuePtr[v] = vv;
      assert(mapValueToVerilogValuePtr[v]->isInitialized());
    }

    void insert(Value v, VerilogValue vv) {
      verilogValueStore.push_back(vv);
      insert_ptr(v, &verilogValueStore.back());
    }
    unsigned stack_level() { return topOfStack.size(); }
  } verilogMapper;

  void region_begin() {
    verilogMapper.push_frame();
    replacer.push_frame();
    module_out << "\n//{Region  (level = " << verilogMapper.stack_level()
               << ").\n";
    module_out << "//#VerilogValues = " << verilogMapper.size() << ".\n";
  }
  void region_end() {
    module_out << "\n// #VerilogValues = " << verilogMapper.size() << ".\n";
    module_out << "//}Region  (level = " << verilogMapper.stack_level()
               << ").\n";

    // replacer should process this frame's replacements before verilogMapper
    // pops its frame.
    replacer.pop_frame();
    verilogMapper.pop_frame();
  }

  LogicalResult printOperation(Operation *op, unsigned indentAmount = 0);

  // Individual op printers.
  LogicalResult printDefOp(DefOp op, unsigned indentAmount = 0);
  LogicalResult printConstantOp(hir::ConstantOp op, unsigned indentAmount = 0);
  LogicalResult printForOp(ForOp op, unsigned indentAmount = 0);
  LogicalResult printUnrollForOp(UnrollForOp op, unsigned indentAmount = 0);
  LogicalResult printMemReadOp(MemReadOp op, unsigned indentAmount = 0);
  LogicalResult printAddOp(hir::AddOp op, unsigned indentAmount = 0);
  LogicalResult printMemWriteOp(MemWriteOp op, unsigned indentAmount = 0);
  LogicalResult printReturnOp(hir::ReturnOp op, unsigned indentAmount = 0);
  LogicalResult printYieldOp(hir::YieldOp op, unsigned indentAmount = 0);
  LogicalResult printWireWriteOp(hir::WireWriteOp op,
                                 unsigned indentAmount = 0);
  LogicalResult printWireReadOp(hir::WireReadOp op, unsigned indentAmount = 0);
  LogicalResult printAllocOp(hir::AllocOp op, unsigned indentAmount = 0);
  LogicalResult printDelayOp(hir::DelayOp op, unsigned indentAmount = 0);
  LogicalResult printCallOp(hir::CallOp op, unsigned indentAmount = 0);

  // Helpers.
  string printDownCounter(stringstream &output, Value maxCount, Value tstart);
  LogicalResult printBody(Block &block, unsigned indentAmount = 0);
  LogicalResult printType(Type type);
  LogicalResult printTimeOffsets(const VerilogValue *timeVar);
  SmallVector<VerilogValue *, 4> convertToVerilog(OperandRange operands);
  SmallVector<VerilogValue *, 4> registerResults(ResultRange operands);

  stringstream module_out;
  llvm::formatted_raw_ostream &out;
  unsigned nextValueNum = 0;
  std::stack<string> yieldTarget;
}; // namespace

SmallVector<VerilogValue *, 4>
VerilogPrinter::convertToVerilog(OperandRange inputs) {
  SmallVector<VerilogValue *, 4> out;
  for (Value input : inputs) {
    out.push_back(verilogMapper.get_mut(input));
  }
  return out;
}

SmallVector<VerilogValue *, 4>
VerilogPrinter::registerResults(ResultRange results) {
  SmallVector<VerilogValue *, 4> out;
  for (Value result : results) {
    verilogMapper.insert(
        result, VerilogValue(result, "v" + to_string(newValueNumber())));
    out.push_back(verilogMapper.get_mut(result));
  }
  return out;
}

string VerilogPrinter::printDownCounter(stringstream &output, Value maxCount,
                                        Value tstart) {
  unsigned counterWidth = getBitWidth(maxCount.getType());

  VerilogValue *v_tstart = verilogMapper.get_mut(tstart);
  unsigned id_counter = newValueNumber();
  VerilogValue *v_maxCount = verilogMapper.get_mut(maxCount);
  string str_counter = "downCount" + to_string(id_counter);
  output << "reg[" << counterWidth - 1 << ":0]" << str_counter << "="
         << counterWidth << "'d0;\n";
  output << "always@(posedge clk) begin\n";
  output << "if(" << v_tstart->strWire() << ")\n";
  output << str_counter << " <= " << v_maxCount->strWire() << ";\n";
  output << "else\n";
  output << str_counter << " <= (" << str_counter << ">0)?(" << str_counter
         << "-1):0;\n";
  output << "end\n";
  return str_counter;
}

LogicalResult VerilogPrinter::printMemReadOp(MemReadOp op,
                                             unsigned indentAmount) {
  Value result = op.res();
  VerilogValue v_result(result, "v" + to_string(newValueNumber()));
  verilogMapper.insert(result, v_result);

  auto addr = convertToVerilog(op.addr());
  Value mem = op.mem();
  auto shape = mem.getType().dyn_cast<hir::MemrefType>().getShape();
  auto packing = mem.getType().dyn_cast<hir::MemrefType>().getPacking();

  assert(addr.size() == shape.size());

  Value tstart = op.tstart();
  VerilogValue *v_tstart = verilogMapper.get_mut(tstart);
  Value offset = op.offset();
  int delayValue = 0;
  if (offset) {
    VerilogValue *v_offset = verilogMapper.get_mut(offset);
    delayValue = offset ? v_offset->getIntegerConst() : 0;
  }
  VerilogValue *v_mem = verilogMapper.get_mut(mem);
  unsigned width_result = getBitWidth(result.getType());
  if (!packing.empty()) {
    // Address bus assignments.
    module_out << "assign "
               << v_mem->strMemrefAddrValid(addr, v_mem->numAccess(addr))
               << " = " << v_tstart->strDelayedWire(delayValue) << ";\n";

    module_out << "assign "
               << v_mem->strMemrefAddrInput(addr, v_mem->numAccess(addr))
               << " = {";
    for (int i = packing.size() - 1; i >= 0; i--) {
      auto p = addr.size() - 1 - packing[i];
      auto addrI = addr[p];
      if (i < packing.size() - 1)
        module_out << ", ";
      module_out << addrI->strConstOrWire();
    }
    module_out << "};\n";
  }

  // Read bus assignments.
  module_out << "wire" << v_result.strWireDecl() << " = "
             << v_mem->strMemrefRdData(addr) << ";\n";
  module_out << "assign " +
                    v_mem->strMemrefRdEnInput(addr, v_mem->numReads(addr))
             << " = " << v_tstart->strDelayedWire(delayValue) << ";\n\n";

  v_mem->incMemrefNumReads(addr);
  return success();
} // namespace

LogicalResult VerilogPrinter::printCallOp(hir::CallOp op,
                                          unsigned indentAmount) {
  ResultRange results = op.res();
  auto v_results = registerResults(results);
  StringRef callee = op.callee();
  OperandRange operands = op.operands();
  Value tstart = op.tstart();
  VerilogValue *v_tstart = verilogMapper.get_mut(tstart);
  Value offset = op.offset();
  int delayValue = 0;
  if (offset) {
    delayValue = verilogMapper.get(offset).getIntegerConst();
  }
  string calleeStr = callee.str();

  for (VerilogValue *v_result : v_results) {
    module_out << "wire " << v_result->strWireDecl() << ";\n";
  }
  module_out << calleeStr << " " << calleeStr << newValueNumber() << "(";
  bool firstArg = true;
  for (VerilogValue *v_result : v_results) {
    assert(!v_result->getType().isa<hir::MemrefType>());
    if (!firstArg) {
      module_out << ",\n";
    }
    firstArg = false;
    module_out << v_result->strConstOrWire();
  }
  for (Value operand : operands) {
    if (auto memrefType = operand.getType().dyn_cast<MemrefType>()) {
      Details::PortKind port = memrefType.getPort();
      VerilogValue *v_operand = verilogMapper.get_mut(operand);
      if (memrefType.getPacking().size() > 0) {
        if (!firstArg) {
          module_out << ",\n";
        }
        firstArg = false;
        module_out << v_operand->strMemrefAddr();
      }
      if (port == Details::PortKind::r || port == Details::PortKind::rw) {
        if (!firstArg) {
          module_out << ",\n";
        }
        firstArg = false;
        module_out << v_operand->strMemrefRdEn() << ",\n"
                   << v_operand->strMemrefRdData();
      }
      if (port == Details::PortKind::w || port == Details::PortKind::rw) {
        if (!firstArg) {
          module_out << ",\n";
        }
        firstArg = false;
        module_out << v_operand->strMemrefWrEn() << ",\n"
                   << v_operand->strMemrefWrData();
      }
    } else {
      if (!firstArg) {
        module_out << ",\n";
      }
      firstArg = false;
      VerilogValue *v_operand = verilogMapper.get_mut(operand);
      module_out << v_operand->strConstOrWire();
    }
  }
  if (!firstArg) {
    module_out << ",\n";
  }
  module_out << v_tstart->strDelayedWire(delayValue);
  module_out << ",\n";
  module_out << "clk\n";

  module_out << ");\n";
  return success();
}
LogicalResult VerilogPrinter::printDelayOp(hir::DelayOp op,
                                           unsigned indentAmount) {
  // contract:
  // delay > 0
  // delay constant

  Value input = op.input();
  VerilogValue *v_input = verilogMapper.get_mut(input);
  Value delay = op.delay();
  VerilogValue *v_delay = verilogMapper.get_mut(delay);
  Value result = op.res();
  verilogMapper.insert(result,
                       VerilogValue(result, "v" + to_string(newValueNumber())));
  VerilogValue *v_result = verilogMapper.get_mut(result);
  v_result->strWireDecl();
  // Propagate constant value.
  if (v_input->isIntegerConst()) {
    v_result->setIntegerConst(v_input->getIntegerConst());
  }
  unsigned input_bitwidth = getBitWidth(input.getType());
  string str_shiftreg = "shiftreg" + to_string(newValueNumber());
  module_out << "reg[" << input_bitwidth - 1 << ":0]" << str_shiftreg << "["
             << v_delay->strConstOrError() << ":0] = "
             << "'{default:0};\n";

  module_out << "always@(*) " << str_shiftreg
             << "[0] = " << v_input->strConstOrWire() << ";\n";
  if (v_delay->getIntegerConst() > 0) {
    module_out << "always@(posedge clk) " << str_shiftreg << "["
               << v_delay->strConstOrError() << ":1] = " << str_shiftreg << "["
               << v_delay->strConstOrError(-1) << ":0];\n";
  }
  module_out << "wire " << v_result->strWireDecl() << " = " << str_shiftreg
             << "[" << v_delay->strConstOrError() << "];\n";
  return success();
}

LogicalResult VerilogPrinter::printAllocOp(hir::AllocOp op,
                                           unsigned indentAmount) {
  Value result = op.res();
  unsigned id_result = newValueNumber();
  VerilogValue v_result(result, "v" + to_string(id_result));
  verilogMapper.insert(result, v_result);
  module_out << "wire " << v_result.strWireDecl() << ";\n";
  return success();
}

LogicalResult VerilogPrinter::printWireReadOp(hir::WireReadOp op,
                                              unsigned indentAmount) {

  Value result = op.res();
  unsigned id_result = newValueNumber();
  VerilogValue v_result(result, "v" + to_string(id_result));
  verilogMapper.insert(result, v_result);

  Value wire = op.wire();
  VerilogValue *v_wire = verilogMapper.get_mut(wire);
  SmallVector<VerilogValue *, 4> addr = convertToVerilog(op.addr());
  Value tstart = op.tstart();
  Value offset = op.offset();
  auto shape = wire.getType().dyn_cast<hir::WireType>().getShape();
  string arrayAccessStr = "";
  for (auto idx : addr) {
    if (idx->isIntegerConst()) {
      arrayAccessStr += "[" + to_string(idx->getIntegerConst()) + "]";
    } else {
      arrayAccessStr +=
          "[" + idx->strWire() + " /*ERROR: Expected integer constant.*/]";
    }
  }
  module_out << "wire " << v_result.strWireDecl() << " = " << v_wire->strWire()
             << arrayAccessStr << ";\n";
}

LogicalResult VerilogPrinter::printWireWriteOp(hir::WireWriteOp op,
                                               unsigned indentAmount) {
  Value value = op.value();
  VerilogValue *v_value = verilogMapper.get_mut(value);
  Value wire = op.wire();
  VerilogValue *v_wire = verilogMapper.get_mut(wire);
  auto addr = convertToVerilog(op.addr());
  Value tstart = op.tstart();
  auto offset = op.offset();
  auto shape = wire.getType().dyn_cast<hir::WireType>().getShape();
  string arrayAccessStr = "";

  for (auto idx : addr) {
    if (idx->isIntegerConst()) {
      arrayAccessStr += "[" + to_string(idx->getIntegerConst()) + "]";
    } else {
      arrayAccessStr +=
          "[" + idx->strWire() + " /*ERROR: Expected integer constant.*/]";
    }
  }

  module_out << "assign " << v_wire->strWire() << arrayAccessStr << " = "
             << v_value->strConstOrWire() << ";\n";
  return success();
}

LogicalResult VerilogPrinter::printMemWriteOp(MemWriteOp op,
                                              unsigned indentAmount) {
  auto addr = convertToVerilog(op.addr());
  Value mem = op.mem();
  Value value = op.value();
  Value tstart = op.tstart();
  VerilogValue *v_tstart = verilogMapper.get_mut(tstart);
  Value offset = op.offset();
  VerilogValue *v_offset = verilogMapper.get_mut(offset);

  auto shape = mem.getType().dyn_cast<hir::MemrefType>().getShape();
  auto packing = mem.getType().dyn_cast<hir::MemrefType>().getPacking();
  if (shape.size() != addr.size()) {
    return emitError(op.getLoc(),
                     "Dimension mismatch. Number of addresses is wrong.");
  }

  VerilogValue *v_value = verilogMapper.get_mut(value);
  VerilogValue *v_mem = verilogMapper.get_mut(mem);

  int delayValue = offset ? v_offset->getIntegerConst() : 0;

  if (!packing.empty()) {
    // Address bus assignments.
    module_out << "assign "
               << v_mem->strMemrefAddrValid(addr, v_mem->numAccess(addr))
               << " = " << v_tstart->strDelayedWire(delayValue) << ";\n";

    module_out << "assign "
               << v_mem->strMemrefAddrInput(addr, v_mem->numAccess(addr))
               << " = {";
    for (int i = packing.size() - 1; i >= 0; i--) {
      auto addrI = addr[i];
      if (i < packing.size() - 1)
        module_out << ", ";
      module_out << addrI->strWire();
    }
    module_out << "};\n";
  }

  module_out << "assign "
             << v_mem->strMemrefWrEnInput(addr, v_mem->numWrites(addr)) << " = "
             << v_tstart->strDelayedWire(delayValue) << ";\n";
  module_out << "assign "
             << v_mem->strMemrefWrDataValid(addr, v_mem->numWrites(addr))
             << " = " << v_tstart->strDelayedWire(delayValue) << ";\n";
  module_out << "assign "
             << v_mem->strMemrefWrDataInput(addr, v_mem->numWrites(addr))
             << " = " << v_value->strConstOrWire() << ";\n\n";

  v_mem->incMemrefNumWrites(addr);
  return success();
}

LogicalResult VerilogPrinter::printAddOp(hir::AddOp op, unsigned indentAmount) {
  // TODO: Handle signed and unsigned integers of unequal width using sign
  // extension.
  Value result = op.res();
  unsigned id_result = newValueNumber();
  VerilogValue v_result(result, "v" + to_string(id_result));

  const VerilogValue v_left = verilogMapper.get(op.left());
  const VerilogValue v_right = verilogMapper.get(op.right());
  Type resultType = result.getType();
  if (resultType.isa<hir::ConstType>()) {
    module_out << "wire " << v_result.strWireDecl() << " = "
               << v_left.strConstOrError() << " + " << v_right.strConstOrError()
               << ";\n";
    v_result.setIntegerConst(v_left.getIntegerConst() +
                             v_right.getIntegerConst());
  } else {
    module_out << "wire " << v_result.strWireDecl() << " = "
               << v_left.strConstOrWire() << " + " << v_right.strConstOrWire()
               << ";\n";
    if (v_left.isIntegerConst() && v_right.isIntegerConst()) {
      v_result.setIntegerConst(v_left.getIntegerConst() +
                               v_right.getIntegerConst());
    }
  }
  verilogMapper.insert(result, v_result);
  return success();
}

LogicalResult VerilogPrinter::printReturnOp(hir::ReturnOp op,
                                            unsigned indentAmount) {
  assert(op.operands().size() == 0);
  module_out << "\n//hir.return => NOP\n"; // TODO
}

LogicalResult VerilogPrinter::printConstantOp(hir::ConstantOp op,
                                              unsigned indentAmount) {
  auto result = op.res();
  unsigned id_result = newValueNumber();
  VerilogValue v_result(result, "v" + to_string(id_result));
  int value = op.value().getLimitedValue();
  v_result.setIntegerConst(value);
  verilogMapper.insert(result, v_result);

  module_out << "//wire " << v_result.strWireDecl();
  module_out << " = " << v_result.getBitWidth() << "'d" << value << ";\n";
  // module_out << "//CHECK: " << verilogMapper.get(result).strConstOrWire();
  return success();
} // namespace

LogicalResult VerilogPrinter::printForOp(hir::ForOp op, unsigned indentAmount) {
  Value lb = op.lb();
  Value ub = op.ub();
  Value step = op.step();
  Value tstart = op.tstart();
  Value tstep = op.tstep();
  Value idx = op.getInductionVar();
  Value tloop = op.getIterTimeVar();

  auto id_loop = newValueNumber();
  yieldTarget.push("assign tloop_in" + to_string(id_loop));

  const VerilogValue v_lb = verilogMapper.get(op.lb());
  const VerilogValue v_ub = verilogMapper.get(op.ub());
  const VerilogValue v_step = verilogMapper.get(op.step());
  const VerilogValue v_tstart = verilogMapper.get(op.tstart());
  VerilogValue v_idx = VerilogValue(idx, "idx" + to_string(id_loop));
  verilogMapper.insert(idx, v_idx);
  verilogMapper.insert(tloop,
                       VerilogValue(tloop, "tloop" + to_string(id_loop)));
  VerilogValue *v_tloop = verilogMapper.get_mut(tloop);

  unsigned width_lb = getBitWidth(op.lb().getType());
  unsigned width_ub = getBitWidth(op.ub().getType());
  unsigned width_step = getBitWidth(op.step().getType());
  unsigned width_tstep = op.tstep() ? getBitWidth(op.tstep().getType()) : 0;
  unsigned width_idx = getBitWidth(idx.getType());

  stringstream loopCounterStream;
  loopCounterStream << loopCounterTemplate;
  if (tstep) {
    string v_delay_counter = printDownCounter(loopCounterStream, tstep, tloop);
    loopCounterStream << "assign tloop_in$n_loop = (" << v_delay_counter
                      << " == 1);\n";
  }

  string loopCounterString = loopCounterStream.str();
  findAndReplaceAll(loopCounterString, "$n_loop", to_string(id_loop));
  findAndReplaceAll(loopCounterString, "$msb_idx", to_string(width_idx - 1));
  findAndReplaceAll(loopCounterString, "$v_lb", v_lb.strConstOrWire());
  findAndReplaceAll(loopCounterString, "$msb_lb", to_string(width_lb - 1));
  findAndReplaceAll(loopCounterString, "$v_ub", v_ub.strConstOrWire());
  findAndReplaceAll(loopCounterString, "$msb_ub", to_string(width_ub - 1));
  findAndReplaceAll(loopCounterString, "$v_step", v_step.strConstOrWire());
  findAndReplaceAll(loopCounterString, "$msb_step", to_string(width_step - 1));
  findAndReplaceAll(loopCounterString, "$v_tstart", v_tstart.strWire());
  findAndReplaceAll(loopCounterString, "$width_tstep", to_string(width_tstep));
  findAndReplaceAll(loopCounterString, "$v_tloop", v_tloop->strWire());
  module_out << "\n//{ Loop" << id_loop << "\n";
  module_out << loopCounterString;
  module_out << "\n//Loop" << id_loop << " body\n";
  printTimeOffsets(v_tloop);
  printBody(op.getLoopBody().front(), indentAmount);
  module_out << "\n//} Loop" << id_loop << "\n";
  yieldTarget.pop();
  return success();
}

LogicalResult VerilogPrinter::printUnrollForOp(UnrollForOp op,
                                               unsigned indentAmount) {
  Value tlast = op.tlast(); // output
  int lb = op.lb().getLimitedValue();
  int ub = op.ub().getLimitedValue();
  int step = op.step().getLimitedValue();
  Value tstart = op.tstart();
  VerilogValue *v_tstart = verilogMapper.get_mut(tstart);
  Value tloop = op.getIterTimeVar();
  Value idx = op.getInductionVar();
  verilogMapper.insert_ptr(tloop, v_tstart);
  auto id_loop = newValueNumber();
  verilogMapper.insert(
      idx, VerilogValue(
               idx,
               "idx" + to_string(
                           id_loop))); // FIXME remove name of the VerilogValue.
  VerilogValue v_tloop;
  for (int i = lb; i < ub; i += step) {
    region_begin();
    auto id_tloop = newValueNumber();

    module_out << "\n//{ Unrolled body " << i
               << " of loop" + to_string(id_loop) + ".\n";
    VerilogValue *v_idx = verilogMapper.get_mut(idx);
    v_idx->setIntegerConst(i);
    module_out << "//DEBUG: " << v_idx->strConstOrWire() << ", expected " << i
               << "\n";
    if (i > lb) {
      printTimeOffsets(&v_tloop);
      verilogMapper.insert_ptr(tloop, &v_tloop);
    }
    VerilogValue next_v_tloop =
        VerilogValue(tloop, "tloop" + to_string(id_tloop));
    yieldTarget.push("wire " + next_v_tloop.strWire());

    // At first iteration, tloop -> v_tstart, whose time offsets are already
    // printed.
    auto status = printBody(op.getLoopBody().front(), indentAmount);

    yieldTarget.pop();
    module_out << "\n//} Unrolled body " << i
               << " of loop" + to_string(id_loop) + ".\n";

    module_out << "//DEBUG: " << verilogMapper.get_mut(idx)->strConstOrWire()
               << ", expected " << i << "\n";
    assert(status.value != LogicalResult::Failure);
    region_end();
    v_tloop = next_v_tloop;
  }

  module_out << "\n//{ Assign tlast of prev UnrollForLoop\n";
  verilogMapper.insert(tlast,
                       VerilogValue(tlast, "t" + to_string(newValueNumber())));
  const VerilogValue *v_tlast = verilogMapper.get_mut(tlast);
  module_out << "wire " << v_tlast->strWireDecl() << ";\n";
  module_out << "assign " << v_tlast->strWire() << " = "
             << verilogMapper.get(tloop).strWire() << ";\n";
  printTimeOffsets(v_tlast);

  module_out << "\n//} Assign tlast of prev UnrollForLoop";
  return success();
}

string formattedOp(Operation *op, string opName) {
  string out;
  string locStr;
  llvm::raw_string_ostream locOstream(locStr);
  op->getLoc().print(locOstream);
  out = opName + " at " + locStr + "\n";
  return out;
}
LogicalResult VerilogPrinter::printOperation(Operation *inst,
                                             unsigned indentAmount) {
  if (auto op = dyn_cast<hir::ConstantOp>(inst)) {
    module_out << "\n//" << formattedOp(inst, "ConstantOp");
    return printConstantOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::CallOp>(inst)) {
    module_out << "\n//" << formattedOp(inst, "CallOp");
    return printCallOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::AllocOp>(inst)) {
    module_out << "\n//" << formattedOp(inst, "AllocOp");
    return printAllocOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::DelayOp>(inst)) {
    module_out << "\n//" << formattedOp(inst, "DelayOp");
    return printDelayOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::ForOp>(inst)) {
    module_out << "\n//" << formattedOp(inst, "ForOp");
    return printForOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::UnrollForOp>(inst)) {
    module_out << "\n//" << formattedOp(inst, "UnrollForOp");
    return printUnrollForOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::ReturnOp>(inst)) {
    module_out << "\n//" << formattedOp(inst, "ReturnOp");
    return printReturnOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::MemReadOp>(inst)) {
    module_out << "\n//" << formattedOp(inst, "MemReadOp");
    return printMemReadOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::MemWriteOp>(inst)) {
    module_out << "\n//" << formattedOp(inst, "MemWriteOp");
    return printMemWriteOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::WireReadOp>(inst)) {
    module_out << "\n//" << formattedOp(inst, "WireReadOp");
    return printWireReadOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::WireWriteOp>(inst)) {
    module_out << "\n//" << formattedOp(inst, "WireWriteOp");
    return printWireWriteOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::AddOp>(inst)) {
    module_out << "\n//" << formattedOp(inst, "AddOp");
    return printAddOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::YieldOp>(inst)) {
    module_out << "\n//" << formattedOp(inst, "YieldOp");
    return printYieldOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::TerminatorOp>(inst)) {
    module_out << "\n//TerminatorOp\n";
    // Do nothing.
  } else {
    return emitError(inst->getLoc(), "Unsupported Operation for codegen!");
  }
} // namespace

LogicalResult VerilogPrinter::printDefOp(DefOp op, unsigned indentAmount) {
  // An EntityOp always has a single block.
  Block &entryBlock = op.getBody().front();
  auto args = entryBlock.getArguments();
  assert(op.getNumResults() == 0);
  // Print the module signature.
  FunctionType funcType = op.getType();
  module_out << "module " << op.getName().str() << "(";
  for (int i = 0; i < args.size(); i++) {
    Value arg = args[i];
    Type argType = arg.getType();

    // Print verilog.
    if (i == 0)
      module_out << "\n";

    if (i > 0)
      module_out << ",\n";

    if (argType.isa<hir::TimeType>()) {
      unsigned id_tstart = newValueNumber();
      /*tstart parameter of func does not use valuenumber*/
      auto tstart = VerilogValue(arg, "tstart");
      verilogMapper.insert(arg, tstart);

      module_out << "//TimeType.\n";
      module_out << "input wire " << tstart.strWireDecl();
    } else if (argType.isa<IntegerType>()) {
      unsigned id_int_arg = newValueNumber();
      auto v_int_arg = VerilogValue(arg, "v" + to_string(id_int_arg));
      verilogMapper.insert(arg, v_int_arg);

      module_out << "//IntegerType.\n";
      unsigned bitwidth = getBitWidth(argType);
      module_out << "input wire" << v_int_arg.strWireDecl();
    } else if (argType.isa<MemrefType>()) {
      unsigned id_memref_arg = newValueNumber();
      auto v_memref_arg = VerilogValue(arg, "v" + to_string(id_memref_arg));
      verilogMapper.insert(arg, v_memref_arg);
      module_out << v_memref_arg.strMemrefArgDecl();
    } else {
      return emitError(op.getLoc(), "Unsupported argument type!");
    }

    if (i == args.size())
      module_out << "\n";
  }
  module_out << ",\n//Clock.\ninput wire clk\n);\n\n";

  region_begin();
  for (auto arg : args) {
    if (arg.getType().isa<hir::MemrefType>()) {
      string loc = replacer.insert([=]() -> string {
        VerilogValue *v_arg = verilogMapper.get_mut(arg);
        return v_arg->strMemrefSelDecl();
      });
      module_out << loc << "\n";
    }
  }

  const VerilogValue *v_tstart = verilogMapper.get_mut(args.back());
  printTimeOffsets(v_tstart);
  printBody(entryBlock);

  region_end();

  module_out << "endmodule\n";

  out << module_out.str();
  return success();
}

LogicalResult VerilogPrinter::printYieldOp(hir::YieldOp op,
                                           unsigned indentAmount) {
  auto operands = op.operands();
  VerilogValue *v_tstart = verilogMapper.get_mut(op.tstart());
  Value offset = op.offset();
  const VerilogValue v_offset = verilogMapper.get(offset);
  int delayValue = v_offset.getIntegerConst();
  if (!operands.empty())
    emitError(op.getLoc(), "YieldOp codegen does not support arguments yet!");
  string tloop_in = yieldTarget.top();
  module_out << tloop_in << " = " << v_tstart->strDelayedWire(delayValue)
             << ";\n";
  return success();
}

LogicalResult VerilogPrinter::printTimeOffsets(const VerilogValue *v_timeVar) {
  module_out << "//printTimeOffset\n";
  string loc = replacer.insert(
      [=]() -> string { return to_string(v_timeVar->getMaxDelay()); });

  module_out << "reg " << v_timeVar->strDelayedWire() << "[" << loc
             << ":0] = '{default:0} ;\n";
  module_out << "always@(*) " << v_timeVar->strDelayedWire()
             << "[0] = " << v_timeVar->strWire() << ";\n";
  string str_i = "i" + to_string(newValueNumber());
  module_out << "generate\n";
  module_out << "genvar " << str_i << ";\n";
  module_out << "\nfor(" << str_i << " = 1; " << str_i << "<= " << loc << "; "
             << str_i << "= " << str_i << " + 1) begin\n";
  module_out << "always@(posedge clk) begin\n";
  module_out << v_timeVar->strDelayedWire() << "[" << str_i
             << "] <= " << v_timeVar->strDelayedWire() << "[" << str_i
             << "-1];\n";
  module_out << "end\n";
  module_out << "end\n";
  module_out << "endgenerate\n\n";
}

LogicalResult VerilogPrinter::printBody(Block &block, unsigned indentAmount) {

  region_begin();

  // Print the operations within the entity.
  for (auto iter = block.begin(); iter != block.end(); ++iter) {
    auto error = printOperation(&(*iter), 4);
    if (failed(error)) {
      return error;
    }
  }

  region_end();
}

LogicalResult VerilogPrinter::printModule(ModuleOp module) {
  module_out << "`default_nettype none\n";
  module_out << "`include \"helper.sv\"\n";
  WalkResult result = module.walk([this](DefOp defOp) -> WalkResult {
    if (!printDefOp(defOp).value)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  // if printing of a single operation failed, fail the whole translation.
  return failure(result.wasInterrupted());
}
} // namespace.

LogicalResult hir::printVerilog(ModuleOp module, raw_ostream &os) {
  llvm::formatted_raw_ostream out(os);
  VerilogPrinter printer(out);
  return printer.printModule(module);
}

void hir::registerHIRToVerilogTranslation() {
  TranslateFromMLIRRegistration registration(
      "hir-to-verilog", [](ModuleOp module, raw_ostream &output) {
        return printVerilog(module, output);
      });
}
