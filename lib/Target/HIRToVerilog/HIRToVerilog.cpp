//=========- HIRToVerilog.cpp - Verilog Printer ---------------------------===//
//
// This is the main HIR to Verilog Printer implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/HIR.h"
#include "circt/Target/HIRToVerilog/HIRToVerilog.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormattedStream.h"

#include <functional>
#include <string>

using namespace mlir;
using namespace hir;

// TODO: Printer should never fail. All the checks we are doing here should
// pass. We should implement these checks in op's verify function.
// TODO: Replace recursive function calls.

namespace {
std::string loopCounterTemplate =
    "\nwire[$msb_lb:0] lb$n_loop = v$n_lb;"
    "\nwire[$msb_ub:0] ub$n_loop = v$n_ub;"
    "\nwire[$msb_step:0] step$n_loop = v$n_step;"
    "\nwire tstart$n_loop = t$n_tstart;"
    "\n"
    "\nreg[$msb_idx:0] prev_idx$n_loop;"
    "\nreg[$msb_ub:0] saved_ub$n_loop;"
    "\nreg[$msb_step:0] saved_step$n_loop;"
    "\nwire[$msb_idx:0] idx$n_loop = tstart$n_loop? lb$n_loop "
    "\n                              : tloop_in$n_loop ? "
    "(prev_idx$n_loop+saved_step)"
    "\n: prev_idx$n_loop;"
    "\nwire tloop_out$n_loop = tstart$n_loop"
    "\n|| ((idx$n_loop < saved_ub$n_loop)?tloop_in$n_loop"
    "\n                          :1'b0);"
    "\nalways@(posedge clk) saved_idx$n_loop <= idx$n_loop;"
    "\nalways@(posedge clk) if (tstart$n_loop) begin"
    "\n  saved_ub$n_loop   <= ub$n_loop;"
    "\n  saved_step <= step$n_loop;"
    "\nend"
    "\n"
    "\nwire t$n_loop = tloop_out$n_loop;"
    "\nwire[$msb_idx:0] v$n_loop = idx$n_loop;\n";

static void findAndReplaceAll(std::string &data, std::string toSearch,
                              std::string replaceStr) {
  // Get the first occurrence.
  size_t pos = data.find(toSearch);
  // Repeat till end is reached.
  while (pos != std::string::npos) {
    // Replace this occurrence of Sub String.
    data.replace(pos, toSearch.size(), replaceStr);
    // Get the next occurrence from the current position.
    pos = data.find(toSearch, pos + replaceStr.size());
  }
}

static bool getBitWidth(Type ty, unsigned &bitwidth) {
  if (auto intTy = ty.dyn_cast<IntegerType>()) {
    bitwidth = intTy.getWidth();
    return true;
  } else if (auto memrefTy = ty.dyn_cast<MemrefType>()) {
    return getBitWidth(memrefTy.getElementType(), bitwidth);
  } else if (auto constTy = ty.dyn_cast<ConstType>()) {
    return getBitWidth(constTy.getElementType(), bitwidth);
  }
  return false;
}

static unsigned calcAddrWidth(hir::MemrefType memrefTy) {
  // FIXME: Currently we assume that all dims are power of two.
  auto shape = memrefTy.getShape();
  auto elementType = memrefTy.getElementType();
  auto packing = memrefTy.getPacking();
  unsigned elementWidth;
  getBitWidth(elementType, elementWidth);
  int max_dim = shape.size() - 1;
  unsigned addrWidth = 0;
  for (auto dim : packing) {
    // dim0 is last in shape.
    int dim_size = shape[max_dim - dim];
    float log_size = log2(dim_size);
    addrWidth += ceil(log_size);
  }
  return addrWidth;
}

/// This class is the verilog output printer for the HIR dialect. It walks
/// throught the module and prints the verilog in the 'out' variable.
class VerilogPrinter {
public:
  VerilogPrinter(llvm::formatted_raw_ostream &output) : out(output) {}

  LogicalResult printModule(ModuleOp op);

private:
  unsigned newValueNumber() { return nextValueNum++; }
  LogicalResult printOperation(Operation *op, unsigned indentAmount = 0);

  // Individual op printers.
  LogicalResult printDefOp(DefOp op, unsigned indentAmount = 0);
  LogicalResult printConstantOp(hir::ConstantOp op, unsigned indentAmount = 0);
  LogicalResult printForOp(ForOp op, unsigned indentAmount = 0);
  LogicalResult printAddOp(AddOp op, unsigned indentAmount = 0);
  LogicalResult printMemReadOp(MemReadOp op, unsigned indentAmount = 0);
  LogicalResult printMemWriteOp(MemWriteOp op, unsigned indentAmount = 0);
  LogicalResult printReturnOp(hir::ReturnOp op, unsigned indentAmount = 0);

  // Helpers.
  LogicalResult printBody(Block &block, unsigned indentAmount = 0);
  LogicalResult printMemrefAddrSelector(Value mem, unsigned indentAmount = 0);
  LogicalResult printType(Type type);
  LogicalResult processReplacements(std::string &code);
  LogicalResult printTimeOffsets(Value timeVar);

  std::stringstream module_out;
  llvm::formatted_raw_ostream &out;
  unsigned nextValueNum = 0;
  llvm::DenseMap<Value, unsigned> mapValueToNum;
  llvm::DenseMap<Value, int> mapConstToInt;
  llvm::DenseMap<Value, unsigned> mapValueToSelWidth;
  llvm::DenseMap<Value, unsigned> mapTimeToMaxOffset;

  SmallVector<std::pair<unsigned, std::function<std::string()>>, 4> replaceLocs;
};

LogicalResult VerilogPrinter::printMemReadOp(MemReadOp op,
                                             unsigned indentAmount) {
  OperandRange addr = op.addr();
  mlir::Value result = op.res();
  mlir::Value mem = op.mem();
  unsigned n_mem = mapValueToNum[mem];
  mlir::Value tstart = op.tstart();
  mlir::Value offset = op.offset();
  int offsetValue = 0;
  if (offset) {
    auto offsetType = offset.getType();
    if (!offsetType.isa<hir::ConstType>())
      return emitError(op.getLoc(), "offset must be a const<int>");
    Type elementType = offsetType.dyn_cast<hir::ConstType>().getElementType();
    if (!elementType.isa<IntegerType>())
      return emitError(op.getLoc(), "offset must be a const<int>");
    offsetValue = mapConstToInt[offset];
  }

  unsigned oldMaxOffset = mapTimeToMaxOffset[tstart];
  mapTimeToMaxOffset[tstart] =
      (offsetValue > oldMaxOffset) ? offsetValue : oldMaxOffset;
  module_out << "assign v_addr" << n_mem << "_sel_valid["
             << mapValueToSelWidth[mem] << "] = "
             << "t" << mapValueToNum[tstart] << "offset[" << offsetValue
             << "];\n";
  module_out << "assign v_addr" << n_mem << "_sel_data["
             << mapValueToSelWidth[mem] << "] = {";
  int i = 0;
  for (auto addrI : addr) {
    if (i++ > 0)
      module_out << ", ";
    module_out << "v" << mapValueToNum[addrI];
  }
  module_out << "};\n";
  unsigned n_result = newValueNumber();
  mapValueToNum.insert(std::make_pair(result, n_result));
  module_out << "assign v" << n_result << " = v_data_rd" << n_mem << "["
             << mapValueToSelWidth[mem] << "];\n\n";

  // We now have one more selector input so incr the width.
  mapValueToSelWidth[mem] += 1;
  return success();
}

LogicalResult VerilogPrinter::printReturnOp(hir::ReturnOp op,
                                            unsigned indentAmount) {
  module_out << "\n//hir.return => NOP\n";
}

LogicalResult VerilogPrinter::printConstantOp(hir::ConstantOp op,
                                              unsigned indentAmount) {
  auto result = op.res();
  int value = op.value().getLimitedValue();
  mapConstToInt.insert(std::make_pair(result, value));
  if (auto intTy = result.getType().dyn_cast<IntegerType>()) {
    unsigned valueNumber = newValueNumber();
    mapValueToNum.insert(std::make_pair(result, valueNumber));
    module_out << "wire [" << intTy.getWidth() - 1 << " : 0] v" << valueNumber;
    module_out << "=" << intTy.getWidth() << "'d" << value << ";\n";
    return success();
  } else if (auto constTy = result.getType().dyn_cast<ConstType>()) {
    if (auto intTy = constTy.getElementType().dyn_cast<IntegerType>()) {
      unsigned valueNumber = newValueNumber();
      mapValueToNum.insert(std::make_pair(result, valueNumber));
      module_out << "wire [" << intTy.getWidth() - 1 << " : 0] v"
                 << valueNumber;
      module_out << "=" << intTy.getWidth() << "'d" << value << ";\n";
      return success();
    }
  }
  return emitError(op.getLoc(), "Only integer or const<integer> are supported");
}

LogicalResult VerilogPrinter::printForOp(hir::ForOp op, unsigned indentAmount) {
  unsigned width_lb;
  unsigned width_ub;
  unsigned width_step;
  unsigned width_tstep;
  unsigned width_idx;
  bool gotBitwidth = getBitWidth(op.lb().getType(), width_lb) &&
                     getBitWidth(op.ub().getType(), width_ub) &&
                     getBitWidth(op.step().getType(), width_step);
  if (!gotBitwidth)
    return emitError(
        op.getLoc(),
        "lb, ub and step must be either integer or const<integer>!");

  if (op.tstep())
    if (getBitWidth(op.tstep().getType(), width_tstep) == false)
      return emitError(op.getLoc(),
                       "tstep must be either integer or const<integer>!");

  unsigned n_lb = mapValueToNum[op.lb()];
  unsigned n_ub = mapValueToNum[op.ub()];
  unsigned n_step = mapValueToNum[op.step()];
  int n_tstep = op.tstep() ? mapValueToNum[op.tstep()] : -1;
  unsigned n_tstart = mapValueToNum[op.tstart()];
  Value idx = op.getInductionVar();
  if (!getBitWidth(idx.getType(), width_idx))
    return emitError(op.getLoc(), "loop induction var must be integer type");
  Value tloop = op.getIterTimeVar();
  unsigned n_loop = newValueNumber();
  mapValueToNum.insert(std::make_pair(idx, n_loop));
  mapValueToNum.insert(std::make_pair(tloop, n_loop));
  if (auto idx_intTy = idx.getType().dyn_cast<IntegerType>()) {
    module_out << "reg[" << idx_intTy.getWidth() << "] counter" << n_loop
               << ";\n";
  } else {
    return emitError(op.getLoc(), "for loop induction var must be an integer");
  }
  std::stringstream loopCounterStream;
  loopCounterStream << loopCounterTemplate;
  if (n_tstep >= 0)
    loopCounterStream << "assign tloop_in$n_loop = "
                         "count_delay#(.WIDTH_input(1),.WIDTH_DELAY($width_"
                         "tstep))(tloop_out$n_loop, v$n_tstep);\n";

  std::string loopCounterString = loopCounterStream.str();
  findAndReplaceAll(loopCounterString, "$n_loop", std::to_string(n_loop));
  findAndReplaceAll(loopCounterString, "$msb_idx",
                    std::to_string(width_idx - 1));
  findAndReplaceAll(loopCounterString, "$n_lb", std::to_string(n_lb));
  findAndReplaceAll(loopCounterString, "$msb_lb", std::to_string(width_lb - 1));
  findAndReplaceAll(loopCounterString, "$n_ub", std::to_string(n_ub));
  findAndReplaceAll(loopCounterString, "$msb_ub", std::to_string(width_ub - 1));
  findAndReplaceAll(loopCounterString, "$n_step", std::to_string(n_step));
  findAndReplaceAll(loopCounterString, "$msb_step",
                    std::to_string(width_step - 1));
  findAndReplaceAll(loopCounterString, "$n_tstart", std::to_string(n_tstart));
  if (n_tstep >= 0)
    findAndReplaceAll(loopCounterString, "$n_tstep", std::to_string(n_tstep));
  findAndReplaceAll(loopCounterString, "$width_tstep",
                    std::to_string(width_tstep));
  module_out << "\n//Loop" << n_loop << " begin\n";
  module_out << loopCounterString;
  module_out << "\n//Loop" << n_loop << " body\n";
  printTimeOffsets(tloop);
  printBody(op.getLoopBody().front(), indentAmount);
  module_out << "\n//Loop" << n_loop << " end\n";
  return success();
}

LogicalResult VerilogPrinter::printOperation(Operation *inst,
                                             unsigned indentAmount) {
  if (auto op = dyn_cast<hir::ConstantOp>(inst)) {
    return printConstantOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::ForOp>(inst)) {
    return printForOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::ReturnOp>(inst)) {
    return printReturnOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::MemReadOp>(inst)) {
    return printMemReadOp(op, indentAmount);
  } else {
    return emitError(inst->getLoc(), "Unsupported Operation!");
  }
}

LogicalResult VerilogPrinter::printDefOp(DefOp op, unsigned indentAmount) {
  // An EntityOp always has a single block.
  Block &entryBlock = op.getBody().front();
  auto args = entryBlock.getArguments();
  // Print the module signature.
  FunctionType funcType = op.getType();
  module_out << "module " << op.getName().str() << "(";
  for (int i = 0; i < args.size(); i++) {
    unsigned bitwidth = 0;
    Value arg = args[i];
    Type argType = arg.getType();
    unsigned valueNumber = newValueNumber();
    mapValueToNum.insert(std::make_pair(arg, valueNumber));

    // Print verilog.
    if (i == 0)
      module_out << "\n";

    if (i > 0)
      module_out << ",\n";

    if (argType.isa<hir::TimeType>()) {
      module_out << "input wire t" << valueNumber;
    } else if (argType.isa<IntegerType>()) {
      getBitWidth(argType, bitwidth);
      module_out << "input wire[" << bitwidth - 1 << ":0]  ";
      module_out << "v" << valueNumber;
    } else if (argType.isa<MemrefType>()) {
      MemrefType memrefTy = argType.dyn_cast<MemrefType>();
      hir::Details::PortKind port = memrefTy.getPort();
      unsigned addrWidth = calcAddrWidth(memrefTy);
      module_out << "output wire[" << addrWidth - 1 << ":0] v_addr"
                 << valueNumber;
      module_out << ",\noutput wire v_addr" << valueNumber << "_valid";
      if (port == hir::Details::r || port == hir::Details::rw) {
        bool bitwidth_valid = getBitWidth(argType, bitwidth);
        if (!bitwidth_valid)
          return emitError(op.getLoc(), "Unsupported argument type!");
        module_out << ",\n"
                   << "input wire[" << bitwidth - 1 << ":0]  ";
        module_out << "v_data_rd" << valueNumber;
      }
      if (port == hir::Details::w || port == hir::Details::rw) {
        bool bitwidth_valid = getBitWidth(argType, bitwidth);
        if (!bitwidth_valid)
          return emitError(op.getLoc(), "Unsupported argument type!");
        module_out << ",\n"
                   << "output wire[" << bitwidth - 1 << ":0]  ";
        module_out << "v_data_wr" << valueNumber;
      }
    } else {
      return emitError(op.getLoc(), "Unsupported argument type!");
    }

    if (i == args.size())
      module_out << "\n";
  }
  module_out << ");\n\n";

  for (auto arg : args)
    if (arg.getType().isa<hir::MemrefType>())
      printMemrefAddrSelector(arg);
  Value tstart = args.back();
  printTimeOffsets(tstart);
  printBody(entryBlock);
  module_out << "endmodule\n";

  // Pass module's code to out.
  std::string code = module_out.str();
  processReplacements(code);
  out << code;
  return success();
}

LogicalResult VerilogPrinter::printTimeOffsets(Value timeVar) {
  unsigned loc = replaceLocs.size();
  unsigned n_timeVar = mapValueToNum[timeVar];
  module_out << "reg t" << n_timeVar << "offset[$loc" << loc << "];\n";
  module_out << "always@(*) "
             << "t" << n_timeVar << "offset[0] <= t" << n_timeVar << ";\n";
  unsigned n_i = newValueNumber();
  module_out << "genvar i" << n_i << ";\n";
  module_out << "\ngenerate for(i" << n_i << " = 1; i" << n_i << "< $loc" << loc
             << "; i++) begin\n";
  module_out << "always@(clk) begin\n";
  module_out << "t" << n_timeVar << "offset[i" << n_i << "] <= "
             << "t" << n_timeVar << "offset[i" << n_i - 1 << "];\n";
  module_out << "end\n";
  module_out << "end\n";
  module_out << "endgenerate\n\n";

  replaceLocs.push_back(std::make_pair(loc, [=]() -> std::string {
    return std::to_string(mapTimeToMaxOffset[timeVar] + 1);
  }));
}

LogicalResult VerilogPrinter::processReplacements(std::string &code) {
  for (auto replacement : replaceLocs) {
    unsigned loc = replacement.first;
    std::function<std::string()> getReplacementString = replacement.second;
    std::string replacementStr = getReplacementString();
    std::stringstream locSStream;
    locSStream << "$loc" << loc;
    findAndReplaceAll(code, locSStream.str(), replacementStr);
  }
  return success();
}

LogicalResult VerilogPrinter::printMemrefAddrSelector(Value mem,
                                                      unsigned indentAmount) {
  unsigned n_mem = mapValueToNum[mem];
  MemrefType memrefTy = mem.getType().dyn_cast<hir::MemrefType>();
  hir::Details::PortKind port = memrefTy.getPort();
  Type elementType = memrefTy.getElementType();
  unsigned elementWidth;
  if (!getBitWidth(elementType, elementWidth))
    return failure();
  unsigned addrWidth = calcAddrWidth(memrefTy);
  unsigned loc = replaceLocs.size();
  replaceLocs.push_back(std::make_pair(loc, [=]() -> std::string {
    unsigned selWidth = mapValueToSelWidth[mem];
    if (selWidth == 0)
      return "";
    std::stringstream output;
    output << "wire[" << selWidth - 1 << ":0] v_addr" << n_mem
           << "_sel_valid;\n";
    output << "assign v_addr" << n_mem << "_valid = |v_addr" << n_mem
           << "_sel_valid;\n";
    output << "wire[" << elementWidth - 1 << ":0] v_addr" << n_mem
           << "_sel_data[" << selWidth - 1 << ":0];\n";
    output << "always@(*) begin\n";
    output << " if(v_addr" << n_mem << "_sel_valid[0] == 1'b1)\n";
    output << "   v_addr" << n_mem << " <= v_addr" << n_mem
           << "_sel_data[0];\n";
    for (int i = 1; i < selWidth; i++) {
      output << " else if(v_addr" << n_mem << "_sel_valid[" << i
             << "] == 1'b1)\n";
      output << "   v_addr" << n_mem << " <= v_addr" << n_mem << "_sel_data["
             << i << "];\n";
    }
    output << "  else\n";
    output << "   v_addr" << n_mem << " <= 0;\n";
    output << "end\n";
    return output.str();
  }));
  module_out << "$loc" << loc << "\n";
  return success();
}

LogicalResult VerilogPrinter::printBody(Block &block, unsigned indentAmount) {
  // Print the operations within the entity.
  for (auto iter = block.begin(); iter != block.end(); ++iter) {
    auto error = printOperation(&(*iter), 4);
    if (failed(error)) {
      return error;
    }
  }
}

LogicalResult VerilogPrinter::printModule(ModuleOp module) {
  WalkResult result = module.walk([this](DefOp defOp) -> WalkResult {
    if (!printDefOp(defOp).value)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  // if printing of a single operation failed, fail the whole translation.
  return failure(result.wasInterrupted());
}
} // namespace.

LogicalResult mlir::hir::printVerilog(ModuleOp module, raw_ostream &os) {
  llvm::formatted_raw_ostream out(os);
  VerilogPrinter printer(out);
  return printer.printModule(module);
}

void mlir::hir::registerHIRToVerilogTranslation() {
  TranslateFromMLIRRegistration registration(
      "hir-to-verilog", [](ModuleOp module, raw_ostream &output) {
        return printVerilog(module, output);
      });
}
