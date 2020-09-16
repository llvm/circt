//=========- HIRToVerilog.cpp - Verilog Printer ---------------------------===//
//
// This is the main HIR to Verilog Printer implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Target/HIRToVerilog/HIRToVerilog.h"
#include "circt/Dialect/HIR/HIR.h"

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

static unsigned max(unsigned x, unsigned y) { return x > y ? x : y; }
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

static unsigned getBitWidth(Type ty) {
  unsigned bitwidth = 0;
  if (auto intTy = ty.dyn_cast<IntegerType>()) {
    bitwidth = intTy.getWidth();
  } else if (auto memrefTy = ty.dyn_cast<MemrefType>()) {
    bitwidth = getBitWidth(memrefTy.getElementType());
  } else if (auto constTy = ty.dyn_cast<ConstType>()) {
    bitwidth = getBitWidth(constTy.getElementType());
  }
  return bitwidth;
}

static unsigned calcAddrWidth(hir::MemrefType memrefTy) {
  // FIXME: Currently we assume that all dims are power of two.
  auto shape = memrefTy.getShape();
  auto elementType = memrefTy.getElementType();
  auto packing = memrefTy.getPacking();
  unsigned elementWidth = getBitWidth(elementType);
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

/// Keeps track of the number of mem_reads and mem_writes on a memref.
struct MemrefParams {
  unsigned numReads = 0;
  unsigned numWrites = 0;
};

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
  LogicalResult printMemReadOp(MemReadOp op, unsigned indentAmount = 0);
  LogicalResult printAddOp(hir::AddOp op, unsigned indentAmount = 0);
  LogicalResult printMemWriteOp(MemWriteOp op, unsigned indentAmount = 0);
  LogicalResult printReturnOp(hir::ReturnOp op, unsigned indentAmount = 0);

  // Helpers.
  unsigned printDownCounter(std::stringstream &output, Value maxCount,
                            Value tstart);
  LogicalResult printBody(Block &block, unsigned indentAmount = 0);
  LogicalResult printMemrefStub(Value mem, unsigned indentAmount = 0);
  LogicalResult printType(Type type);
  LogicalResult processReplacements(std::string &code);
  LogicalResult printTimeOffsets(Value timeVar);

  std::stringstream module_out;
  llvm::formatted_raw_ostream &out;
  unsigned nextValueNum = 0;
  llvm::DenseMap<Value, unsigned> mapValueToNum;
  llvm::DenseMap<Value, int> mapConstToInt;
  llvm::DenseMap<Value, MemrefParams> mapValueToMemrefParams;
  llvm::DenseMap<Value, unsigned> mapTimeToMaxOffset;
  SmallVector<std::pair<unsigned, std::function<std::string()>>, 4> replaceLocs;
};

unsigned VerilogPrinter::printDownCounter(std::stringstream &output,
                                          Value maxCount, Value tstart) {
  unsigned id_tstart = mapValueToNum[tstart];
  unsigned id_maxCount = mapValueToNum[maxCount];
  unsigned counterWidth = getBitWidth(maxCount.getType());

  std::string t_tstart = "t" + std::to_string(id_tstart);
  unsigned id_counter = newValueNumber();
  std::string v_maxCount = "v" + std::to_string(id_maxCount);
  std::string v_counter = "v" + std::to_string(id_counter);
  output << "reg[" << counterWidth - 1 << ":0]" << v_counter << ";\n";
  output << "always@(posedge clk) begin\n";
  output << "if(" << t_tstart << ")\n";
  output << v_counter << " <= " << v_maxCount << ";\n";
  output << "else\n";
  output << v_counter << " <= (" << v_counter << ">0)?(" << v_counter
         << "-1):0;\n";
  output << "end\n";
  return id_counter;
}

LogicalResult VerilogPrinter::printMemReadOp(MemReadOp op,
                                             unsigned indentAmount) {
  OperandRange addr = op.addr();
  mlir::Value result = op.res();
  mlir::Value mem = op.mem();
  mlir::Value tstart = op.tstart();
  mlir::Value offset = op.offset();

  unsigned id_mem = mapValueToNum[mem];
  unsigned id_tstart = mapValueToNum[tstart];
  unsigned id_result = newValueNumber();

  mapValueToNum.insert(std::make_pair(result, id_result));
  int offsetValue = offset ? mapConstToInt[offset] : 0;
  unsigned width_result = getBitWidth(result.getType());

  mapTimeToMaxOffset[tstart] = max(offsetValue, mapTimeToMaxOffset[tstart]);
  MemrefParams memParams = mapValueToMemrefParams[mem];

  std::string v_addr_valid =
      "v_addr" + std::to_string(id_mem) + "_valid[" +
      std::to_string(memParams.numReads + memParams.numWrites) + "]";
  std::string v_addr_input =
      "v_addr" + std::to_string(id_mem) + "_input[" +
      std::to_string(memParams.numReads + memParams.numWrites) + "]";
  std::string t_offset = "t" + std::to_string(id_tstart) + "offset[" +
                         std::to_string(offsetValue) + "]";

  std::string v_result = "v" + std::to_string(id_result);
  std::string v_rd_en_input = "v_rd_en" + std::to_string(id_mem) + "_input[" +
                              std::to_string(memParams.numReads) + "]";
  std::string v_rd_data = "v_rd_data" + std::to_string(id_mem);

  // Address bus assignments.
  module_out << "assign " << v_addr_valid << " = " << t_offset << ";\n";
  module_out << "assign " << v_addr_input << " = {";
  int i = 0;
  for (auto addrI : addr) {
    if (i++ > 0)
      module_out << ", ";
    module_out << "v" << mapValueToNum[addrI];
  }
  module_out << "};\n";

  // Read bus assignments.
  module_out << "wire[" << (width_result - 1) << ":0]" + v_result << " = "
             << v_rd_data << ";\n\n";
  module_out << "assign " + v_rd_en_input << " = " << t_offset << ";\n\n";

  // Increment number of reads on the memref.
  memParams.numReads++;
  mapValueToMemrefParams[mem] = memParams;
  return success();
}

LogicalResult VerilogPrinter::printMemWriteOp(MemWriteOp op,
                                              unsigned indentAmount) {
  OperandRange addr = op.addr();
  mlir::Value mem = op.mem();
  mlir::Value value = op.value();
  mlir::Value tstart = op.tstart();
  mlir::Value offset = op.offset();

  unsigned id_mem = mapValueToNum[mem];
  unsigned id_tstart = mapValueToNum[tstart];
  unsigned id_value = mapValueToNum[value];

  int offsetValue = offset ? mapConstToInt[offset] : 0;

  mapTimeToMaxOffset[tstart] = max(offsetValue, mapTimeToMaxOffset[tstart]);
  MemrefParams memParams = mapValueToMemrefParams[mem];

  std::string v_addr_valid =
      "v_addr" + std::to_string(id_mem) + "_valid[" +
      std::to_string(memParams.numReads + memParams.numWrites) + "]";
  std::string v_addr_input =
      "v_addr" + std::to_string(id_mem) + "_input[" +
      std::to_string(memParams.numReads + memParams.numWrites) + "]";
  std::string t_offset = "t" + std::to_string(id_tstart) + "offset[" +
                         std::to_string(offsetValue) + "]";

  module_out << "assign " << v_addr_valid << " = " << t_offset << ";\n";
  module_out << "assign " << v_addr_input << " = {";
  int i = 0;
  for (auto addrI : addr) {
    if (i++ > 0)
      module_out << ", ";
    module_out << "v" << mapValueToNum[addrI];
  }
  module_out << "};\n";

  std::string v_wr_en_input = "v_wr_en" + std::to_string(id_mem) + "_input[" +
                              std::to_string(memParams.numWrites) + "]";
  std::string v_wr_data_input = "v_wr_data" + std::to_string(id_mem) +
                                "_input[" +
                                std::to_string(memParams.numWrites) + "]";
  std::string v_wr_data_valid = "v_wr_data" + std::to_string(id_mem) +
                                "_valid[" +
                                std::to_string(memParams.numWrites) + "]";

  std::string v_value = "v" + std::to_string(id_value);
  module_out << "assign " << v_wr_en_input << " = " << t_offset << ";\n";
  module_out << "assign " << v_wr_data_valid << " = " << t_offset << ";\n";
  module_out << "assign " << v_wr_data_input << " = " << v_value << ";\n\n";

  // Increment number of writes on the memref.
  memParams.numWrites++;
  mapValueToMemrefParams[mem] = memParams;
  return success();
}

LogicalResult VerilogPrinter::printAddOp(hir::AddOp op, unsigned indentAmount) {
  Value left = op.left();
  Value right = op.right();
  Value result = op.res();
  Type resultType = result.getType();
  unsigned id_result = newValueNumber();
  mapValueToNum.insert(std::make_pair(result, id_result));
  if (auto resultIntType = resultType.dyn_cast<IntegerType>()) {
    unsigned resultWidth = resultIntType.getWidth();
    module_out << "wire [" << resultWidth - 1 << " : 0] v" << id_result << " = "
               << "v" << mapValueToNum[left] << " + "
               << "v" << mapValueToNum[right] << ";\n";
    return success();
  } else if (auto resultConstType = resultType.dyn_cast<ConstType>()) {
    if (auto elementIntType =
            resultConstType.getElementType().dyn_cast<IntegerType>()) {
      int leftValue = mapConstToInt[left];
      int rightValue = mapConstToInt[right];
      unsigned elementWidth = elementIntType.getWidth();
      mapConstToInt.insert(std::make_pair(result, leftValue + rightValue));
      module_out << "wire [" << elementWidth - 1 << " : 0] v" << id_result
                 << " = " << leftValue + rightValue << ";\n";
    }
  }
  return emitError(op.getLoc(), "result must be either int or const<int>!");
}

LogicalResult VerilogPrinter::printReturnOp(hir::ReturnOp op,
                                            unsigned indentAmount) {
  module_out << "\n//hir.return => NOP\n"; // TODO
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
  Value lb = op.lb();
  Value ub = op.ub();
  Value step = op.step();
  Value tstart = op.tstart();
  Value tstep = op.tstep();
  Value idx = op.getInductionVar();
  Value tloop = op.getIterTimeVar();

  unsigned id_lb = mapValueToNum[op.lb()];
  unsigned id_ub = mapValueToNum[op.ub()];
  unsigned id_step = mapValueToNum[op.step()];
  unsigned id_tstep = tstep ? mapValueToNum[op.tstep()] : -1;
  unsigned id_tstart = mapValueToNum[op.tstart()];
  unsigned id_loop = newValueNumber();

  mapValueToNum.insert(std::make_pair(idx, id_loop));
  mapValueToNum.insert(std::make_pair(tloop, id_loop));

  unsigned width_lb = getBitWidth(op.lb().getType());
  unsigned width_ub = getBitWidth(op.ub().getType());
  unsigned width_step = getBitWidth(op.step().getType());
  unsigned width_tstep = op.tstep() ? getBitWidth(op.tstep().getType()) : 0;
  unsigned width_idx = getBitWidth(idx.getType());

  if (auto idx_intTy = idx.getType().dyn_cast<IntegerType>()) {
    module_out << "reg[" << idx_intTy.getWidth() << "] counter" << id_loop
               << ";\n";
  } else {
    return emitError(op.getLoc(), "for loop induction var must be an integer");
  }
  std::stringstream loopCounterStream;
  loopCounterStream << loopCounterTemplate;
  if (id_tstep >= 0) {
    unsigned id_delay_counter =
        printDownCounter(loopCounterStream, tstep, tloop);
    loopCounterStream << "assign tloop_in$n_loop = v" << id_delay_counter
                      << "==1;\n";
  }

  std::string loopCounterString = loopCounterStream.str();
  findAndReplaceAll(loopCounterString, "$n_loop", std::to_string(id_loop));
  findAndReplaceAll(loopCounterString, "$msb_idx",
                    std::to_string(width_idx - 1));
  findAndReplaceAll(loopCounterString, "$n_lb", std::to_string(id_lb));
  findAndReplaceAll(loopCounterString, "$msb_lb", std::to_string(width_lb - 1));
  findAndReplaceAll(loopCounterString, "$n_ub", std::to_string(id_ub));
  findAndReplaceAll(loopCounterString, "$msb_ub", std::to_string(width_ub - 1));
  findAndReplaceAll(loopCounterString, "$n_step", std::to_string(id_step));
  findAndReplaceAll(loopCounterString, "$msb_step",
                    std::to_string(width_step - 1));
  findAndReplaceAll(loopCounterString, "$n_tstart", std::to_string(id_tstart));
  if (id_tstep >= 0)
    findAndReplaceAll(loopCounterString, "$id_tstep", std::to_string(id_tstep));
  findAndReplaceAll(loopCounterString, "$width_tstep",
                    std::to_string(width_tstep));
  module_out << "\n//Loop" << id_loop << " begin\n";
  module_out << loopCounterString;
  module_out << "\n//Loop" << id_loop << " body\n";
  printTimeOffsets(tloop);
  printBody(op.getLoopBody().front(), indentAmount);
  module_out << "\n//Loop" << id_loop << " end\n";
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
  } else if (auto op = dyn_cast<hir::MemWriteOp>(inst)) {
    return printMemWriteOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::AddOp>(inst)) {
    return printAddOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::TerminatorOp>(inst)) {
    // Do nothing.
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
      bitwidth = getBitWidth(argType);
      module_out << "input wire[" << bitwidth - 1 << ":0]  ";
      module_out << "v" << valueNumber;
    } else if (argType.isa<MemrefType>()) {
      MemrefType memrefTy = argType.dyn_cast<MemrefType>();
      hir::Details::PortKind port = memrefTy.getPort();
      unsigned addrWidth = calcAddrWidth(memrefTy);
      module_out << "output wire[" << addrWidth - 1 << ":0] v_addr"
                 << valueNumber;
      if (port == hir::Details::r || port == hir::Details::rw) {
        bitwidth = getBitWidth(argType);
        module_out << ",\noutput wire v_rd_en" << valueNumber;
        module_out << ",\ninput wire[" << bitwidth - 1 << ":0]  "
                   << "v_rd_data" << valueNumber;
      }
      if (port == hir::Details::w || port == hir::Details::rw) {
        module_out << ",\noutput wire v_wr_en" << valueNumber;
        bitwidth = getBitWidth(argType);
        module_out << ",\n"
                   << "output wire[" << bitwidth - 1 << ":0]  ";
        module_out << "v_wr_data" << valueNumber;
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
      printMemrefStub(arg);
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
  unsigned id_timeVar = mapValueToNum[timeVar];
  module_out << "reg t" << id_timeVar << "offset[$loc" << loc << "];\n";
  module_out << "always@(*) "
             << "t" << id_timeVar << "offset[0] <= t" << id_timeVar << ";\n";
  unsigned id_i = newValueNumber();
  module_out << "genvar i" << id_i << ";\n";
  module_out << "\ngenerate for(i" << id_i << " = 1; i" << id_i << "< $loc"
             << loc << "; i++) begin\n";
  module_out << "always@(clk) begin\n";
  module_out << "t" << id_timeVar << "offset[i" << id_i << "] <= "
             << "t" << id_timeVar << "offset[i" << id_i << "-1];\n";
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

std::string buildEnableSelectorStr(std::string &v_en, unsigned numInputs) {
  std::stringstream output;
  std::string v_en_input = v_en + "_input";
  output << "wire[" << numInputs - 1 << ":0]" << v_en_input << ";\n";
  output << "assign " << v_en << " = |" << v_en_input << ";\n";
  return output.str();
}

std::string buildDataSelectorStr(std::string &v, unsigned numInputs,
                                 unsigned dataWidth) {
  std::stringstream output;
  std::string v_valid = v + "_valid";
  std::string v_input = v + "_input";
  output << "wire " << v_valid << "[" << numInputs << "];\n";
  output << "wire[" << dataWidth - 1 << ":0] " << v_input << "[" << numInputs
         << "];\n";
  output << "always@(*) begin\n";
  output << "if(" << v_valid << "[0] == 1'b1)\n";
  output << v << " <= " << v_input << "[0];\n";
  for (int i = 1; i < numInputs; i++) {
    output << "else if(" << v_valid << "[" << i << "] == 1'b1)\n";
    output << v << " <= " << v_input << "[" << i << "];\n";
  }
  output << "  else\n";
  output << v << " <= " << dataWidth << "'d0;\n";
  output << "end\n";
  return output.str();
}

LogicalResult VerilogPrinter::printMemrefStub(Value mem,
                                              unsigned indentAmount) {
  /// Prints a placeholder $locN which is later replaced by selectors for
  /// addr,
  // rd_data, wr_data and rd_en, wr_en.
  unsigned id_mem = mapValueToNum[mem];
  MemrefType memrefTy = mem.getType().dyn_cast<hir::MemrefType>();
  hir::Details::PortKind port = memrefTy.getPort();
  Type elementType = memrefTy.getElementType();
  unsigned dataWidth;
  dataWidth = getBitWidth(elementType);
  unsigned addrWidth = calcAddrWidth(memrefTy);
  unsigned loc = replaceLocs.size();
  replaceLocs.push_back(std::make_pair(loc, [=]() -> std::string {
    MemrefParams memParams = mapValueToMemrefParams[mem];
    unsigned numAccess = memParams.numReads + memParams.numWrites;
    if (numAccess == 0)
      return "";
    std::stringstream output;
    std::string v_addr = "v_addr" + std::to_string(id_mem);
    // print addr bus selector.
    output << buildDataSelectorStr(v_addr, numAccess, addrWidth);
    output << "\n";
    // print read bus selector.
    if (memParams.numReads > 0) {
      std::string v_rd_en = "v_rd_en" + std::to_string(id_mem);
      output << buildEnableSelectorStr(v_rd_en, memParams.numReads);
      std::string v_rd_data = "v_rd_data" + std::to_string(id_mem);
      output << buildDataSelectorStr(v_rd_data, memParams.numReads, dataWidth);
      output << "\n";
    }
    // print write bus selector.
    if (memParams.numWrites > 0) {
      std::string v_wr_en = "v_wr_en" + std::to_string(id_mem);
      output << buildEnableSelectorStr(v_wr_en, memParams.numWrites);
      std::string v_wr_data = "v_wr_data" + std::to_string(id_mem);
      output << buildDataSelectorStr(v_wr_data, memParams.numWrites, dataWidth);
      output << "\n";
    }
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
