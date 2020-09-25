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
    "\nreg[$msb_idx:0] prev_idx$n_loop;"
    "\nreg[$msb_ub:0] saved_ub$n_loop;"
    "\nreg[$msb_step:0] saved_step$n_loop;"
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

static unsigned max(unsigned x, unsigned y) { return x > y ? x : y; }
static void findAndReplaceAll(string &data, string toSearch,
                              string replaceStr) {
  // Get the first occurrence.
  size_t pos = data.find(toSearch);
  // Repeat till end is reached.
  while (pos != string::npos) {
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

/// This class is the verilog output printer for the HIR dialect. It walks
/// throught the module and prints the verilog in the 'out' variable.
class VerilogPrinter {
public:
  VerilogPrinter(llvm::formatted_raw_ostream &output) : out(output) {}

  LogicalResult printModule(ModuleOp op);

private:
  enum VerilogValueTypes { vUnknown, vWire, vMemref };

  class VerilogValue {
  public:
    VerilogValue() : type(vUnknown), name("unknown") {}
    VerilogValue(VerilogValueTypes type, string name)
        : type(type), name(name) {}

    string strWire() { return name; }
    string strWireValid() { return name + "_valid"; }
    string strWireInput() { return name + "_input"; }
    string strWireInput(unsigned idx) {
      return strWireInput() + "{" + to_string(idx) + "}";
    }
    string strDelayedWire() { return name + "delay"; }
    string strDelayedWire(unsigned delay) {
      return strDelayedWire() + "[" + to_string(delay) + "]";
    }

    string strMemrefAddr() { return name + "_addr"; }
    string strMemrefAddrValid() { return name + "_addr_valid"; }
    string strMemrefAddrValid(unsigned idx) {
      return strMemrefAddrValid() + "[" + to_string(idx) + "]";
    }
    string strMemrefAddrInput() { return name + "_addr_input"; }
    string strMemrefAddrInput(unsigned idx) {
      return strMemrefAddrInput() + "[" + to_string(idx) + "]";
    }

    string strMemrefRdData() { return name + "_rd_data"; }

    string strMemrefWrData() { return name + "_wr_data"; }
    string strMemrefWrDataValid() { return name + "_wr_data_valid"; }
    string strMemrefWrDataValid(unsigned idx) {
      return strMemrefWrDataValid() + "[" + to_string(idx) + "]";
    }
    string strMemrefWrDataInput() { return name + "_wr_data_input"; }
    string strMemrefWrDataInput(unsigned idx) {
      return strMemrefWrDataInput() + "[" + to_string(idx) + "]";
    }

    string strMemrefRdEn() { return name + "_rd_en"; }
    string strMemrefRdEnInput() { return name + "_rd_en_input"; }
    string strMemrefRdEnInput(unsigned idx) {
      return strMemrefRdEnInput() + "[" + to_string(idx) + "]";
    }

    string strMemrefWrEn() { return name + "_wr_en"; }
    string strMemrefWrEnInput() { return name + "_wr_en_input"; }
    string strMemrefWrEnInput(unsigned idx) {
      return strMemrefWrEnInput() + "[" + to_string(idx) + "]";
    }
    int numReads() { return info.numReads; }
    void incNumReads() { info.numReads++; }
    int numWrites() { return info.numWrites; }
    void incNumWrites() { info.numWrites++; }
    int numAccess() { return info.numReads + info.numWrites; }

  private:
    struct {
      unsigned numReads = 0;
      unsigned numWrites = 0;
    } info;
    VerilogValueTypes type;
    string name;
  };

  unsigned newValueNumber() { return nextValueNum++; }

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

  // Helpers.
  string printDownCounter(stringstream &output, Value maxCount, Value tstart);
  LogicalResult printBody(Block &block, unsigned indentAmount = 0);
  LogicalResult printMemrefStub(Value mem, unsigned indentAmount = 0);
  LogicalResult printType(Type type);
  LogicalResult processReplacements(string &code);
  LogicalResult printTimeOffsets(Value timeVar);

  stringstream module_out;
  llvm::formatted_raw_ostream &out;
  unsigned nextValueNum = 0;
  // llvm::DenseMap<Value, unsigned> mapValueToNum;
  llvm::DenseMap<Value, VerilogValue> mapValueToVerilog;
  llvm::DenseMap<Value, int> mapConstToInt;
  llvm::DenseMap<Value, unsigned> mapTimeToMaxOffset;
  SmallVector<pair<unsigned, function<string()>>, 4> replaceLocs;
  std::stack<string> yieldTarget;
  std::stack<hir::UnrollForOp> unrollForStack;
}; // namespace

string VerilogPrinter::printDownCounter(stringstream &output, Value maxCount,
                                        Value tstart) {
  unsigned counterWidth = getBitWidth(maxCount.getType());

  string v_tstart = mapValueToVerilog[tstart].strWire();
  unsigned id_counter = newValueNumber();
  string v_maxCount = mapValueToVerilog[maxCount].strWire();
  string v_counter = "downCount" + to_string(id_counter);
  output << "reg[" << counterWidth - 1 << ":0]" << v_counter << ";\n";
  output << "always@(posedge clk) begin\n";
  output << "if(" << v_tstart << ")\n";
  output << v_counter << " <= " << v_maxCount << ";\n";
  output << "else\n";
  output << v_counter << " <= (" << v_counter << ">0)?(" << v_counter
         << "-1):0;\n";
  output << "end\n";
  return v_counter;
}

LogicalResult VerilogPrinter::printMemReadOp(MemReadOp op,
                                             unsigned indentAmount) {
  mlir::Value result = op.res();
  unsigned id_result = newValueNumber();
  VerilogValue v_result(vWire, "v" + to_string(id_result));
  mapValueToVerilog.insert(make_pair(result, v_result));

  OperandRange addr = op.addr();
  mlir::Value mem = op.mem();
  mlir::Value tstart = op.tstart();
  mlir::Value offset = op.offset();

  auto v_mem = mapValueToVerilog[mem];

  int delayValue = offset ? mapConstToInt[offset] : 0;
  mapTimeToMaxOffset[tstart] = max(delayValue, mapTimeToMaxOffset[tstart]);
  unsigned width_result = getBitWidth(result.getType());

  auto v_tstart = mapValueToVerilog[tstart];

  // Address bus assignments.
  module_out << "assign " << v_mem.strMemrefAddrValid(v_mem.numAccess())
             << " = " << v_tstart.strDelayedWire(delayValue) << ";\n";
  module_out << "assign " << v_mem.strMemrefAddrInput(v_mem.numAccess())
             << " = {";
  int i = 0;
  for (auto addrI : addr) {
    if (i++ > 0)
      module_out << ", ";
    module_out << mapValueToVerilog[addrI].strWire();
  }
  module_out << "};\n";

  // Read bus assignments.
  module_out << "wire[" << (width_result - 1) << ":0]" + v_result.strWire()
             << " = " << v_mem.strMemrefRdData() << ";\n\n";
  module_out << "assign " + v_mem.strMemrefRdEnInput(v_mem.numReads()) << " = "
             << v_tstart.strDelayedWire(delayValue) << ";\n\n";

  mapValueToVerilog[mem].incNumReads();
  return success();
}

LogicalResult VerilogPrinter::printMemWriteOp(MemWriteOp op,
                                              unsigned indentAmount) {
  OperandRange addr = op.addr();
  mlir::Value mem = op.mem();
  mlir::Value value = op.value();
  mlir::Value tstart = op.tstart();
  mlir::Value offset = op.offset();

  auto v_value = mapValueToVerilog[value];
  auto v_mem = mapValueToVerilog[mem];

  int delayValue = offset ? mapConstToInt[offset] : 0;
  mapTimeToMaxOffset[tstart] = max(delayValue, mapTimeToMaxOffset[tstart]);

  auto v_tstart = mapValueToVerilog[tstart];

  module_out << "assign " << v_mem.strMemrefAddrValid(v_mem.numAccess())
             << " = " << v_tstart.strDelayedWire(delayValue) << ";\n";
  module_out << "assign " << v_mem.strMemrefAddrInput(v_mem.numAccess())
             << " = {";
  int i = 0;
  for (auto addrI : addr) {
    if (i++ > 0)
      module_out << ", ";
    module_out << mapValueToVerilog[addrI].strWire();
  }
  module_out << "};\n";

  module_out << "assign " << v_mem.strMemrefWrEnInput(v_mem.numWrites())
             << " = " << v_tstart.strDelayedWire(delayValue) << ";\n";
  module_out << "assign " << v_mem.strMemrefWrDataValid(v_mem.numWrites())
             << " = " << v_tstart.strDelayedWire(delayValue) << ";\n";
  module_out << "assign " << v_mem.strMemrefWrDataInput(v_mem.numWrites())
             << " = " << v_value.strWire() << ";\n\n";

  mapValueToVerilog[mem].incNumWrites();
  return success();
}

LogicalResult VerilogPrinter::printAddOp(hir::AddOp op, unsigned indentAmount) {
  // TODO: Handle signed and unsigned integers of unequal width using sign
  // extension.
  Value result = op.res();
  unsigned id_result = newValueNumber();
  VerilogValue v_result(vWire, "v" + to_string(id_result));
  mapValueToVerilog.insert(make_pair(result, v_result));

  Value left = op.left();
  Value right = op.right();
  Type resultType = result.getType();
  if (auto resultIntType = resultType.dyn_cast<IntegerType>()) {
    unsigned resultWidth = resultIntType.getWidth();
    module_out << "wire [" << resultWidth - 1 << " : 0] v" << id_result << " = "
               << mapValueToVerilog[left].strWire() << " + "
               << mapValueToVerilog[right].strWire() << ";\n";
    return success();
  } else if (auto resultConstType = resultType.dyn_cast<ConstType>()) {
    if (auto elementIntType =
            resultConstType.getElementType().dyn_cast<IntegerType>()) {
      int leftValue = mapConstToInt[left];
      int rightValue = mapConstToInt[right];
      unsigned elementWidth = elementIntType.getWidth();
      mapConstToInt.insert(make_pair(result, leftValue + rightValue));
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
  unsigned id_result = newValueNumber();
  VerilogValue v_result(vWire, "v" + to_string(id_result));
  mapValueToVerilog.insert(make_pair(result, v_result));

  int value = op.value().getLimitedValue();
  mapConstToInt.insert(make_pair(result, value));

  if (auto intTy = result.getType().dyn_cast<IntegerType>()) {
    module_out << "wire [" << intTy.getWidth() - 1 << " : 0]"
               << v_result.strWire();
    module_out << " = " << intTy.getWidth() << "'d" << value << ";\n";
    return success();
  } else if (auto constTy = result.getType().dyn_cast<ConstType>()) {
    if (auto intTy = constTy.getElementType().dyn_cast<IntegerType>()) {
      module_out << "wire [" << intTy.getWidth() - 1 << " : 0]"
                 << v_result.strWire();
      module_out << " = " << intTy.getWidth() << "'d" << value << ";\n";
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

  auto id_loop = newValueNumber();
  yieldTarget.push("tloop_in" + to_string(id_loop));

  auto v_lb = mapValueToVerilog[op.lb()];
  auto v_ub = mapValueToVerilog[op.ub()];
  auto v_step = mapValueToVerilog[op.step()];
  auto v_tstart = mapValueToVerilog[op.tstart()];
  auto v_idx = VerilogValue(vWire, "idx" + to_string(id_loop));
  auto v_tloop = VerilogValue(vWire, "tloop" + to_string(id_loop));
  mapValueToVerilog.insert(make_pair(idx, v_idx));
  mapValueToVerilog.insert(make_pair(tloop, v_tloop));

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
  findAndReplaceAll(loopCounterString, "$v_lb", v_lb.strWire());
  findAndReplaceAll(loopCounterString, "$msb_lb", to_string(width_lb - 1));
  findAndReplaceAll(loopCounterString, "$v_ub", v_ub.strWire());
  findAndReplaceAll(loopCounterString, "$msb_ub", to_string(width_ub - 1));
  findAndReplaceAll(loopCounterString, "$v_step", v_step.strWire());
  findAndReplaceAll(loopCounterString, "$msb_step", to_string(width_step - 1));
  findAndReplaceAll(loopCounterString, "$v_tstart", v_tstart.strWire());
  findAndReplaceAll(loopCounterString, "$width_tstep", to_string(width_tstep));
  findAndReplaceAll(loopCounterString, "$v_tloop", v_tloop.strWire());
  module_out << "\n//{ Loop" << id_loop << "\n";
  module_out << loopCounterString;
  module_out << "\n//Loop" << id_loop << " body\n";
  printTimeOffsets(tloop);
  printBody(op.getLoopBody().front(), indentAmount);
  module_out << "\n//} Loop" << id_loop << "\n";
  yieldTarget.pop();
  return success();
}

LogicalResult VerilogPrinter::printUnrollForOp(UnrollForOp op,
                                               unsigned indentAmount) {
  int lb = op.lb().getLimitedValue();
  int ub = op.ub().getLimitedValue();
  int step = op.step().getLimitedValue();
  int tstep = op.tstep().getValueOr(APInt()).getLimitedValue();
  Value tstart = op.tstart();
  auto v_tstart = mapValueToVerilog[tstart];
  Value idx = op.getInductionVar();
  Value tloop = op.getIterTimeVar();

  auto id_loop = newValueNumber();

  auto v_i = VerilogValue(vWire, "i" + to_string(id_loop));
  auto v_tloop = VerilogValue(vWire, "tloop" + to_string(id_loop));
  mapValueToVerilog.insert(make_pair(idx, v_i));
  mapValueToVerilog.insert(make_pair(tloop, v_tloop));
  yieldTarget.push("tloop_in" + to_string(id_loop) + "[" + v_i.strWire() +
                   "+1]");

  module_out << "\n//{ Loop" << id_loop << "\n";
  module_out << "genvar " << v_i.strWire() << ";\n";
  module_out << "wire[" + to_string(ub - 1) + " : " + to_string(lb) +
                    "] tloop_in"
             << to_string(id_loop) << ";\n";
  module_out << "assign tloop_in" << to_string(id_loop)
             << "[0] = " << v_tstart.strWire() << ";\n";
  if (unrollForStack.empty())
    module_out << "generate"; // this is the first generate
  unrollForStack.push(op);
  module_out << " for(" << v_i.strWire() << " = " << lb << ";" << v_i.strWire()
             << " < " << ub << "; " << v_i.strWire() << " = " << v_i.strWire()
             << " + " << step << ") begin\n";
  module_out << "wire " << v_tloop.strWire() << " = tloop_in"
             << to_string(id_loop) << "[" << v_i.strWire() << "];\n";

  printTimeOffsets(tloop);
  printBody(op.getLoopBody().front(), indentAmount);

  module_out << "end\n";
  unrollForStack.pop();
  if (unrollForStack.empty())
    module_out << "endgenerate";

  yieldTarget.pop();
  return success();
}

LogicalResult VerilogPrinter::printOperation(Operation *inst,
                                             unsigned indentAmount) {
  if (auto op = dyn_cast<hir::ConstantOp>(inst)) {
    module_out << "\n//ConstantOp\n";
    return printConstantOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::ForOp>(inst)) {
    module_out << "\n//ForOp\n";
    return printForOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::UnrollForOp>(inst)) {
    module_out << "\n//UnrollForOp\n";
    return printUnrollForOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::ReturnOp>(inst)) {
    module_out << "\n//ReturnOp\n";
    return printReturnOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::MemReadOp>(inst)) {
    module_out << "\n//MemReadOp\n";
    return printMemReadOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::MemWriteOp>(inst)) {
    module_out << "\n//MemWriteOp\n";
    return printMemWriteOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::AddOp>(inst)) {
    module_out << "\n//AddOp\n";
    return printAddOp(op, indentAmount);
  } else if (auto op = dyn_cast<hir::TerminatorOp>(inst)) {
    module_out << "\n//TerminatorOp\n";
    // Do nothing.
  } else if (auto op = dyn_cast<hir::YieldOp>(inst)) {
    module_out << "\n//YieldOp\n";
    return printYieldOp(op, indentAmount);
  } else {
    return emitError(inst->getLoc(), "Unsupported Operation for codegen!");
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
      auto tstart = VerilogValue(vWire, "tstart");
      mapValueToVerilog.insert(make_pair(arg, tstart));

      module_out << "//TimeType.\n";
      module_out << "input wire " << tstart.strWire();
    } else if (argType.isa<IntegerType>()) {
      unsigned id_int_arg = newValueNumber();
      auto v_int_arg = VerilogValue(vWire, "v" + to_string(id_int_arg));
      mapValueToVerilog.insert(make_pair(arg, v_int_arg));

      module_out << "//IntegerType.\n";
      unsigned bitwidth = getBitWidth(argType);
      module_out << "input wire[" << bitwidth - 1 << ":0]  "
                 << v_int_arg.strWire();
    } else if (argType.isa<MemrefType>()) {
      unsigned id_memref_arg = newValueNumber();
      auto v_memref_arg = VerilogValue(vMemref, "v" + to_string(id_memref_arg));
      mapValueToVerilog.insert(make_pair(arg, v_memref_arg));

      MemrefType memrefTy = argType.dyn_cast<MemrefType>();
      hir::Details::PortKind port = memrefTy.getPort();
      module_out << "//MemrefType : port = "
                 << ((port == hir::Details::r)
                         ? "r"
                         : (port == hir::Details::w) ? "w" : "rw")
                 << ".\n";
      unsigned addrWidth = calcAddrWidth(memrefTy);
      module_out << "output reg[" << addrWidth - 1 << ":0] "
                 << v_memref_arg.strMemrefAddr();
      if (port == hir::Details::r || port == hir::Details::rw) {
        unsigned bitwidth = getBitWidth(argType);
        module_out << ",\noutput wire " << v_memref_arg.strMemrefRdEn();
        module_out << ",\ninput wire[" << bitwidth - 1 << ":0]  "
                   << v_memref_arg.strMemrefRdData();
      }
      if (port == hir::Details::w || port == hir::Details::rw) {
        module_out << ",\noutput wire " << v_memref_arg.strMemrefWrEn();
        unsigned bitwidth = getBitWidth(argType);
        module_out << ",\n"
                   << "output reg[" << bitwidth - 1 << ":0] "
                   << v_memref_arg.strMemrefWrData();
      }
    } else {
      return emitError(op.getLoc(), "Unsupported argument type!");
    }

    if (i == args.size())
      module_out << "\n";
  }
  module_out << ",\n//Clock.\ninput wire clk\n);\n\n";

  for (auto arg : args)
    if (arg.getType().isa<hir::MemrefType>())
      printMemrefStub(arg);
  Value tstart = args.back();
  printTimeOffsets(tstart);
  printBody(entryBlock);
  module_out << "endmodule\n";

  // Pass module's code to out.
  string code = module_out.str();
  processReplacements(code);
  out << code;
  return success();
}

LogicalResult VerilogPrinter::printYieldOp(hir::YieldOp op,
                                           unsigned indentAmount) {
  auto operands = op.operands();
  Value tstart = op.tstart();
  Value offset = op.offset();
  int delayValue = mapConstToInt[offset];
  mapTimeToMaxOffset[tstart] = max(delayValue, mapTimeToMaxOffset[tstart]);
  auto v_tstart = mapValueToVerilog[tstart];
  if (!operands.empty())
    emitError(op.getLoc(), "YieldOp codegen does not support arguments yet!");
  string tloop_in = yieldTarget.top();
  module_out << "assign " << tloop_in << " = "
             << v_tstart.strDelayedWire(delayValue) << ";\n";
  return success();
}

LogicalResult VerilogPrinter::printTimeOffsets(Value timeVar) {
  module_out << "//printTimeOffset\n";
  unsigned loc = replaceLocs.size();
  auto v_timeVar = mapValueToVerilog[timeVar];
  module_out << "reg " << v_timeVar.strDelayedWire() << "[$loc" << loc
             << ":0];\n";
  module_out << "always@(*) " << v_timeVar.strDelayedWire(0) << " = "
             << v_timeVar.strWire() << ";\n";
  string str_i = "i" + to_string(newValueNumber());
  if (unrollForStack.empty())
    module_out << "generate\n";
  module_out << "genvar " << str_i << ";\n";
  module_out << "\nfor(" << str_i << " = 1; " << str_i << "<= $loc" << loc
             << "; " << str_i << "= " << str_i << " + 1) begin\n";
  module_out << "always@(posedge clk) begin\n";
  module_out << v_timeVar.strDelayedWire() << "[" << str_i
             << "] <= " << v_timeVar.strDelayedWire() << "[" << str_i
             << "-1];\n";
  module_out << "end\n";
  module_out << "end\n";
  if (unrollForStack.empty())
    module_out << "endgenerate\n\n";

  replaceLocs.push_back(make_pair(
      loc, [=]() -> string { return to_string(mapTimeToMaxOffset[timeVar]); }));
}

LogicalResult VerilogPrinter::processReplacements(string &code) {
  for (auto replacement : replaceLocs) {
    unsigned loc = replacement.first;
    function<string()> getReplacementString = replacement.second;
    string replacementStr = getReplacementString();
    stringstream locSStream;
    locSStream << "$loc" << loc;
    findAndReplaceAll(code, locSStream.str(), replacementStr);
  }
  return success();
}

string buildEnableSelectorStr(string &v_en, unsigned numInputs) {
  stringstream output;
  string v_en_input = v_en + "_input";
  output << "wire[" << numInputs - 1 << ":0]" << v_en_input << ";\n";
  output << "assign " << v_en << " = |" << v_en_input << ";\n";
  return output.str();
}

string buildDataSelectorStr(string &v, unsigned numInputs, unsigned dataWidth) {
  stringstream output;
  string v_valid = v + "_valid";
  string v_input = v + "_input";
  output << "wire " << v_valid << "[" << numInputs - 1 << ":0];\n";
  output << "wire[" << dataWidth - 1 << ":0] " << v_input << "["
         << numInputs - 1 << ":0];\n";
  output << "always@(*) begin\n";
  output << "if(" << v_valid << "[0] == 1'b1)\n";
  output << v << " = " << v_input << "[0];\n";
  for (int i = 1; i < numInputs; i++) {
    output << "else if(" << v_valid << "[" << i << "] == 1'b1)\n";
    output << v << " = " << v_input << "[" << i << "];\n";
  }
  output << "  else\n";
  output << v << " = " << dataWidth << "'d0;\n";
  output << "end\n";
  return output.str();
}

LogicalResult VerilogPrinter::printMemrefStub(Value mem,
                                              unsigned indentAmount) {
  /// Prints a placeholder $locN which is later replaced by selectors for
  /// addr,
  // rd_data, wr_data and rd_en, wr_en.
  MemrefType memrefTy = mem.getType().dyn_cast<hir::MemrefType>();
  hir::Details::PortKind port = memrefTy.getPort();
  Type elementType = memrefTy.getElementType();
  unsigned dataWidth;
  dataWidth = getBitWidth(elementType);
  unsigned addrWidth = calcAddrWidth(memrefTy);
  unsigned loc = replaceLocs.size();
  replaceLocs.push_back(make_pair(loc, [=]() -> string {
    VerilogValue v_mem = mapValueToVerilog[mem];
    if (v_mem.numAccess() == 0)
      return "//Unused memref " + v_mem.strWire();
    stringstream output;
    auto v_addr = v_mem.strMemrefAddr();
    // print addr bus selector.
    output << buildDataSelectorStr(v_addr, v_mem.numAccess(), addrWidth);
    output << "\n";
    // print rd_en selector.
    if (v_mem.numReads() > 0) {
      string v_rd_en = v_mem.strMemrefRdEn();
      output << buildEnableSelectorStr(v_rd_en, v_mem.numReads());
      output << "\n";
    }

    // print write bus selector.
    if (v_mem.numWrites() > 0) {
      string str_wr_en = v_mem.strMemrefWrEn();
      output << buildEnableSelectorStr(str_wr_en, v_mem.numWrites());
      string str_wr_data = v_mem.strMemrefWrData();
      output << buildDataSelectorStr(str_wr_data, v_mem.numWrites(), dataWidth);
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
