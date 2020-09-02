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

using namespace mlir;
using namespace hir;

namespace {

class VerilogPrinter {
public:
  VerilogPrinter(llvm::formatted_raw_ostream &output) : out(output) {}

  LogicalResult printModule(ModuleOp op);

private:
  unsigned newValueNumber() { return nextValueNum++; }
  LogicalResult printDefOp(DefOp op, unsigned indentAmount = 0);
  LogicalResult printConstantOp(hir::ConstantOp op, unsigned indentAmount = 0);
  LogicalResult printForOp(ForOp op, unsigned indentAmount = 0);
  LogicalResult printAddOp(AddOp op, unsigned indentAmount = 0);
  LogicalResult printMemReadOp(MemReadOp op, unsigned indentAmount = 0);
  LogicalResult printMemWriteOp(MemWriteOp op, unsigned indentAmount = 0);
  LogicalResult printType(Type type);

  llvm::formatted_raw_ostream &out;
  unsigned nextValueNum = 0;
  DenseMap<Value, unsigned> mapValueToNum;
};

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

LogicalResult VerilogPrinter::printDefOp(DefOp op, unsigned indentAmount) {
  // An EntityOp always has a single block
  Block &entryBlock = op.getBody().front();
  auto args = entryBlock.getArguments();
  // Print the module signature
  FunctionType funcType = op.getType();
  out << "module " << op.getName() << "(";
  for (int i = 0; i < args.size() - 1; i++) {
    unsigned bitwidth = 0;
    Value arg = args[i];
    Type argType = arg.getType();
    unsigned valueNumber = newValueNumber();
    mapValueToNum.insert(std::make_pair(arg, valueNumber));

    // Print verilog
    if(i==0)
      out <<"\n";

    if (i > 0)
      out << ",\n";

    if (argType.isa<IntegerType>()) {
      getBitWidth(argType, bitwidth);
      out << "input wire[" << bitwidth - 1 << ":0]  ";
      out << "v" << valueNumber;
    } else if (argType.isa<MemrefType>()) {
      MemrefType memrefTy = argType.dyn_cast<MemrefType>();
      ArrayRef<unsigned> shape = memrefTy.getShape();
      Type elementType = memrefTy.getElementType();
      ArrayRef<unsigned> packing = memrefTy.getPacking();
      hir::Details::PortKind port = memrefTy.getPort();
      unsigned addrWidth = 0;
      // FIXME: Currently we assume that all dims are power of two.
      for (auto dim : shape) {
        addrWidth += ceil(log2(dim));
      }
      out << "output wire[" << addrWidth - 1 << ":0] v_addr" << valueNumber;
      if (port == hir::Details::r || port == hir::Details::rw) {
        bool bitwidth_valid = getBitWidth(argType, bitwidth);
        if (!bitwidth_valid)
          return emitError(arg.getLoc(), "Unsupported argument type!");
        out << ",\n"
            << "input wire[" << bitwidth - 1 << ":0]  ";
        out << "v_data_rd" << valueNumber;
      }
      if (port == hir::Details::w || port == hir::Details::rw) {
        bool bitwidth_valid = getBitWidth(argType, bitwidth);
        if (!bitwidth_valid)
          return emitError(arg.getLoc(), "Unsupported argument type!");
        out << ",\n"
            << "output wire[" << bitwidth - 1 << ":0]  ";
        out << "v_data_wr" << valueNumber;
      }
    } else
      return emitError(arg.getLoc(), "Unsupported argument type!");
    if(i==args.size())
      out <<"\n";
  }
  out << ")";

  out << ";\n";
  out << "endmodule\n";
  return success();
}

LogicalResult VerilogPrinter::printModule(ModuleOp module) {
  WalkResult result = module.walk([this](DefOp defOp) -> WalkResult {
    if (!printDefOp(defOp).value)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  // if printing of a single operation failed, fail the whole translation
  return failure(result.wasInterrupted());
}
} // namespace

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
