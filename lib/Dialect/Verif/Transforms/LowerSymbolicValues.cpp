//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"

using namespace circt;
using namespace mlir;
using namespace verif;
using namespace hw;

namespace circt {
namespace verif {
#define GEN_PASS_DEF_LOWERSYMBOLICVALUESPASS
#include "circt/Dialect/Verif/Passes.h.inc"
} // namespace verif
} // namespace circt

namespace {
struct LowerSymbolicValuesPass
    : verif::impl::LowerSymbolicValuesPassBase<LowerSymbolicValuesPass> {
  using LowerSymbolicValuesPassBase::LowerSymbolicValuesPassBase;
  void runOnOperation() override;
  LogicalResult lowerToExtModule();
  void lowerToAnySeqWire();
};
} // namespace

void LowerSymbolicValuesPass::runOnOperation() {
  switch (mode) {
  case SymbolicValueLowering::ExtModule:
    if (failed(lowerToExtModule()))
      signalPassFailure();
    break;
  case SymbolicValueLowering::Yosys:
    lowerToAnySeqWire();
    break;
  }
}

/// Replace all `SymbolicValueOp`s with instances of corresponding extmodules.
/// This allows tools to treat the modules as blackboxes, or definitions of the
/// modules may be provided later by the user.
LogicalResult LowerSymbolicValuesPass::lowerToExtModule() {
  auto &symbolTable = getAnalysis<SymbolTable>();
  DenseMap<Type, HWModuleExternOp> extmoduleOps;
  auto result = getOperation().walk([&](SymbolicValueOp op) -> WalkResult {
    // Determine the number of bits needed for the symbolic value.
    auto numBits = hw::getBitWidth(op.getType());
    if (!numBits)
      return op.emitError() << "symbolic value bit width unknown";

    // If we don't already have an extmodule for this number of bits, create
    // one.
    auto builder = OpBuilder::atBlockEnd(getOperation().getBody());
    auto flatType = builder.getIntegerType(*numBits);
    auto &extmoduleOp = extmoduleOps[flatType];
    if (!extmoduleOp) {
      extmoduleOp = HWModuleExternOp::create(
          builder, op.getLoc(),
          builder.getStringAttr(Twine("circt.symbolic_value.") +
                                Twine(numBits)),
          PortInfo{{builder.getStringAttr("z"), flatType, ModulePort::Output}},
          "circt_symbolic_value",
          builder.getArrayAttr(ParamDeclAttr::get(
              builder.getContext(), builder.getStringAttr("WIDTH"),
              builder.getI32Type(), Attribute())));
      symbolTable.insert(extmoduleOp);
    }

    // Instantiate the extmodule as a means of generating a symbolic value with
    // the correct number of bits.
    builder.setInsertionPoint(op);
    auto instOp = InstanceOp::create(
        builder, op.getLoc(), extmoduleOp,
        builder.getStringAttr("symbolic_value"), ArrayRef<Value>{},
        builder.getArrayAttr(ParamDeclAttr::get(
            builder.getContext(), builder.getStringAttr("WIDTH"),
            builder.getI32Type(), builder.getI32IntegerAttr(numBits))));
    Value value = instOp.getResult(0);

    // Insert a bit cast if needed to obtain the original symbolic value's type.
    if (op.getType() != value.getType())
      value = BitcastOp::create(builder, op.getLoc(), op.getType(), value);

    // Replace the `verif.symbolic_value` op.
    op.replaceAllUsesWith(value);
    op.erase();
    return success();
  });
  return failure(result.wasInterrupted());
}

/// Replace `SymbolicValueOp`s with an `(* anyseq *)` wire declaration.
void LowerSymbolicValuesPass::lowerToAnySeqWire() {
  getOperation().walk([&](SymbolicValueOp op) {
    // Create a replacement wire declaration with a `(* anyseq *)` Verilog
    // attribute.
    OpBuilder builder(op);
    auto wireOp = sv::WireOp::create(builder, op.getLoc(), op.getType());
    sv::addSVAttributes(wireOp,
                        sv::SVAttributeAttr::get(&getContext(), "anyseq"));

    // Create a read from the wire and replace the `verif.symbolic_value` op.
    Value value = sv::ReadInOutOp::create(builder, op.getLoc(), wireOp);
    op.replaceAllUsesWith(value);
    op.erase();
  });
}
