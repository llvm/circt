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
#include "llvm/ADT/DenseSet.h"

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
  LogicalResult lowerToHWInputs();
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
  case SymbolicValueLowering::HWInput:
    if (failed(lowerToHWInputs()))
      signalPassFailure();
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
    if (numBits < 0)
      return op.emitError() << "symbolic value bit width unknown";

    // If we don't already have an extmodule for this number of bits, create
    // one.
    auto builder = OpBuilder::atBlockEnd(getOperation().getBody());
    auto flatType = builder.getIntegerType(numBits);
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

/// Replace `SymbolicValueOp`s in `hw.module` bodies with input ports.
LogicalResult LowerSymbolicValuesPass::lowerToHWInputs() {
  auto module = getOperation();
  DenseSet<StringAttr> referencedModules;
  module.walk([&](InstanceOp op) {
    referencedModules.insert(op.getReferencedModuleNameAttr());
  });

  auto result = module.walk([&](SymbolicValueOp op) -> WalkResult {
    if (!op->getParentOfType<HWModuleOp>())
      return op.emitError()
             << "cannot lower symbolic value to hw.module input outside of an "
                "hw.module";
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  for (auto hwModule : module.getOps<HWModuleOp>()) {
    SmallVector<SymbolicValueOp> symbolicValues;
    hwModule.walk([&](SymbolicValueOp op) { symbolicValues.push_back(op); });
    if (symbolicValues.empty())
      continue;

    if (referencedModules.contains(hwModule.getModuleNameAttr())) {
      hwModule.emitError()
          << "cannot lower symbolic values in instantiated module '"
          << hwModule.getModuleName()
          << "' to HW inputs; run the 'hw-input' lowering strategy after "
             "flattening modules";
      return failure();
    }

    for (auto symbolicValue : symbolicValues) {
      auto [name, arg] = hwModule.insertInput(
          hwModule.getNumInputPorts(),
          StringAttr::get(module.getContext(), "symbolic_value"),
          symbolicValue.getType());
      (void)name;
      symbolicValue.getResult().replaceAllUsesWith(arg);
      symbolicValue.erase();
    }
  }

  return success();
}
