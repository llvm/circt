//===- VerifyObjectFields.cpp - Verify Object fields -------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the OM verify object fields pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Dialect/OM/OMAttributes.h"
#include "circt/Dialect/OM/OMOps.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

namespace circt {
namespace om {
#define GEN_PASS_DEF_VERIFYOBJECTFIELDS
#include "circt/Dialect/OM/OMPasses.h.inc"
} // namespace om
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace om;

namespace {
struct VerifyObjectFieldsPass
    : public circt::om::impl::VerifyObjectFieldsBase<VerifyObjectFieldsPass> {
  void runOnOperation() override;
  bool canScheduleOn(RegisteredOperationName opName) const override {
    return opName.getStringRef() == "firrtl.circuit" ||
           opName.getStringRef() == "builtin.module";
  }
};
} // namespace

void VerifyObjectFieldsPass::runOnOperation() {
  auto *module = getOperation();
  assert(module->getNumRegions() == 1 &&
         module->hasTrait<OpTrait::SymbolTable>() &&
         "op must have a single region and symbol table trait");
  auto &symbolTable = getAnalysis<SymbolTable>();

  // Run actual verification. Make sure not to mutate `tables`.
  auto result = mlir::failableParallelForEach(
      &getContext(), module->getRegion(0).getOps<om::ClassLike>(),
      [&symbolTable](ClassLike classLike) {
        auto result =
            classLike.walk([&](ObjectFieldOp objectField) -> WalkResult {
              auto objectInstType =
                  cast<ClassType>(objectField.getObject().getType());
              ClassLike classDef = symbolTable.lookup<ClassLike>(
                  objectInstType.getClassName().getAttr());
              if (!classDef) {
                objectField.emitError()
                    << "class " << objectInstType.getClassName()
                    << " was not found";
                return WalkResult::interrupt();
              }

              auto field = objectField.getFieldAttr();
              std::optional<Type> fieldTypeOpt = classDef.getFieldType(field);
              if (!fieldTypeOpt.has_value()) {
                auto error =
                    objectField.emitOpError("referenced non-existent field ")
                    << field;
                error.attachNote(classDef.getLoc()) << "class defined here";
                return WalkResult::interrupt();
              }

              // Verify the accessed field type matches the result type.
              if (*fieldTypeOpt != objectField.getResult().getType()) {
                objectField.emitOpError("expected type ")
                    << objectField.getResult().getType()
                    << ", but accessed field has type " << *fieldTypeOpt;

                return WalkResult::interrupt();
              }
              return WalkResult::advance();
            });
        return LogicalResult::failure(result.wasInterrupted());
      });
  if (failed(result))
    return signalPassFailure();
  return markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass> circt::om::createVerifyObjectFieldsPass() {
  return std::make_unique<VerifyObjectFieldsPass>();
}
