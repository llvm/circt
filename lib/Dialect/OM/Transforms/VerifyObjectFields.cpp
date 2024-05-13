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

#include "PassDetails.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Dialect/OM/OMAttributes.h"
#include "circt/Dialect/OM/OMOps.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace circt;
using namespace om;

namespace {
struct VerifyObjectFieldsPass
    : public VerifyObjectFieldsBase<VerifyObjectFieldsPass> {
  void runOnOperation() override;
};
} // namespace

void VerifyObjectFieldsPass::runOnOperation() {
  auto module = getOperation();
  assert(module->getNumRegions() == 1 && module->hasTrait<OpTrait::SymbolTable>());
  auto &symbolTable = getAnalysis<SymbolTable>();

  /// A map from a class and field name to a field.
  llvm::MapVector<ClassLike, llvm::DenseMap<StringAttr, ClassFieldLike>> tables;
  for (auto op : module->getRegion(0).getOps<om::ClassLike>())
    tables.insert({op, llvm::DenseMap<StringAttr, ClassFieldLike>()});

  // Peel tables parallelly.
  mlir::parallelForEach(&getContext(), tables, [](auto &entry) {
    ClassLike classLike = entry.first;
    auto &table = entry.second;
    classLike.walk([&](ClassFieldLike fieldLike) {
      table.insert({fieldLike.getNameAttr(), fieldLike});
    });
  });

  // Run actual verification. Make sure not to mutate `tables`.
  auto result = mlir::failableParallelForEach(
      &getContext(), tables, [&tables, &symbolTable](const auto &entry) {
        ClassLike classLike = entry.first;
        auto result =
            classLike.walk([&](ObjectFieldOp objectField) -> WalkResult {
              ObjectOp objectInst =
                  objectField.getObject().getDefiningOp<ObjectOp>();
              ClassLike classDef =
                  cast<ClassLike>(symbolTable.lookupNearestSymbolFrom(
                      objectField, objectInst.getClassNameAttr()));

              // Traverse the field path, verifying each field exists.
              ClassFieldLike finalField;
              auto fields = SmallVector<FlatSymbolRefAttr>(
                  objectField.getFieldPath().getAsRange<FlatSymbolRefAttr>());
              for (size_t i = 0, e = fields.size(); i < e; ++i) {
                // Verify the field exists on the ClassOp.
                auto field = fields[i];
                ClassFieldLike fieldDef;
                auto it = tables.find(classDef);
                assert(it != tables.end() && "must be vistied");
                fieldDef = it->second.lookup(field.getAttr());

                if (!fieldDef) {
                  auto error =
                      objectField.emitOpError("referenced non-existant field ")
                      << field;
                  error.attachNote(classDef.getLoc()) << "class defined here";
                  return WalkResult::interrupt();
                }

                // If there are more fields, verify the current field is of
                // ClassType, and look up the ClassOp for that field.
                if (i < e - 1) {
                  auto classType = dyn_cast<ClassType>(fieldDef.getType());
                  if (!classType) {
                    objectField.emitOpError("nested field access into ")
                        << field << " requires a ClassType, but found "
                        << fieldDef.getType();
                    return WalkResult::interrupt();
                  }

                  // The nested ClassOp must exist, since a field with ClassType
                  // must be an ObjectInstOp, which already verifies the class
                  // exists.
                  classDef =
                      cast<ClassLike>(symbolTable.lookupNearestSymbolFrom(
                          objectField, classType.getClassName()));

                  // Proceed to the next field in the path.
                  continue;
                }

                // On the last iteration down the path, save the final field
                // being accessed.
                finalField = fieldDef;
              }

              // Verify the accessed field type matches the result type.
              if (finalField.getType() != objectField.getResult().getType()) {
                objectField.emitOpError("expected type ")
                    << objectField.getResult().getType()
                    << ", but accessed field has type " << finalField.getType();

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
