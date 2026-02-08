//===- OMReductions.cpp - Reduction patterns for the OM dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/OMReductions.h"
#include "circt/Dialect/OM/OMOps.h"
#include "circt/Reduce/ReductionUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "om-reductions"

using namespace mlir;
using namespace circt;
using namespace om;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace {

/// A utility for caching symbol tables and symbol user maps.
struct SymbolCache {
  SymbolCache() : tables(std::make_unique<SymbolTableCollection>()) {}

  SymbolTable &getSymbolTable(Operation *op) {
    return tables->getSymbolTable(op);
  }

  SymbolTable &getNearestSymbolTable(Operation *op) {
    return getSymbolTable(SymbolTable::getNearestSymbolTable(op));
  }

  SymbolUserMap &getSymbolUserMap(Operation *op) {
    auto it = userMaps.find(op);
    if (it != userMaps.end())
      return it->second;
    return userMaps.insert({op, SymbolUserMap(*tables, op)}).first->second;
  }

  SymbolUserMap &getNearestSymbolUserMap(Operation *op) {
    return getSymbolUserMap(SymbolTable::getNearestSymbolTable(op));
  }

  void clear() {
    tables = std::make_unique<SymbolTableCollection>();
    userMaps.clear();
  }

private:
  std::unique_ptr<SymbolTableCollection> tables;
  llvm::SmallDenseMap<Operation *, SymbolUserMap, 2> userMaps;
};

} // namespace

//===----------------------------------------------------------------------===//
// Reduction Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Replaces om.object instantiations with om.unknown when the object's fields
/// are not accessed (only used in om.any_cast or not used at all).
struct OMObjectToUnknownReplacer : public OpReduction<ObjectOp> {
  void beforeReduction(mlir::ModuleOp op) override { symbols.clear(); }

  void matches(ObjectOp objectOp,
               llvm::function_ref<void(uint64_t, uint64_t)> addMatch) override {
    // Check if the object is only used in om.any_cast operations or not used
    bool onlyAnyCastOrUnused =
        llvm::all_of(objectOp->getUsers(),
                     [](Operation *user) { return isa<om::AnyCastOp>(user); });

    if (!onlyAnyCastOrUnused)
      return;

    // Return a benefit proportional to the number of operands we can eliminate
    addMatch(1 + objectOp.getActualParams().size(), 0);
  }

  LogicalResult rewriteMatches(ObjectOp objectOp,
                               ArrayRef<uint64_t> matches) override {
    OpBuilder builder(objectOp);
    auto unknownOp = om::UnknownValueOp::create(builder, objectOp.getLoc(),
                                                objectOp.getResult().getType());
    objectOp.getResult().replaceAllUsesWith(unknownOp.getResult());
    objectOp->erase();
    return success();
  }

  std::string getName() const override { return "om-object-to-unknown"; }

  SymbolCache symbols;
};

/// Removes unused elements from om.list_create operations.
struct OMListElementPruner : public OpReduction<ListCreateOp> {
  void matches(ListCreateOp listOp,
               llvm::function_ref<void(uint64_t, uint64_t)> addMatch) override {
    // Create one match for each element in the list
    auto inputs = listOp.getInputs();
    for (size_t i = 0; i < inputs.size(); ++i)
      addMatch(1, i);
  }

  LogicalResult rewriteMatches(ListCreateOp listOp,
                               ArrayRef<uint64_t> matches) override {
    // Convert matches to a set for fast lookup
    llvm::SmallDenseSet<uint64_t, 4> matchesSet(matches.begin(), matches.end());

    // Collect inputs that should be kept (not in matches)
    SmallVector<Value> newInputs;
    auto inputs = listOp.getInputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!matchesSet.contains(i))
        newInputs.push_back(inputs[i]);
    }

    // Create a new list with the remaining inputs
    OpBuilder builder(listOp);
    auto newListOp = ListCreateOp::create(
        builder, listOp.getLoc(), listOp.getResult().getType(), newInputs);
    listOp.getResult().replaceAllUsesWith(newListOp.getResult());
    listOp->erase();
    return success();
  }

  std::string getName() const override { return "om-list-element-pruner"; }
};

/// Removes unused output fields from om.class definitions.
struct OMClassFieldPruner : public OpReduction<ClassOp> {
  void beforeReduction(mlir::ModuleOp op) override { symbols.clear(); }

  void matches(ClassOp classOp,
               llvm::function_ref<void(uint64_t, uint64_t)> addMatch) override {
    // Get the field names
    auto fieldNames = classOp.getFieldNames();
    if (fieldNames.empty())
      return;

    // Find which fields are actually accessed via om.object.field
    llvm::DenseSet<StringAttr> usedFields;
    auto &symbolUserMap = symbols.getNearestSymbolUserMap(classOp);

    for (auto *user : symbolUserMap.getUsers(classOp)) {
      if (auto objectOp = dyn_cast<ObjectOp>(user)) {
        // Check all uses of this object
        for (auto *objectUser : objectOp->getUsers()) {
          if (auto fieldOp = dyn_cast<ObjectFieldOp>(objectUser)) {
            // Mark the accessed field as used
            auto fieldPath = fieldOp.getFieldPath();
            if (!fieldPath.empty()) {
              auto symRef = cast<FlatSymbolRefAttr>(fieldPath[0]);
              usedFields.insert(symRef.getAttr());
            }
          }
        }
      }
    }

    // Create one match for each unused field
    for (auto [idx, fieldName] : llvm::enumerate(fieldNames)) {
      if (!usedFields.contains(cast<StringAttr>(fieldName)))
        addMatch(1, idx);
    }
  }

  LogicalResult rewriteMatches(ClassOp classOp,
                               ArrayRef<uint64_t> matches) override {
    // Convert matches to a set for fast lookup
    llvm::SmallDenseSet<uint64_t, 4> matchesSet(matches.begin(), matches.end());

    // Build new field lists with only fields not in matches
    auto fieldsOp = classOp.getFieldsOp();
    auto oldFieldNames = classOp.getFieldNames();
    auto oldFieldValues = fieldsOp.getFields();

    SmallVector<Attribute> newFieldNames;
    SmallVector<Value> newFieldValues;
    SmallVector<NamedAttribute> newFieldTypes;

    for (auto [idx, nameValue] :
         llvm::enumerate(llvm::zip(oldFieldNames, oldFieldValues))) {
      if (!matchesSet.contains(idx)) {
        auto [name, value] = nameValue;
        auto nameAttr = cast<StringAttr>(name);
        newFieldNames.push_back(name);
        newFieldValues.push_back(value);
        newFieldTypes.push_back(
            NamedAttribute(nameAttr, TypeAttr::get(value.getType())));
      }
    }

    // Update the class
    OpBuilder builder(classOp);
    classOp.setFieldNamesAttr(builder.getArrayAttr(newFieldNames));
    classOp.setFieldTypesAttr(builder.getDictionaryAttr(newFieldTypes));
    fieldsOp.getFieldsMutable().assign(newFieldValues);

    return success();
  }

  std::string getName() const override { return "om-class-field-pruner"; }

  SymbolCache symbols;
};

/// Removes unused parameters from om.class definitions.
struct OMClassParameterPruner : public OpReduction<ClassOp> {
  void beforeReduction(mlir::ModuleOp op) override { symbols.clear(); }

  void matches(ClassOp classOp,
               llvm::function_ref<void(uint64_t, uint64_t)> addMatch) override {
    auto *bodyBlock = classOp.getBodyBlock();
    if (bodyBlock->getNumArguments() == 0)
      return;

    // Create one match for each unused parameter
    for (auto [idx, arg] : llvm::enumerate(bodyBlock->getArguments())) {
      if (arg.use_empty())
        addMatch(1, idx);
    }
  }

  LogicalResult rewriteMatches(ClassOp classOp,
                               ArrayRef<uint64_t> matches) override {
    // Convert matches to a set for fast lookup
    llvm::SmallDenseSet<uint64_t, 4> matchesSet(matches.begin(), matches.end());

    auto *bodyBlock = classOp.getBodyBlock();

    // Collect all object instantiations that need to be updated
    // We need to walk all operations in the module because om.class operations
    // are IsolatedFromAbove, so the SymbolUserMap doesn't find nested uses
    SmallVector<ObjectOp> objectsToUpdate;
    auto moduleOp = classOp->getParentOfType<mlir::ModuleOp>();
    moduleOp.walk([&](ObjectOp objectOp) {
      if (objectOp.getClassNameAttr() == classOp.getSymNameAttr())
        objectsToUpdate.push_back(objectOp);
    });

    // Update the class formal parameters FIRST, before updating instantiations
    auto oldParamNames = classOp.getFormalParamNames();
    SmallVector<Attribute> newParamNames;
    for (auto [idx, name] : llvm::enumerate(oldParamNames)) {
      if (!matchesSet.contains(idx))
        newParamNames.push_back(name);
    }

    OpBuilder builder(classOp);
    classOp.setFormalParamNamesAttr(builder.getArrayAttr(newParamNames));

    // Remove the block arguments in reverse order to maintain indices
    SmallVector<unsigned> indicesToRemove(matches.begin(), matches.end());
    llvm::sort(indicesToRemove, std::greater<unsigned>());
    for (unsigned idx : indicesToRemove)
      bodyBlock->eraseArgument(idx);

    // Now update all om.object instantiations to match the new signature
    for (auto objectOp : objectsToUpdate) {
      // Build new actual parameters without the removed parameters
      SmallVector<Value> newParams;
      for (auto [idx, param] : llvm::enumerate(objectOp.getActualParams())) {
        if (!matchesSet.contains(idx))
          newParams.push_back(param);
      }

      // Create new object op
      OpBuilder builder(objectOp);
      auto newObjectOp = ObjectOp::create(
          builder, objectOp.getLoc(), objectOp.getResult().getType(),
          objectOp.getClassNameAttr(), newParams);
      objectOp.getResult().replaceAllUsesWith(newObjectOp.getResult());
      objectOp->erase();
    }

    return success();
  }

  std::string getName() const override { return "om-class-parameter-pruner"; }

  SymbolCache symbols;
};

/// Removes unused om.class definitions that are never instantiated.
struct OMUnusedClassRemover : public OpReduction<ClassOp> {
  void beforeReduction(mlir::ModuleOp op) override { symbols.clear(); }

  void matches(ClassOp classOp,
               llvm::function_ref<void(uint64_t, uint64_t)> addMatch) override {
    // Check if this class is ever instantiated via om.object
    // We need to walk all operations because SymbolUserMap doesn't find
    // nested symbol uses inside IsolatedFromAbove operations like om.class
    auto moduleOp = classOp->getParentOfType<mlir::ModuleOp>();
    auto className = classOp.getSymNameAttr();
    bool hasObjectInstantiation = false;
    bool containsObjectInstantiation = false;

    // Check if this class is instantiated via om.object anywhere
    moduleOp.walk([&](ObjectOp objectOp) {
      if (objectOp.getClassNameAttr() == className) {
        hasObjectInstantiation = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    // Check if this class contains any om.object instantiations
    // (classes that instantiate other classes should be kept as they might be
    // entry points)
    classOp.walk([&](ObjectOp objectOp) {
      containsObjectInstantiation = true;
      return WalkResult::interrupt();
    });

    // Remove the class if:
    // 1. It's never instantiated via om.object, AND
    // 2. It doesn't contain any om.object instantiations (not an entry point)
    if (!hasObjectInstantiation && !containsObjectInstantiation)
      addMatch(10, 0);
  }

  LogicalResult rewriteMatches(ClassOp classOp,
                               ArrayRef<uint64_t> matches) override {
    classOp->erase();
    return success();
  }

  std::string getName() const override { return "om-unused-class-remover"; }

  SymbolCache symbols;
};

/// Simplifies om.unknown -> om.any_cast chains by replacing with a direct
/// om.unknown of the target type.
struct OMAnyCastOfUnknownSimplifier : public OpReduction<om::AnyCastOp> {
  void matches(om::AnyCastOp anyCastOp,
               llvm::function_ref<void(uint64_t, uint64_t)> addMatch) override {
    // Check if the input is an om.unknown
    if (auto unknownOp =
            anyCastOp.getInput().getDefiningOp<om::UnknownValueOp>())
      addMatch(2, 0);
  }

  LogicalResult rewriteMatches(om::AnyCastOp anyCastOp,
                               ArrayRef<uint64_t> matches) override {
    OpBuilder builder(anyCastOp);
    auto unknownOp = om::UnknownValueOp::create(
        builder, anyCastOp.getLoc(), anyCastOp.getResult().getType());
    anyCastOp.getResult().replaceAllUsesWith(unknownOp.getResult());
    anyCastOp->erase();
    return success();
  }

  std::string getName() const override {
    return "om-anycast-of-unknown-simplifier";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Reduction Registration
//===----------------------------------------------------------------------===//

void om::OMReducePatternDialectInterface::populateReducePatterns(
    circt::ReducePatternSet &patterns) const {
  // High priority reductions
  patterns.add<OMClassParameterPruner, 50>();
  patterns.add<OMClassFieldPruner, 49>();
  patterns.add<OMObjectToUnknownReplacer, 48>();
  patterns.add<OMListElementPruner, 45>();

  // Medium priority reductions
  patterns.add<OMUnusedClassRemover, 40>();
  patterns.add<OMAnyCastOfUnknownSimplifier, 35>();
}

void om::registerReducePatternDialectInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, OMDialect *dialect) {
    dialect->addInterfaces<OMReducePatternDialectInterface>();
  });
}
