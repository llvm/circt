//===- MaterializeDebugInfo.cpp - DI materialization ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Support/Naming.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Parallel.h"

#include "circt/Dialect/FIRRTL/AnnotationDetails.h"

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "materialize-di"

using namespace mlir;
using namespace circt;
using namespace firrtl;

namespace {
/// Internal struct to represent a tywaves annotation.
struct TywavesAnnotation {
  // StringAttr target;// Used here?
  StringAttr typeName;
  ArrayAttr params;

  TywavesAnnotation(StringAttr typeName)
      : TywavesAnnotation(typeName, ArrayAttr()) {}

  TywavesAnnotation(StringAttr typeName, ArrayAttr params)
      : typeName(typeName), params(params) {}

  // Define the << operator (debug)
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const TywavesAnnotation &tywavesAnno) {
    os << "  TywavesAnnotation { " << "\n\ttypeName: " << tywavesAnno.typeName
       << "\n\tparams  : " << tywavesAnno.params << "\n  }";
    return os;
  }
};
struct MaterializeDebugInfoPass
    : public MaterializeDebugInfoBase<MaterializeDebugInfoPass> {
  void runOnOperation() override;
  void materializeVariable(OpBuilder &builder, StringAttr name, Value value,
                           const mlir::ArrayAttr &annoList);

  std::optional<TywavesAnnotation> getTywavesInfo(Operation *op);

  std::optional<TywavesAnnotation>
  getTywavesInfo(const mlir::ArrayAttr &annoList, const mlir::Location &loc);

  std::pair<Value, std::optional<TywavesAnnotation>>
  convertToDebugAggregates(OpBuilder &builder, Value value, StringAttr name,
                           const mlir::ArrayAttr &annoList,
                           const mlir::ArrayAttr &inputFilteredAnnoList,
                           unsigned int circtSubFieldId = 0);
};

//===----------------------------------------------------------------------===//
// Helper functions to process and filter annotations
//===----------------------------------------------------------------------===//
namespace annohelper {

/// Get the list of firrtl annotations from an operation.
mlir::ArrayAttr getAnnotationList(Operation *op) {
  return dyn_cast_or_null<mlir::ArrayAttr>(op->getAttr("annotations"));
}

/// Filter the input list of annotations by a circt.fieldID attribute. The
/// function returns the annotation (if any) from the input list that has the
/// same fieldID passed as argument.
/// This helps to find the annotation associated to a subfield target.
mlir::ArrayAttr
filterAnnotationsByCirctFieldID(OpBuilder &builder, int fieldId,
                                const mlir::ArrayAttr &annoList) {

  // Filter the annotation list through the index: fieldId == index
  SmallVector<Attribute> filtered;
  for (const auto &annoAttr : annoList)
    // 1. Convert the attribute into a dictionary
    if (const auto &annoDict = dyn_cast<mlir::DictionaryAttr>(annoAttr))
      // 2. Compare the iterated fieldId with the input fieldId
      if (const auto &idxAttr = annoDict.get("circt.fieldID"))
        if (const auto &idx = dyn_cast<IntegerAttr>(idxAttr);
            idx && idx.getInt() == fieldId) {
          // 3. Add the matchin Id to the filtered annotation list
          filtered.push_back(annoAttr);
        }

  auto result = builder.getArrayAttr(filtered);

  // Debug
  LLVM_DEBUG(llvm::dbgs() << " ============================================\n");
  LLVM_DEBUG(llvm::dbgs() << "  - Filtering fieldId: " << fieldId << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  - Filtered anno list: " << result << "\n");
  LLVM_DEBUG(llvm::dbgs() << " ============================================\n");

  return result;
}

/// Return the total numer of subfields nested into a FIRRTL type.
int getNumSubFields(const FIRRTLBaseType &type) {
  return FIRRTLTypeSwitch<Type, int>(type)
      .Case<BundleType>([&](auto type) {
        int count = 1;
        // Count all the elements of a bundle: in face each element is marked
        // with an annotation
        for (const auto &element : type.getElements())
          count += getNumSubFields(element.type);

        return count;
      })
      .Case<FVectorType>([&](auto type) {
        return 1 +
               type.getNumElements() * getNumSubFields(type.getElementType());
      })
      .Case<FIRRTLBaseType>([&](auto type) { return 1; }) // Ground type
      .Default([&](auto type) { return 0; });
}
} // namespace annohelper
} // namespace

/// Get the Tywaves annotation in a structured representation from a single
/// target representation. It contains the source language type information.
std::optional<TywavesAnnotation>
MaterializeDebugInfoPass::getTywavesInfo(Operation *op) {
  if (auto annoListArray = annohelper::getAnnotationList(op))
    return getTywavesInfo(annoListArray, op->getLoc());

  // No annotation found
  return std::nullopt;
}

/// Get the Tywaves annotation from a list of annotations. It assumes the
/// list comes from a single target operation.
///
/// The function should be called on the final target operation.
/// For aggregates, it should be called on each sub-operation singularly.
///
/// @param annoList The list of annotations associated with the target
/// operation.
/// @param loc The location of the target operation. For error report.
std::optional<TywavesAnnotation>
MaterializeDebugInfoPass::getTywavesInfo(const mlir::ArrayAttr &annoList,
                                         const mlir::Location &loc) {

  auto foundOneTywavesAnnoClass = false;
  std::optional<TywavesAnnotation> result = std::nullopt;

  // 1. Filter tywaves annotations
  for (const auto &annoAttr : annoList)
    if (const auto &annoDict = dyn_cast<mlir::DictionaryAttr>(annoAttr))
      if (annoDict.template getAs<mlir::StringAttr>("class") ==
          tywavesAnnoClass) {

        // Check: there should be only one tywave annotation per target
        if (foundOneTywavesAnnoClass) {
          mlir::emitError(loc)
              << "Expected a single tywaves annotation. Found: "
              << annoList.size() << "\n";
          signalPassFailure();
          return std::nullopt;
        }
        foundOneTywavesAnnoClass = true;

        // 2. Extract useful information: target, typeName, params (if
        // any)
        if (!annoDict.contains("typeName")) {
          mlir::emitError(loc)
              << "Missing expected typeName in tywaves annotation.";
          signalPassFailure();
          return std::nullopt;
        }
        // const auto &target = annoDict.get("target").cast<StringAttr>();
        const auto &typeName = cast<StringAttr>(annoDict.get("typeName"));
        if (const auto &params = annoDict.get("params")) {

          if (!isa<mlir::ArrayAttr>(params)) {

            mlir::emitError(loc) << "Parameters are expressed in the wrong "
                                    "format, expected array. Found: "
                                 << params << "\n";
            signalPassFailure();
            return std::nullopt;
          }

          auto paramsArr = cast<mlir::ArrayAttr>(params);
          // It must contain name, type, value: check them singularly
          for (const auto &param : paramsArr) {
            if (auto paramDict = dyn_cast<mlir::DictionaryAttr>(param);
                !paramDict || (!paramDict.contains("name") ||
                               !paramDict.contains("typeName"))) {
              mlir::emitError(loc)
                  << "Parameters are missing one of the following "
                     "fields: name, typeName. Found: "
                  << params << "\n";
              signalPassFailure();
              return std::nullopt;
            }
          }

          // 3. Create the TywavesAnnotation: with params
          result = TywavesAnnotation(typeName, paramsArr);
        } else {
          // 3. Create the TywavesAnnotation: without params
          result = TywavesAnnotation(typeName);
        }
      }
  // TODO: use this control outside the function
  // if (!result) {
  //   mlir::emitError(loc) << "Missing expected tywaves annotation for "
  //   << op
  //                        << "\n";
  //   signalPassFailure();
  // }

  // Return the result or std::nullopt
  return result;
}

//===----------------------------------------------------------------------===//
// MaterializeDebugInfoPass
//===----------------------------------------------------------------------===//

void MaterializeDebugInfoPass::runOnOperation() {
  auto module = getOperation();
  auto builder = OpBuilder::atBlockBegin(module.getBodyBlock());

  // Create DI variables for each port.
  for (const auto &[port, value] :
       llvm::zip(module.getPorts(), module.getArguments())) {

    materializeVariable(builder, port.name, value,
                        port.annotations.getArrayAttr());

    // Remove all the tywaves annotations from the port: prevent errors during
    // LowerSignatures transfomation
    port.annotations.removePortAnnotations(
        module, [&](int portId, Annotation anno) -> bool {
          return anno.getClass().equals(tywavesAnnoClass);
        });
  }

  // Create DI variables for each declaration in the module body.
  module.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op).Case<WireOp, NodeOp, RegOp, RegResetOp>(
        [&](auto op) {
          builder.setInsertionPointAfter(op);
          materializeVariable(builder, op.getNameAttr(), op.getResult(),
                              annohelper::getAnnotationList(op));

          // TODO: Maybe remove the annotations from the operation since done
          // with ports? clearAnnotationList(op);
          // op->removeAttr("annotations");
        });
  });
}

/// Materialize debug variable ops for a value.
void MaterializeDebugInfoPass::materializeVariable(
    OpBuilder &builder, StringAttr name, Value value,
    const mlir::ArrayAttr &annoList) {
  if (!name || isUselessName(name.getValue()))
    return;
  if (name.getValue().starts_with("_"))
    return;

  if (auto [dbgValue, tywavesAnno] = convertToDebugAggregates(
          builder, value, name, annoList, mlir::ArrayAttr{}, 0);
      dbgValue) {
    // LLVM creating variable
    LLVM_DEBUG(llvm::dbgs() << "  - CREATING TOP VARIABLE: " << name << "\n");

    auto typeName = tywavesAnno ? tywavesAnno->typeName : StringAttr{};
    auto params = tywavesAnno ? tywavesAnno->params : ArrayAttr{};
    builder.create<debug::VariableOp>(value.getLoc(), name, dbgValue,
                                      /*typeName=*/typeName,
                                      /*params=*/params,
                                      /*scope=*/Value{});
  }
}

/// Unpack all aggregates in a FIRRTL value and repack them as debug
/// aggregates. For example, converts a FIRRTL vector `v` into `dbg.array
/// [v[0],v[1],...]`.
///
/// @param annoList is the fill list of annotations for the current value.
/// @param maybeSingleAnno the candidate annotation for the current value used
/// in the recursive call (should contain only one element).
/// @param circtSubFieldId the current searched field ID of the current value.
std::pair<Value, std::optional<TywavesAnnotation>>
MaterializeDebugInfoPass::convertToDebugAggregates(
    OpBuilder &builder, Value value, StringAttr name,
    const mlir::ArrayAttr &annoList, const mlir::ArrayAttr &maybeSingleAnno,
    unsigned int circtSubFieldId) {

  // 1. Extract the tywaves annotation from the current value
  std::optional<TywavesAnnotation> tywavesAnno = std::nullopt;
  if (maybeSingleAnno && !maybeSingleAnno.empty()) {
    // If maybeSingleAnno is not empty, it means that the current value may be
    // part of an aggregate, and maybeSingleAnno has been filtered during a
    // previous recursive call.
    tywavesAnno = getTywavesInfo(maybeSingleAnno, value.getLoc());
  } else {
    // If maybeSingleAnno is empty or not defined, it means that the current
    // value is either a ground type or the aggregate itself, for example:
    //  - a of --> x : !firrtl.uint<8>
    //  - b of --> b : !firrtl.bundle<a: uint<8>, b: sint<8>>
    SmallVector<Attribute> filteredAnnoList;
    if (annoList)
      for (const auto &annoAttr : annoList) {
        if (const auto &annoDict = dyn_cast<mlir::DictionaryAttr>(annoAttr);
            annoDict && !annoDict.contains("circt.fieldID"))
          filteredAnnoList.push_back(annoAttr);
      }

    tywavesAnno =
        getTywavesInfo(builder.getArrayAttr(filteredAnnoList), value.getLoc());
  }

  LLVM_DEBUG(llvm::dbgs() << "  - CURRENT VALUE: " << value << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  - EXTRACTED TYWAVES INFO:\n"
                          << tywavesAnno << "\n");

  // 2. Extract the source language type information
  const auto typeName = tywavesAnno ? tywavesAnno->typeName : StringAttr{};
  const auto params = tywavesAnno ? tywavesAnno->params : ArrayAttr();
  // This enables the creation of a SubFieldOp for the current value (only if
  // child of an aggregate and tywavesAnno is not null/none)
  auto isChildOfAggregate = circtSubFieldId && tywavesAnno;

  // 3. Generate the result

  // Lambda function to generate the result or create a SubFieldOp if it is a
  // subfield of an aggregate.
  auto generateResult = [&builder, &typeName,
                         &params](bool buildSubFieldOp, const Value &result,
                                  const Location &loc, const StringAttr &name) {
    if (buildSubFieldOp) {
      Value x = builder.create<debug::SubFieldOp>(loc, name, result,
                                                  /*typeName=*/typeName,
                                                  /*params=*/params);
      return x;
    }
    return result;
  };

  auto result =
      FIRRTLTypeSwitch<Type, Value>(value.getType())
          .Case<BundleType>([&](auto type) {
            SmallVector<Value> fields;
            SmallVector<Attribute> names;
            SmallVector<Operation *> subOps;

            for (auto [index, element] : llvm::enumerate(type.getElements())) {
              auto subOp =
                  builder.create<SubfieldOp>(value.getLoc(), value, index);
              subOps.push_back(subOp);

              // Filter the annotation list through the index:
              // circtSubFieldId == index
              auto filteredAnnoList =
                  annohelper::filterAnnotationsByCirctFieldID(
                      builder, circtSubFieldId + 1, annoList);
              // TODO: Now I'm using the full name, I may want just the
              // variable name, since I can build it while inspecting the
              // hierarchy of the elements
              auto completeName = builder.getStringAttr(
                  name.getValue() + "." + element.name.getValue());

              if (auto dbgValue = convertToDebugAggregates(
                                      builder, subOp, completeName, annoList,
                                      filteredAnnoList, circtSubFieldId + 1)
                                      .first) {
                fields.push_back(dbgValue);
                names.push_back(element.name);
              }
              // Update the circtSubFieldId with the number of subfields of
              // the current element
              circtSubFieldId += annohelper::getNumSubFields(element.type);
            }

            Value result = builder.create<debug::StructOp>(
                value.getLoc(), fields, builder.getArrayAttr(names));
            for (auto *subOp : subOps)
              if (subOp->use_empty())
                subOp->erase();

            return generateResult(isChildOfAggregate, result, value.getLoc(),
                                  name);
          })
          .Case<FVectorType>([&](auto type) -> Value {
            SmallVector<Value> elements;
            SmallVector<Operation *> subOps;

            for (unsigned index = 0; index < type.getNumElements(); ++index) {
              auto subOp =
                  builder.create<SubindexOp>(value.getLoc(), value, index);
              subOps.push_back(subOp);

              // Filter the annotation list through the index:
              // circtSubFieldId == index
              if (index == 0)
                circtSubFieldId++;
              auto filteredAnnoList =
                  annohelper::filterAnnotationsByCirctFieldID(
                      builder, circtSubFieldId, annoList);
              auto completeName =
                  name.getValue() + "[" + std::to_string(index) + "]";

              if (auto dbgValue =
                      convertToDebugAggregates(
                          builder, subOp, builder.getStringAttr(completeName),
                          annoList, filteredAnnoList, circtSubFieldId)
                          .first)
                elements.push_back(dbgValue);
            }

            Value result;
            if (!elements.empty() && elements.size() == type.getNumElements())
              result = builder.create<debug::ArrayOp>(value.getLoc(), elements);
            for (auto *subOp : subOps)
              if (subOp->use_empty())
                subOp->erase();

            return generateResult(isChildOfAggregate, result, value.getLoc(),
                                  name);
          })
          .Case<FIRRTLBaseType>([&](auto type) {
            if (type.getBitWidthOrSentinel() < 1)
              return Value{};
            // Emit the result directly
            return type.isGround() ? generateResult(isChildOfAggregate, value,
                                                    value.getLoc(), name)
                                   : Value{};
          })
          .Default({});

  return std::make_pair(result, tywavesAnno);
}

std::unique_ptr<Pass> firrtl::createMaterializeDebugInfoPass() {
  return std::make_unique<MaterializeDebugInfoPass>();
}
