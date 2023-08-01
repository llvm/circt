//===- ResolveTraces.cpp - Resolve TraceAnnotations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Find any TraceAnnotations in the design, update their targets, and write the
// annotations out to an output annotation file.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"

#define DEBUG_TYPE "firrtl-resolve-traces"

using namespace circt;
using namespace firrtl;

/// Expand a TraceNameAnnotation (which has don't touch semantics) into a
/// TraceAnnotation (which does NOT have don't touch semantics) and separate
/// DontTouchAnnotations for targets that are not modules, external modules, or
/// instances (as these targets are not valid for a don't touch).
LogicalResult circt::firrtl::applyTraceName(const AnnoPathValue &target,
                                            DictionaryAttr anno,
                                            ApplyState &state) {

  auto *context = anno.getContext();

  NamedAttrList trace, dontTouch;
  for (auto namedAttr : anno.getValue()) {
    if (namedAttr.getName() == "class") {
      trace.append("class", StringAttr::get(context, traceAnnoClass));
      dontTouch.append("class", StringAttr::get(context, dontTouchAnnoClass));
      continue;
    }
    trace.append(namedAttr);

    // When we see the "target", check to see if this is not targeting a module,
    // extmodule, or instance (as these are invalid "don't touch" targets).  If
    // it is not, then add a DontTouchAnnotation.
    if (namedAttr.getName() == "target" &&
        !target.isOpOfType<FModuleOp, FExtModuleOp, InstanceOp>()) {
      dontTouch.append(namedAttr);
      state.addToWorklistFn(DictionaryAttr::getWithSorted(context, dontTouch));
    }
  }

  state.addToWorklistFn(DictionaryAttr::getWithSorted(context, trace));

  return success();
}

struct ResolveTracesPass : public ResolveTracesBase<ResolveTracesPass> {
  using ResolveTracesBase::outputAnnotationFilename;

  void runOnOperation() override;

private:
  /// Stores a pointer to an NLA Table.  This is populated during
  /// runOnOperation.
  NLATable *nlaTable;

  /// Internal implementation that updates an Annotation to add a "target" field
  /// based on the current location of the annotation in the circuit.  The value
  /// of the "target" will be a local target if the Annotation is local and a
  /// non-local target if the Annotation is non-local.
  bool updateTargetImpl(Annotation &anno, FModuleLike &module,
                        FIRRTLBaseType type, StringRef name) {
    if (!anno.isClass(traceAnnoClass))
      return false;

    LLVM_DEBUG(llvm::dbgs() << "  - before: " << anno.getDict() << "\n");

    SmallString<64> newTarget("~");
    newTarget.append(module->getParentOfType<CircuitOp>().getName());
    newTarget.append("|");

    if (auto nla = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
      hw::HierPathOp path = nlaTable->getNLA(nla.getAttr());
      for (auto part : path.getNamepath().getValue().drop_back()) {
        auto inst = cast<hw::InnerRefAttr>(part);
        newTarget.append(inst.getModule());
        newTarget.append("/");
        newTarget.append(inst.getName());
        newTarget.append(":");
      }
    }
    newTarget.append(module.getModuleName());
    newTarget.append(">");

    newTarget.append(name);

    // Add information if this is an aggregate.
    auto targetFieldID = anno.getFieldID();
    while (targetFieldID) {
      FIRRTLTypeSwitch<FIRRTLBaseType>(type)
          .Case<FVectorType>([&](FVectorType vector) {
            auto index = vector.getIndexForFieldID(targetFieldID);
            newTarget.append("[");
            newTarget.append(Twine(index).str());
            newTarget.append("]");
            type = vector.getElementType();
            targetFieldID -= vector.getFieldID(index);
          })
          .Case<BundleType>([&](BundleType bundle) {
            auto index = bundle.getIndexForFieldID(targetFieldID);
            newTarget.append(".");
            newTarget.append(bundle.getElementName(index));
            type = bundle.getElementType(index);
            targetFieldID -= bundle.getFieldID(index);
          })
          .Default([&](auto) { targetFieldID = 0; });
    }

    anno.setMember("target", StringAttr::get(module->getContext(), newTarget));

    LLVM_DEBUG(llvm::dbgs() << "    after:  " << anno.getDict() << "\n");

    return true;
  }

  /// Add a "target" field to a port Annotation that indicates the current
  /// location of the port in the circuit.
  bool updatePortTarget(FModuleLike &module, Annotation &anno,
                        unsigned portIdx) {
    auto type = getBaseType(type_cast<FIRRTLType>(module.getPortType(portIdx)));
    return updateTargetImpl(anno, module, type, module.getPortName(portIdx));
  }

  /// Add a "target" field to an Annotation that indicates the current location
  /// of a component in the circuit.
  bool updateTarget(FModuleLike &module, Operation *op, Annotation &anno) {

    // If this operation doesn't have a name, then do nothing.
    StringAttr name = op->getAttrOfType<StringAttr>("name");
    if (!name)
      return false;

    // Get the type of the operation either by checking for the
    // result targeted by symbols on it (which are used to track the op)
    // or by inspecting its single result.
    auto is = dyn_cast<hw::InnerSymbolOpInterface>(op);
    Type type;
    if (is && is.getTargetResult())
      type = is.getTargetResult().getType();
    else {
      if (op->getNumResults() != 1)
        return false;
      type = op->getResultTypes().front();
    }

    auto baseType = getBaseType(type_cast<FIRRTLType>(type));
    return updateTargetImpl(anno, module, baseType, name);
  }

  /// Add a "target" field to an Annotation on a Module that indicates the
  /// current location of the module.  This will be local or non-local depending
  /// on the Annotation.
  bool updateModuleTarget(FModuleLike &module, Annotation &anno) {

    if (!anno.isClass(traceAnnoClass))
      return false;

    LLVM_DEBUG(llvm::dbgs() << "  - before: " << anno.getDict() << "\n");

    SmallString<64> newTarget("~");
    newTarget.append(module->getParentOfType<CircuitOp>().getName());
    newTarget.append("|");

    if (auto nla = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
      hw::HierPathOp path = nlaTable->getNLA(nla.getAttr());
      for (auto part : path.getNamepath().getValue().drop_back()) {
        auto inst = cast<hw::InnerRefAttr>(part);
        newTarget.append(inst.getModule());
        newTarget.append("/");
        newTarget.append(inst.getName());
        newTarget.append(":");
      }
    }
    newTarget.append(module.getModuleName());

    anno.setMember("target", StringAttr::get(module->getContext(), newTarget));

    LLVM_DEBUG(llvm::dbgs() << "    after:  " << anno.getDict() << "\n");

    return true;
  }
};

void ResolveTracesPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "==----- Running ResolveTraces "
                      "-----------------------------------------------===\n");

  // Populate the NLA Table.
  nlaTable = &getAnalysis<NLATable>();

  // Grab the circuit (as this is used a few times below).
  CircuitOp circuit = getOperation();
  MLIRContext *context = circuit.getContext();

  // Function to find all Trace Annotations in the circuit, add a "target" field
  // to them indicating the current local/non-local target of the operation/port
  // the Annotation is attached to, copy the annotation into an
  // "outputAnnotations" return vector, and delete the original Annotation.  If
  // a component or port is targeted by a Trace Annotation it will be given a
  // symbol to prevent the output Trace Annotation from being made invalid by a
  // later optimization.
  auto onModule = [&](FModuleLike moduleLike) {
    // Output Trace Annotations from this module only.
    SmallVector<Annotation> outputAnnotations;

    // A lazily constructed module namespace.
    std::optional<ModuleNamespace> moduleNamespace;

    // Return a cached module namespace, lazily constructing it if needed.
    auto getNamespace = [&](FModuleLike module) -> ModuleNamespace & {
      if (!moduleNamespace)
        moduleNamespace = ModuleNamespace(module);
      return *moduleNamespace;
    };

    // Visit the module.
    AnnotationSet::removeAnnotations(moduleLike, [&](Annotation anno) {
      if (!updateModuleTarget(moduleLike, anno))
        return false;

      outputAnnotations.push_back(anno);
      return true;
    });

    // Visit port annotations.
    AnnotationSet::removePortAnnotations(
        moduleLike, [&](unsigned portIdx, Annotation anno) {
          if (!updatePortTarget(moduleLike, anno, portIdx))
            return false;

          getOrAddInnerSym(moduleLike, portIdx, getNamespace);
          outputAnnotations.push_back(anno);
          return true;
        });

    // Visit component annotations.
    moduleLike.walk([&](Operation *component) {
      AnnotationSet::removeAnnotations(component, [&](Annotation anno) {
        if (!updateTarget(moduleLike, component, anno))
          return false;

        getOrAddInnerSym(component, getNamespace);
        outputAnnotations.push_back(anno);
        return true;
      });
    });

    return outputAnnotations;
  };

  // Function to append one vector after another.  This is used to merge results
  // from parallel executions of "onModule".
  auto appendVecs = [](auto &&a, auto &&b) {
    a.append(b.begin(), b.end());
    return std::forward<decltype(a)>(a);
  };

  // Process all the modules in parallel or serially, depending on the
  // multithreading context.
  SmallVector<FModuleLike, 0> mods(circuit.getOps<FModuleLike>());
  auto outputAnnotations = transformReduce(
      context, mods, SmallVector<Annotation>{}, appendVecs, onModule);

  // Do not generate an output Annotation file if no Annotations exist.
  if (outputAnnotations.empty())
    return markAllAnalysesPreserved();

  // Write out all the Trace Annotations to a JSON buffer.
  std::string jsonBuffer;
  llvm::raw_string_ostream jsonStream(jsonBuffer);
  llvm::json::OStream json(jsonStream, /*IndentSize=*/2);
  json.arrayBegin();
  for (auto anno : outputAnnotations) {
    json.objectBegin();
    json.attribute("class", anno.getClass());
    json.attribute("target", anno.getMember<StringAttr>("target").getValue());
    json.attribute("chiselTarget",
                   anno.getMember<StringAttr>("chiselTarget").getValue());
    json.objectEnd();
  }
  json.arrayEnd();

  // Write the JSON-encoded Trace Annotation to a file called
  // "$circuitName.anno.json".  (This is implemented via an SVVerbatimOp that is
  // inserted before the FIRRTL circuit.
  OpBuilder b(circuit);
  auto verbatimOp = b.create<sv::VerbatimOp>(b.getUnknownLoc(), jsonBuffer);
  hw::OutputFileAttr fileAttr;
  if (this->outputAnnotationFilename.empty())
    fileAttr = hw::OutputFileAttr::getFromFilename(
        context, circuit.getName() + ".anno.json",
        /*excludeFromFilelist=*/true, false);
  else
    fileAttr = hw::OutputFileAttr::getFromFilename(
        context, outputAnnotationFilename,
        /*excludeFromFilelist=*/true, false);
  verbatimOp->setAttr("output_file", fileAttr);

  return markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass>
circt::firrtl::createResolveTracesPass(StringRef outputAnnotationFilename) {
  auto pass = std::make_unique<ResolveTracesPass>();
  if (!outputAnnotationFilename.empty())
    pass->outputAnnotationFilename = outputAnnotationFilename.str();
  return pass;
}
