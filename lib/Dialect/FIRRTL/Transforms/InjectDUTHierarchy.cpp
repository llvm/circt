//===- InjectDUTHierarchy.cpp - Add hierarchy above the DUT ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the SiFive transform InjectDUTHierarchy.  This moves all
// the logic inside the DUT into a new module named using an annotation.
//
// As the terminology in this pass is a constant source of confusion, FIGURE 1
// below, and the accompanying description of terms is provided to clarify what
// is going on.
//
//         BEFORE                 AFTER
//     +-------------+       +-------------+
//     |     DUT     |       |     DUT     |
//     |             |       | +---------+ |
//     |             |       | | WRAPPER | |
//     |             | ====> | |         | |
//     |    LOGIC    |       | |  LOGIC  | |
//     |             |       | +---------+ |
//     +-------------+       +-------------+
//
// FIGURE 1: A graphical view of this pass
//
// In FIGURE 1, the DUT (design under test which roughly is the unit of
// compilation excluding all testharness, testbenches, or tests) is the module
// in the circuit which is annotated with a `MarkDUTAnnotation`.  Inside the
// DUT, there exists some LOGIC.  This is the complete contents of all
// operations inside the DUT module.  Logically, this pass takes the LOGIC and
// puts it into a new MODULE, called the WRAPPER.
//
// This pass operates in two modes, moderated by a `moveDut` boolean parameter
// on the controlling annotation.  When `moveDut=false`, the DUT in FIGURE 1 is
// treated as the design under test.  When `moveDut=true`, the WRAPPER in FIGURE
// 1 is treated as the design under test.  Mechanically, this means that
// annotations on the DUT will be moved to the wrapper.
//
// The names of the WRAPPER and DUT change based on the mode.  If
// `moveDut=false`, then the WRAPPER is named using the `name` field of the
// controlling annotation and the DUT gets the name of the original DUT.  If
// `moveDut=true`, then the DUT is named using the `name` field and the WRAPPER
// gets the name of the original DUT.
//
// This pass is closely coupled to `ExtractInstances`.  This pass is intended to
// be used to create a "space" where modules can be extracted or groups of
// modules can be extracted to.  Commonly, the LOGIC will be extracted to the
// WRAPPER, memories will be extracted to a MEMORIES module, and other things
// (clock gates and blackboxes) will be extracted to other modules.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Support/Debug.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-inject-dut-hier"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_INJECTDUTHIERARCHY
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
struct InjectDUTHierarchy
    : public circt::firrtl::impl::InjectDUTHierarchyBase<InjectDUTHierarchy> {
  void runOnOperation() override;
};
} // namespace

/// Add an extra level of hierarchy to a hierarchical path that places the
/// wrapper instance after the DUT.  This appends to the existing path
/// immediately after the `dut`.
///
/// E.g., this is converting:
///
///   firrtl.hierpath [@Top::@dut, @DUT]
///
/// Into:
///
///   firrtl.hierpath [@Top::@dut, @DUT::@wrapper, @Wrapper]
///
/// The `oldDutNameAttr` parameter controls the insertion point.  I.e., this is
/// the location of a module wehre an insertion will happen.  By separating this
/// from the `dut` paramter, this allows this function to work for both the
/// `moveDut=false` and `moveDut=true` cases.  In the `moveDut=false` case the
/// `dut` and `oldDutNameAttr` refer to the same module.
static void addHierarchy(hw::HierPathOp path, FModuleOp dut,
                         InstanceOp wrapperInst, StringAttr oldDutNameAttr) {

  auto namepath = path.getNamepath().getValue();

  size_t nlaIdx = 0;
  SmallVector<Attribute> newNamepath;
  newNamepath.reserve(namepath.size() + 1);
  while (path.modPart(nlaIdx) != oldDutNameAttr)
    newNamepath.push_back(namepath[nlaIdx++]);
  newNamepath.push_back(hw::InnerRefAttr::get(dut.getModuleNameAttr(),
                                              getInnerSymName(wrapperInst)));

  // Add the extra level of hierarchy.
  if (auto dutRef = dyn_cast<hw::InnerRefAttr>(namepath[nlaIdx]))
    newNamepath.push_back(hw::InnerRefAttr::get(
        wrapperInst.getModuleNameAttr().getAttr(), dutRef.getName()));
  else
    newNamepath.push_back(
        FlatSymbolRefAttr::get(wrapperInst.getModuleNameAttr().getAttr()));

  // Add anything left over.
  auto back = namepath.drop_front(nlaIdx + 1);
  newNamepath.append(back.begin(), back.end());
  path.setNamepathAttr(ArrayAttr::get(dut.getContext(), newNamepath));
}

void InjectDUTHierarchy::runOnOperation() {
  CIRCT_DEBUG_SCOPED_PASS_LOGGER(this);

  CircuitOp circuit = getOperation();

  /// The wrapper module that is created inside the DUT to house all its logic.
  FModuleOp wrapper;

  /// The name of the new module to create under the DUT.
  StringAttr wrapperName;

  /// If true, then move the MarkDUTAnnotation to the newly created module.
  bool moveDut = false;

  /// Mutable indicator that an error occurred for some reason.  If this is ever
  /// true, then the pass can just signalPassFailure.
  bool error = false;

  // Do not remove the injection annotation as this is necessary to additionally
  // influence ExtractInstances.
  for (Annotation anno : AnnotationSet(circuit)) {
    if (!anno.isClass(injectDUTHierarchyAnnoClass))
      continue;

    auto name = anno.getMember<StringAttr>("name");
    if (!name) {
      emitError(circuit->getLoc())
          << "contained a malformed "
             "'sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation' "
             "annotation that did not contain a 'name' field";
      error = true;
      continue;
    }

    if (wrapperName) {
      emitError(circuit->getLoc())
          << "contained multiple "
             "'sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation' "
             "annotations when at most one is allowed";
      error = true;
      continue;
    }

    wrapperName = name;
    if (auto moveDutAnnoAttr = anno.getMember<BoolAttr>("moveDut"))
      moveDut = moveDutAnnoAttr.getValue();
  }

  if (error)
    return signalPassFailure();

  // The prerequisites for the pass to run were not met.  Indicate that no work
  // was done and exit.
  if (!wrapperName)
    return markAllAnalysesPreserved();

  // A DUT must exist in order to continue.  The pass could silently ignore this
  // case and do nothing, but it is better to provide an error.
  InstanceInfo &instanceInfo = getAnalysis<InstanceInfo>();
  if (!instanceInfo.getDut()) {
    emitError(circuit->getLoc())
        << "contained a '" << injectDUTHierarchyAnnoClass << "', but no '"
        << markDUTAnnoClass << "' was provided";
    return signalPassFailure();
  }

  /// The design-under-test (DUT).  This is kept up-to-date by the pass as the
  /// DUT changes due to internal logic.
  FModuleOp dut = cast<FModuleOp>(instanceInfo.getDut());

  // Create a module that will become the new DUT.  The original DUT is renamed
  // to become the wrapper.  This is done to save copying into the wrapper.
  // While the logical movement is "copy the body of the DUT into a wrapper", it
  // is mechanically more straigthforward to make the DUT the wrappper.  After
  // this block finishes, the "dut" and "wrapper" variables are set correctly.
  // This logic is intentionally put into a block to avoid confusion while the
  // dut and wrapper do not match the logical definition.
  OpBuilder b(circuit.getContext());
  CircuitNamespace circuitNS(circuit);
  b.setInsertionPointAfter(dut);

  // After this, he original DUT module is now the "wrapper".  The new module we
  // just created becomes the "DUT".
  auto oldDutNameAttr = dut.getNameAttr();
  wrapper = dut;
  dut = FModuleOp::create(b, dut.getLoc(), oldDutNameAttr,
                          dut.getConventionAttr(), dut.getPorts());

  // Finish setting up the DUT and the Wrapper.  This depends on if we are in
  // `moveDut` mode or not.  If `moveDut=false` (the normal, legacy behavior),
  // then the wrapper is a wrapper of logic inside the original DUT.  The newly
  // created module to instantiate the wrapper becomes the DUT.  In this mode,
  // we need to move all annotations over to the new DUT.  If `moveDut=true`,
  // then we need to move all the annotations/information from the wrapper onto
  // the DUT.
  //
  // This pass shouldn't create new public modules.  It should only preserve the
  // existing public modules.  In "moveDut" mode, then the wrapper is the new
  // DUT and we should move the publicness from the old DUT to the wrapper.
  // When not in "moveDut" mode, then the wrapper should be made private.
  //
  // Note: `movedDut=true` violates the FIRRTL ABI unless the user it doing
  // something clever with module prefixing.  Because this annotation is already
  // outside the specification, this workflow is allowed even though it violates
  // the FIRRTL ABI.  The mid-term plan is to remove this pass to avoid the tech
  // debt that it creates.
  auto emptyArray = b.getArrayAttr({});
  auto name = circuitNS.newName(wrapperName.getValue());
  if (moveDut) {
    dut.setPortAnnotationsAttr(emptyArray);
    dut.setName(b.getStringAttr(name));
    // If the wrapper is the circuit's main module, then we need to rename the
    // circuit.  This mandates that the wrapper is now public.
    if (circuit.getNameAttr() == wrapper.getNameAttr()) {
      circuit.setNameAttr(dut.getNameAttr());
      dut.setPublic();
    } else {
      dut.setPrivate();
    }
    // The DUT name has changed.  Rewrite instances to use the new DUT name.
    InstanceGraph &instanceGraph = getAnalysis<InstanceGraph>();
    for (auto *use : instanceGraph.lookup(wrapper.getNameAttr())->uses()) {
      auto instanceOp = use->getInstance<InstanceOp>();
      if (!instanceOp) {
        use->getInstance().emitOpError()
            << "instantiates the design-under-test, but "
               "is not a 'firrtl.instance'";
        return signalPassFailure();
      }
      instanceOp.setModuleNameAttr(FlatSymbolRefAttr::get(dut.getNameAttr()));
    }
  } else {
    dut.setVisibility(wrapper.getVisibility());
    dut.setAnnotationsAttr(wrapper.getAnnotationsAttr());
    dut.setPortAnnotationsAttr(wrapper.getPortAnnotationsAttr());
    wrapper.setPrivate();
    wrapper.setAnnotationsAttr(emptyArray);
    wrapper.setPortAnnotationsAttr(emptyArray);
    wrapper.setName(name);
  }

  // Instantiate the wrapper inside the DUT and wire it up.
  b.setInsertionPointToStart(dut.getBodyBlock());
  hw::InnerSymbolNamespace dutNS(dut);
  auto wrapperInst =
      InstanceOp::create(b, b.getUnknownLoc(), wrapper, wrapper.getModuleName(),
                         NameKindEnum::DroppableName, ArrayRef<Attribute>{},
                         ArrayRef<Attribute>{}, false, false,
                         hw::InnerSymAttr::get(b.getStringAttr(
                             dutNS.newName(wrapper.getModuleName()))));
  for (const auto &pair : llvm::enumerate(wrapperInst.getResults())) {
    Value lhs = dut.getArgument(pair.index());
    Value rhs = pair.value();
    if (dut.getPortDirection(pair.index()) == Direction::In)
      std::swap(lhs, rhs);
    emitConnect(b, b.getUnknownLoc(), lhs, rhs);
  }

  // Compute a set of paths that are used _inside_ the wrapper.
  DenseSet<StringAttr> dutPaths, dutPortSyms;
  for (auto anno : AnnotationSet(dut)) {
    auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
    if (sym)
      dutPaths.insert(sym.getAttr());
  }
  for (size_t i = 0, e = dut.getNumPorts(); i != e; ++i) {
    auto portSym = dut.getPortSymbolAttr(i);
    if (portSym)
      dutPortSyms.insert(portSym.getSymName());
    for (auto anno : AnnotationSet::forPort(dut, i)) {
      auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
      if (sym)
        dutPaths.insert(sym.getAttr());
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "DUT Symbol Users:\n";
    for (auto path : dutPaths)
      llvm::dbgs() << "  - " << FlatSymbolRefAttr::get(path) << "\n";
    llvm::dbgs() << "Port Symbols:\n";
    for (auto sym : dutPortSyms)
      llvm::dbgs() << "  - " << FlatSymbolRefAttr::get(sym) << "\n";
  });

  // Update NLAs involving the DUT.
  //
  // In the `moveDut=true` case, the WRAPPER will have the `MarkDUTAnnotation`
  // moved onto it and this will be the "design-under-test" from the perspective
  // of later passes.  In the `moveDut=false` case, the DUT will be the
  // "design-under-test.
  //
  // There are three cases to consider:
  //   1. The DUT or a DUT port is a leaf ref.  Do nothing.
  //   2. The DUT is the root.  Update the root module to be the wrapper.
  //   3. The NLA passes through the DUT.  Remove the original InnerRef and
  //      replace it with two InnerRefs: (1) on the DUT and (2) one the wrapper.
  LLVM_DEBUG(llvm::dbgs() << "Processing hierarchical paths:\n");
  auto &nlaTable = getAnalysis<NLATable>();
  DenseMap<StringAttr, hw::HierPathOp> dutRenames;
  for (auto nla : llvm::make_early_inc_range(nlaTable.lookup(oldDutNameAttr))) {
    LLVM_DEBUG(llvm::dbgs() << "  - " << nla << "\n");
    auto namepath = nla.getNamepath().getValue();

    // The DUT is the root module.  Just update the root module to point at the
    // wrapper.
    //
    // TODO: It _may_ be desirable to only do this in the `moveDut=true` case.
    // In the `moveDut=false` case, this will change the semantic of the
    // annotation if the annotation user is assuming that the annotation root is
    // locked to the DUT.  However, annotations are not supposed to have
    // semantics like this.
    if (nla.root() == oldDutNameAttr) {
      assert(namepath.size() > 1 && "namepath size must be greater than one");
      SmallVector<Attribute> newNamepath{hw::InnerRefAttr::get(
          wrapper.getNameAttr(),
          cast<hw::InnerRefAttr>(namepath.front()).getName())};
      auto tail = namepath.drop_front();
      newNamepath.append(tail.begin(), tail.end());
      nla->setAttr("namepath", b.getArrayAttr(newNamepath));
      continue;
    }

    // The path ends at the DUT.  This may be a reference path (ends in
    // hw::InnerRefAttr) or a module path (ends in FlatSymbolRefAttr).  There
    // are a number of patterns to disambiguate:
    //
    // NOTE: the _DUT_ is the new DUT and all the original DUT contents are put
    // inside the DUT in the _wrapper_.
    //
    //   1. Reference path on DUT port.  Do nothing.
    //   2. Reference path on component.  Add hierarchy
    //   3. Module path on DUT/DUT port.  Clone path, add hier to original path.
    //   4. Module path on component.  Add hierarchy.
    //
    if (nla.leafMod() == oldDutNameAttr) {
      // Case (1): ref path targeting a DUT port.  Do nothing.  When
      // `moveDut=true`, this is always false.
      if (nla.isComponent() && dutPortSyms.count(nla.ref()))
        continue;

      // Case (3): the module path is used by the DUT module or a port. Create a
      // clone of the path and update dutRenames so that this path symbol will
      // get updated for annotations on the DUT or on its ports.
      if (nla.isModule() && dutPaths.contains(nla.getSymNameAttr())) {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPoint(nla);
        auto clone = cast<hw::HierPathOp>(b.clone(*nla));
        clone.setSymNameAttr(b.getStringAttr(
            circuitNS.newName(clone.getSymNameAttr().getValue())));
        dutRenames.insert({nla.getSymNameAttr(), clone});
      }

      // Cases (2), (3), and (4): fallthrough to add hierarchy to original path.
    }

    addHierarchy(nla, dut, wrapperInst, oldDutNameAttr);
  }

  SmallVector<Annotation> newAnnotations;
  auto removeAndUpdateNLAs = [&](Annotation anno) -> bool {
    auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
    if (!sym)
      return false;
    if (!dutRenames.count(sym.getAttr()))
      return false;
    anno.setMember(
        "circt.nonlocal",
        FlatSymbolRefAttr::get(dutRenames[sym.getAttr()].getSymNameAttr()));
    newAnnotations.push_back(anno);
    return true;
  };

  // Replace any annotations on the DUT or DUT ports to use the cloned path.
  AnnotationSet annotations(dut);
  annotations.removeAnnotations(removeAndUpdateNLAs);
  annotations.addAnnotations(newAnnotations);
  annotations.applyToOperation(dut);
  for (size_t i = 0, e = dut.getNumPorts(); i != e; ++i) {
    newAnnotations.clear();
    auto annotations = AnnotationSet::forPort(dut, i);
    annotations.removeAnnotations(removeAndUpdateNLAs);
    annotations.addAnnotations(newAnnotations);
    annotations.applyToPort(dut, i);
  }

  // Update rwprobe operations' local innerrefs within the module.
  wrapper.walk([&](RWProbeOp rwp) {
    rwp.setTargetAttr(hw::InnerRefAttr::get(wrapper.getModuleNameAttr(),
                                            rwp.getTarget().getName()));
  });
}
