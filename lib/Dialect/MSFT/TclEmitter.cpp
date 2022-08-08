//===- TclEmitter.cpp - TCL emitter base implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/TclEmitter.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/MSFT/DeviceDB.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace hw;
using namespace msft;

LogicalResult TclEmitter::populate() {
  if (populated)
    return success();

  // Populated the symbol cache.
  for (auto symOp : topLevel.getOps<mlir::SymbolOpInterface>())
    if (auto name = symOp.getNameAttr())
      topLevelSymbols.addDefinition(name, symOp);
  topLevelSymbols.freeze();
  populated = true;

  // Bin any operations we may need to emit based on the root module in the
  // instance hierarchy path and the potential instance name.

  // Look in InstanceHierarchyOps to get the instance named ones.
  for (auto hier : topLevel.getOps<InstanceHierarchyOp>()) {
    Operation *mod = topLevelSymbols.getDefinition(hier.topModuleRefAttr());
    auto &tclOps = tclOpsForModInstance[mod][hier.instNameAttr()];
    for (auto tclOp : hier.getOps<DynInstDataOpInterface>()) {
      assert(tclOp.getTopModule(topLevelSymbols) == mod &&
             "Referenced mod does does not match");
      tclOps.push_back(tclOp);
    }
  }

  // Locations at the global scope are assumed to refer to the module without an
  // instance.
  for (auto tclOp : topLevel.getOps<DynInstDataOpInterface>()) {
    Operation *mod = tclOp.getTopModule(topLevelSymbols);
    assert(mod && "Must be able to resolve top module");
    tclOpsForModInstance[mod][{}].push_back(tclOp);
  }

  return success();
}

Operation *TclEmitter::getDefinition(FlatSymbolRefAttr sym) {
  if (failed(populate()))
    return nullptr;
  return topLevelSymbols.getDefinition(sym);
}

LogicalResult TclEmitter::emit(Operation *hwMod, StringRef outputFile) {
  if (failed(populate()))
    return failure();

  // Build up the output Tcl, tracking symbol references in state.
  std::string s;
  llvm::raw_string_ostream os(s);
  std::unique_ptr<TclOutputState> state = newOutputState(os);

  // Iterate through all the "instances" for 'hwMod' and produce a tcl proc
  // for each one.
  for (auto tclOpsForInstancesKV : tclOpsForModInstance[hwMod]) {
    StringAttr instName = tclOpsForInstancesKV.first;
    os << "proc {{" << state->symbolRefs.size() << "}}";
    if (instName)
      os << '_' << instName.getValue();
    os << "_config { parent } {\n";
    state->symbolRefs.push_back(SymbolRefAttr::get(hwMod));

    // Loop through the ops relevant to the specified root module "instance".
    LogicalResult ret = success();
    auto &tclOpsForMod = tclOpsForInstancesKV.second;
    for (Operation *tclOp : tclOpsForMod) {
      LogicalResult rc =
          TypeSwitch<Operation *, LogicalResult>(tclOp)
              .Case([&](PDPhysLocationOp op) { return state->emit(op); })
              .Case([&](PDRegPhysLocationOp op) { return state->emit(op); })
              .Case([&](PDPhysRegionOp op) { return state->emit(op); })
              .Case([&](DynamicInstanceVerbatimAttrOp op) {
                return state->emit(op);
              })
              .Default([](Operation *op) {
                return op->emitOpError("could not determine how to output tcl");
              });
      if (failed(rc))
        ret = failure();
    }
    os << "}\n\n";
  }

  // Create a verbatim op containing the Tcl and symbol references.
  OpBuilder builder = OpBuilder::atBlockEnd(hwMod->getBlock());
  auto verbatim = builder.create<sv::VerbatimOp>(
      builder.getUnknownLoc(), os.str(), ValueRange{},
      builder.getArrayAttr(state->symbolRefs));

  // When requested, give the verbatim op an output file.
  if (!outputFile.empty()) {
    auto outputFileAttr =
        OutputFileAttr::getFromFilename(builder.getContext(), outputFile);
    verbatim->setAttr("output_file", outputFileAttr);
  }

  return success();
}

/// Get the GlobalRefOp to which the given operation is pointing. Add it to
/// the set of used global refs.
GlobalRefOp TclOutputState::getRefOp(DynInstDataOpInterface op) {
  auto ref = dyn_cast_or_null<hw::GlobalRefOp>(
      emitter.getDefinition(op.getGlobalRefSym()));
  if (ref)
    emitter.usedRef(ref);
  else
    op.emitOpError("could not find hw.globalRef named ")
        << op.getGlobalRefSym();
  return ref;
}

void TclOutputState::emitInnerRefPart(hw::InnerRefAttr innerRef) {
  // We append new symbolRefs to the state, so s.symbolRefs.size() is the
  // index of the InnerRefAttr we are about to add.
  os << "{{" << symbolRefs.size() << "}}";

  // Append a new inner reference for the template above.
  symbolRefs.push_back(innerRef);
}

void TclOutputState::emitPath(hw::GlobalRefOp ref,
                              Optional<StringRef> subpath) {
  // Traverse each part of the path.
  auto parts = ref.getNamepathAttr().getAsRange<hw::InnerRefAttr>();
  auto lastPart = std::prev(parts.end());
  for (auto part : parts) {
    emitInnerRefPart(part);
    if (part != *lastPart)
      os << '|';
  }

  // Some placements don't require subpaths.
  if (subpath)
    os << subpath;
}

void TclOutputState::emit(PhysLocationAttr attr) {
  assert(false && "unimplemented!");
}

LogicalResult TclOutputState::emit(PDPhysRegionOp region) {
  assert(false && "unimplemented!");
  return failure();
}
LogicalResult TclOutputState::emit(PDPhysLocationOp loc) {
  assert(false && "unimplemented!");
  return failure();
}
LogicalResult TclOutputState::emit(PDRegPhysLocationOp) {
  assert(false && "unimplemented!");
  return failure();
}
LogicalResult TclOutputState::emit(DynamicInstanceVerbatimAttrOp attr) {
  assert(false && "unimplemented!");
  return failure();
}

LogicalResult
TclOutputState::emitLocationAssignment(DynInstDataOpInterface refOp,
                                       PhysLocationAttr,
                                       Optional<StringRef> subpath) {
  assert(false && "unimplemented!");
  return failure();
}

TclOutputState::~TclOutputState() {}
