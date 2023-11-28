//===- LowerSignatures.cpp - Lower Module Signatures ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerSignatures pass.  This pass replaces aggregate
// types with expanded values in module arguments as specified by the ABI
// information.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Parallel.h"

#define DEBUG_TYPE "firrtl-lower-signatures"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Module Type Lowering
//===----------------------------------------------------------------------===//
namespace {

struct AttrCache {
  AttrCache(MLIRContext *context) {
    nameAttr = StringAttr::get(context, "name");
    sPortDirections = StringAttr::get(context, "portDirections");
    sPortNames = StringAttr::get(context, "portNames");
    sPortTypes = StringAttr::get(context, "portTypes");
    sPortLocations = StringAttr::get(context, "portLocations");
    sPortAnnotations = StringAttr::get(context, "portAnnotations");
  }
  AttrCache(const AttrCache &) = default;

  StringAttr nameAttr, sPortDirections, sPortNames, sPortTypes, sPortLocations,
      sPortAnnotations;
};

struct FieldMapEntry : public PortInfo {
  size_t portID;
  size_t resultID;
  size_t fieldID;
};

using PortConversion = SmallVector<FieldMapEntry>;
} // namespace

static hw::InnerSymAttr symbolsForFieldIDRange(hw::InnerSymAttr syms,
                                               uint64_t low, uint64_t high) {
  // No symbols, early exit
  if (!syms || syms.empty())
    return {};

  SmallVector<hw::InnerSymPropertiesAttr, 4> newSyms;
  for (auto sym : syms) {
    auto sfield = sym.getFieldID();
    if (sfield >= low && sfield <= high) {
      newSyms.push_back(hw::InnerSymPropertiesAttr::get(
          syms.getContext(), sym.getName(), sfield - low,
          sym.getSymVisibility()));
    }
  }

  if (newSyms.empty())
    return {};

  return hw::InnerSymAttr::get(syms.getContext(), newSyms);
}

static AnnotationSet annosForFieldIDRange(const AnnotationSet &annos,
                                          uint64_t low, uint64_t high) {
  AnnotationSet newAnnos(annos.getContext());
  for (auto anno : annos) {
    auto afield = anno.getFieldID();
    if (afield >= low && afield <= high) {
      newAnnos.addAnnotations(anno);
    }
  }
  return newAnnos;
}

// TODO: check for dead/unattachable symbols and put them on the wire if
// possible
static LogicalResult
computeLoweringImpl(FModuleLike mod, PortConversion &newPorts, Convention conv,
                    size_t portID, const PortInfo &port, bool isFlip,
                    Twine name, FIRRTLType type, uint64_t fieldID) {
  auto *ctx = type.getContext();
  return FIRRTLTypeSwitch<FIRRTLType, LogicalResult>(type)
      .Case<BundleType>([&](BundleType bundle) -> LogicalResult {
        // This should be enhanced to be able to handle bundle<all flips of
        // passive>, or this should be a canonicalizer
        if (conv != Convention::Scalarized && bundle.isPassive()) {
          auto lastId = fieldID + bundle.getMaxFieldID();
          newPorts.push_back(
              {{StringAttr::get(ctx, name), type,
                isFlip ? Direction::Out : Direction::In,
                symbolsForFieldIDRange(port.sym, fieldID, lastId), port.loc,
                annosForFieldIDRange(port.annotations, fieldID, lastId)},
               portID,
               newPorts.size(),
               fieldID});
        } else {
          for (auto [idx, elem] : llvm::enumerate(bundle.getElements())) {
            if (failed(computeLoweringImpl(
                    mod, newPorts, conv, portID, port, isFlip ^ elem.isFlip,
                    name + "_" + elem.name.getValue(), elem.type,
                    fieldID + bundle.getFieldID(idx))))
              return failure();
            if (port.sym && port.sym.getSymIfExists(fieldID))
              return mod.emitError("Port [")
                     << port.name
                     << "] should be subdivided, but cannot be because of "
                        "symbol ["
                     << port.sym.getSymIfExists(fieldID) << "] on a bundle";
            if (!annosForFieldIDRange(port.annotations, fieldID, fieldID)
                     .empty())
              return mod.emitError("Port [")
                     << port.name
                     << "] should be subdivided, but cannot be because of "
                        "annotations ["
                     << annosForFieldIDRange(port.annotations, fieldID, fieldID)
                            .getArrayAttr()
                     << "] on a bundle";
          }
        }
        return success();
      })
      .template Case<FVectorType>([&](FVectorType vector) -> LogicalResult {
        if (conv != Convention::Scalarized &&
            vector.getElementType().isPassive()) {
          auto lastId = fieldID + vector.getMaxFieldID();
          newPorts.push_back(
              {{StringAttr::get(ctx, name), type,
                isFlip ? Direction::Out : Direction::In,
                symbolsForFieldIDRange(port.sym, fieldID, lastId), port.loc,
                annosForFieldIDRange(port.annotations, fieldID, lastId)},
               portID,
               newPorts.size(),
               fieldID});
        } else {
          for (size_t i = 0, e = vector.getNumElements(); i < e; ++i) {
            if (failed(computeLoweringImpl(mod, newPorts, conv, portID, port,
                                           isFlip, name + "_" + Twine(i),
                                           vector.getElementType(),
                                           fieldID + vector.getFieldID(i))))
              return failure();
            if (port.sym && port.sym.getSymIfExists(fieldID))
              return mod.emitError("Port [")
                     << port.name
                     << "] should be subdivided, but cannot be because of "
                        "symbol ["
                     << port.sym.getSymIfExists(fieldID) << "] on a vector";
            if (!annosForFieldIDRange(port.annotations, fieldID, fieldID)
                     .empty())
              return mod.emitError("Port [")
                     << port.name
                     << "] should be subdivided, but cannot be because of "
                        "annotations ["
                     << annosForFieldIDRange(port.annotations, fieldID, fieldID)
                            .getArrayAttr()
                     << "] on a vector";
          }
        }
        return success();
      })
      .template Case<FEnumType>([&](FEnumType fenum) { return failure(); })
      .template Case<RefType>([&](RefType ref) {
        newPorts.push_back({{StringAttr::get(ctx, name),
                             ref,
                             isFlip ? Direction::Out : Direction::In,
                             {},
                             port.loc,
                             {}},
                            portID,
                            newPorts.size(),
                            fieldID});
        return success();
      })
      .template Case<FIRRTLBaseType>([&](FIRRTLBaseType groundType) {
        assert(groundType.isGround() && "only ground types are expected here");
        newPorts.push_back(
            {{StringAttr::get(ctx, name), groundType,
              isFlip ? Direction::Out : Direction::In,
              symbolsForFieldIDRange(port.sym, fieldID, fieldID), port.loc,
              annosForFieldIDRange(port.annotations, fieldID, fieldID)},
             portID,
             newPorts.size(),
             fieldID});
        return success();
      });
}

// compute a new moduletype from an old module type and lowering convention.
// Also compute a fieldID map from port, fieldID -> port
static LogicalResult computeLowering(FModuleLike mod, Convention conv,
                                     PortConversion &newPorts) {
  for (auto [idx, port] : llvm::enumerate(mod.getPorts()))
    if (computeLoweringImpl(
            mod, newPorts, conv, idx, port, port.direction == Direction::Out,
            port.name.getValue(), type_cast<FIRRTLType>(port.type), 0)
            .failed())
      return failure();
  return success();
}

static LogicalResult lowerModuleSignature(FModuleLike module, Convention conv,
                                          AttrCache &cache,
                                          PortConversion &newPorts) {
  ImplicitLocOpBuilder theBuilder(module.getLoc(), module.getContext());
  if (computeLowering(module, conv, newPorts).failed())
    return failure();
  if (auto mod = dyn_cast<FModuleOp>(module.getOperation())) {
    Block *body = mod.getBodyBlock();
    theBuilder.setInsertionPointToStart(body);
    auto oldNumArgs = body->getNumArguments();

    // Compute the replacement value for old arguments
    // This creates all the new arguments and produces bounce wires when
    // necessary
    SmallVector<Value> bounceWires(oldNumArgs);
    for (auto &p : newPorts) {
      auto newArg = body->addArgument(p.type, p.loc);
      // Get or create a bounce wire for changed ports
      // For unmodified ports, move the uses to the replacement port
      if (p.fieldID != 0) {
        auto &wire = bounceWires[p.portID];
        if (!wire)
          wire = theBuilder
                     .create<WireOp>(module.getPortType(p.portID),
                                     module.getPortNameAttr(p.portID),
                                     NameKindEnum::InterestingName)
                     .getResult();
      } else {
        bounceWires[p.portID] = newArg;
      }
    }
    // replace old arguments.  Somethings get dropped completely, like
    // zero-length vectors.
    for (auto idx = 0U; idx < oldNumArgs; ++idx) {
      if (!bounceWires[idx]) {
        bounceWires[idx] =
            theBuilder.create<WireOp>(module.getPortType(idx)).getResult();
      }
      body->getArgument(idx).replaceAllUsesWith(bounceWires[idx]);
    }

    // Goodbye old ports, now ResultID in the PortInfo is correct.
    body->eraseArguments(0, oldNumArgs);

    // Connect the bounce wires to the new arguments
    for (auto &p : newPorts) {
      if (isa<BlockArgument>(bounceWires[p.portID]))
        continue;
      if (p.isOutput())
        emitConnect(
            theBuilder, body->getArgument(p.resultID),
            getValueByFieldID(theBuilder, bounceWires[p.portID], p.fieldID));
      else
        emitConnect(
            theBuilder,
            getValueByFieldID(theBuilder, bounceWires[p.portID], p.fieldID),
            body->getArgument(p.resultID));
    }
  }

  SmallVector<NamedAttribute, 8> newModuleAttrs;

  // Copy over any attributes that weren't original argument attributes.
  for (auto attr : module->getAttrDictionary())
    // Drop old "portNames", directions, and argument attributes.  These are
    // handled differently below.
    if (attr.getName() != "portNames" && attr.getName() != "portDirections" &&
        attr.getName() != "portTypes" && attr.getName() != "portAnnotations" &&
        attr.getName() != "portSyms" && attr.getName() != "portLocations")
      newModuleAttrs.push_back(attr);

  SmallVector<Direction> newPortDirections;
  SmallVector<Attribute> newPortNames;
  SmallVector<Attribute> newPortTypes;
  SmallVector<Attribute> newPortSyms;
  SmallVector<Attribute> newPortLocations;
  SmallVector<Attribute, 8> newPortAnnotations;
  for (auto p : newPorts) {
    newPortTypes.push_back(TypeAttr::get(p.type));
    newPortNames.push_back(p.name);
    newPortDirections.push_back(p.direction);
    newPortSyms.push_back(p.sym);
    newPortLocations.push_back(p.loc);
    newPortAnnotations.push_back(p.annotations.getArrayAttr());
  }
  newModuleAttrs.push_back(NamedAttribute(
      cache.sPortDirections,
      direction::packAttribute(module.getContext(), newPortDirections)));

  newModuleAttrs.push_back(
      NamedAttribute(cache.sPortNames, theBuilder.getArrayAttr(newPortNames)));

  newModuleAttrs.push_back(
      NamedAttribute(cache.sPortTypes, theBuilder.getArrayAttr(newPortTypes)));

  newModuleAttrs.push_back(NamedAttribute(
      cache.sPortLocations, theBuilder.getArrayAttr(newPortLocations)));

  newModuleAttrs.push_back(NamedAttribute(
      cache.sPortAnnotations, theBuilder.getArrayAttr(newPortAnnotations)));

  // Update the module's attributes.
  module->setAttrs(newModuleAttrs);
  module.setPortSymbols(newPortSyms);
  return success();
}

static void lowerModuleBody(FModuleOp mod,
                            const DenseMap<StringAttr, PortConversion> &ports) {
  ImplicitLocOpBuilder theBuilder(mod.getLoc(), mod.getContext());
  mod->walk([&](InstanceOp inst) -> void {
    theBuilder.setInsertionPoint(inst);
    const auto &modPorts = ports.at(inst.getModuleNameAttr().getAttr());

    SmallVector<Value> bounceWires;
    // Create bounce wires for old signals
    for (auto r : inst.getResults()) {
      auto wire = theBuilder.create<WireOp>(r.getType());
      bounceWires.push_back(wire.getResult());
      r.replaceAllUsesWith(wire.getResult());
    }
    // Fix up the Instance
    SmallVector<PortInfo> instPorts; // Oh I wish ArrayRef was polymorphic.
    for (auto p : modPorts)
      instPorts.push_back(p);
    auto annos = inst.getAnnotations();
    annos.dump();
    auto newOp = theBuilder.create<InstanceOp>(
        instPorts, inst.getModuleName(), inst.getName(), inst.getNameKind(),
        annos.getValue(), inst.getLowerToBind(), inst.getInnerSymAttr());

    auto oldDict = inst->getDiscardableAttrDictionary();
    auto newDict = newOp->getDiscardableAttrDictionary();
    SmallVector<NamedAttribute> newAttrs;
    for (auto na : oldDict)
      if (!newDict.contains(na.getName()))
        newOp->setDiscardableAttr(na.getName(), na.getValue());

    // Connect up the Instance to the bounce wires
    for (auto [idx, p] : llvm::enumerate(modPorts)) {
      if (p.isInput())
        emitConnect(
            theBuilder, newOp.getResult(p.resultID),
            getValueByFieldID(theBuilder, bounceWires[p.portID], p.fieldID));
      else
        emitConnect(
            theBuilder,
            getValueByFieldID(theBuilder, bounceWires[p.portID], p.fieldID),
            newOp.getResult(p.resultID));
    }

    inst.erase();

    return;
  });
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerSignaturesPass : public LowerSignaturesBase<LowerSignaturesPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// This is the main entrypoint for the lowering pass.
void LowerSignaturesPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "===- Running Lower Signature Pass "
                      "------------------------------------------------===\n");
  // Cached attr
  AttrCache cache(&getContext());

  DenseMap<StringAttr, PortConversion> portMap;
  auto circuit = getOperation();

  for (auto mod : circuit.getOps<FModuleLike>()) {
    if (lowerModuleSignature(mod, mod.getConvention(), cache,
                             portMap[mod.getNameAttr()])
            .failed())
      return signalPassFailure();
  }
  parallelForEach(&getContext(), circuit.getOps<FModuleOp>(),
                  [&portMap](FModuleOp mod) { lowerModuleBody(mod, portMap); });
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerSignaturesPass() {
  return std::make_unique<LowerSignaturesPass>();
}
