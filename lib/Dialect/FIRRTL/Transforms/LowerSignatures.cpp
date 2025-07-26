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

#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Support/Debug.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-lower-signatures"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWERSIGNATURES
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

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
    sInternalPaths = StringAttr::get(context, "internalPaths");
  }
  AttrCache(const AttrCache &) = default;

  StringAttr nameAttr, sPortDirections, sPortNames, sPortTypes, sPortLocations,
      sPortAnnotations, sInternalPaths;
};

struct FieldMapEntry : public PortInfo {
  size_t portID;
  size_t resultID;
  size_t fieldID;
};

using PortConversion = SmallVector<FieldMapEntry>;

template <typename T>
class FieldIDSearch {
  using E = typename T::ElementType;
  using V = SmallVector<E>;

public:
  using const_iterator = typename V::const_iterator;

  template <typename Container>
  FieldIDSearch(const Container &src) {
    if constexpr (std::is_convertible_v<Container, Attribute>)
      if (!src)
        return;
    for (auto attr : src)
      vals.push_back(attr);
    std::sort(vals.begin(), vals.end(), fieldComp);
  }

  std::pair<const_iterator, const_iterator> find(uint64_t low,
                                                 uint64_t high) const {
    return {std::lower_bound(vals.begin(), vals.end(), low, fieldCompInt2),
            std::upper_bound(vals.begin(), vals.end(), high, fieldCompInt1)};
  }

  bool empty(uint64_t low, uint64_t high) const {
    auto [b, e] = find(low, high);
    return b == e;
  }

private:
  static constexpr auto fieldComp = [](const E &lhs, const E &rhs) {
    return lhs.getFieldID() < rhs.getFieldID();
  };
  static constexpr auto fieldCompInt2 = [](const E &lhs, uint64_t rhs) {
    return lhs.getFieldID() < rhs;
  };
  static constexpr auto fieldCompInt1 = [](uint64_t lhs, const E &rhs) {
    return lhs < rhs.getFieldID();
  };

  V vals;
};

} // namespace

static hw::InnerSymAttr
symbolsForFieldIDRange(MLIRContext *ctx,
                       const FieldIDSearch<hw::InnerSymAttr> &syms,
                       uint64_t low, uint64_t high) {
  auto [b, e] = syms.find(low, high);
  SmallVector<hw::InnerSymPropertiesAttr, 4> newSyms(b, e);
  if (newSyms.empty())
    return {};
  for (auto &sym : newSyms)
    sym = hw::InnerSymPropertiesAttr::get(
        ctx, sym.getName(), sym.getFieldID() - low, sym.getSymVisibility());
  return hw::InnerSymAttr::get(ctx, newSyms);
}

static AnnotationSet
annosForFieldIDRange(MLIRContext *ctx,
                     const FieldIDSearch<AnnotationSet> &annos, uint64_t low,
                     uint64_t high) {
  AnnotationSet newAnnos(ctx);
  auto [b, e] = annos.find(low, high);
  for (; b != e; ++b)
    newAnnos.addAnnotations(Annotation(*b, b->getFieldID() - low));
  return newAnnos;
}

static LogicalResult
computeLoweringImpl(FModuleLike mod, PortConversion &newPorts, Convention conv,
                    size_t portID, const PortInfo &port, bool isFlip,
                    Twine name, FIRRTLType type, uint64_t fieldID,
                    const FieldIDSearch<hw::InnerSymAttr> &syms,
                    const FieldIDSearch<AnnotationSet> &annos) {
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
                symbolsForFieldIDRange(ctx, syms, fieldID, lastId), port.loc,
                annosForFieldIDRange(ctx, annos, fieldID, lastId)},
               portID,
               newPorts.size(),
               fieldID});
        } else {
          for (auto [idx, elem] : llvm::enumerate(bundle.getElements())) {
            if (failed(computeLoweringImpl(
                    mod, newPorts, conv, portID, port, isFlip ^ elem.isFlip,
                    name + "_" + elem.name.getValue(), elem.type,
                    fieldID + bundle.getFieldID(idx), syms, annos)))
              return failure();
            if (!syms.empty(fieldID, fieldID))
              return mod.emitError("Port [")
                     << port.name
                     << "] should be subdivided, but cannot be because of "
                        "symbol ["
                     << port.sym.getSymIfExists(fieldID) << "] on a bundle";
            if (!annos.empty(fieldID, fieldID)) {
              auto err = mod.emitError("Port [")
                         << port.name
                         << "] should be subdivided, but cannot be because of "
                            "annotations [";
              auto [b, e] = annos.find(fieldID, fieldID);
              err << b->getClass() << "(" << b->getFieldID() << ")";
              b++;
              for (; b != e; ++b)
                err << ", " << b->getClass() << "(" << b->getFieldID() << ")";
              err << "] on a bundle";
              return err;
            }
          }
        }
        return success();
      })
      .Case<FVectorType>([&](FVectorType vector) -> LogicalResult {
        if (conv != Convention::Scalarized &&
            vector.getElementType().isPassive()) {
          auto lastId = fieldID + vector.getMaxFieldID();
          newPorts.push_back(
              {{StringAttr::get(ctx, name), type,
                isFlip ? Direction::Out : Direction::In,
                symbolsForFieldIDRange(ctx, syms, fieldID, lastId), port.loc,
                annosForFieldIDRange(ctx, annos, fieldID, lastId)},
               portID,
               newPorts.size(),
               fieldID});
        } else {
          for (size_t i = 0, e = vector.getNumElements(); i < e; ++i) {
            if (failed(computeLoweringImpl(
                    mod, newPorts, conv, portID, port, isFlip,
                    name + "_" + Twine(i), vector.getElementType(),
                    fieldID + vector.getFieldID(i), syms, annos)))
              return failure();
            if (!syms.empty(fieldID, fieldID))
              return mod.emitError("Port [")
                     << port.name
                     << "] should be subdivided, but cannot be because of "
                        "symbol ["
                     << port.sym.getSymIfExists(fieldID) << "] on a vector";
            if (!annos.empty(fieldID, fieldID)) {
              auto err = mod.emitError("Port [")
                         << port.name
                         << "] should be subdivided, but cannot be because of "
                            "annotations [";
              auto [b, e] = annos.find(fieldID, fieldID);
              err << b->getClass();
              ++b;
              for (; b != e; ++b)
                err << ", " << b->getClass();
              err << "] on a vector";
              return err;
            }
          }
        }
        return success();
      })
      .Default([&](FIRRTLType type) {
        // Properties and other types wind up here.
        newPorts.push_back(
            {{StringAttr::get(ctx, name), type,
              isFlip ? Direction::Out : Direction::In,
              symbolsForFieldIDRange(ctx, syms, fieldID, fieldID), port.loc,
              annosForFieldIDRange(ctx, annos, fieldID, fieldID)},
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
  for (auto [idx, port] : llvm::enumerate(mod.getPorts())) {
    if (failed(computeLoweringImpl(
            mod, newPorts, conv, idx, port, port.direction == Direction::Out,
            port.name.getValue(), type_cast<FIRRTLType>(port.type), 0,
            FieldIDSearch<hw::InnerSymAttr>(port.sym),
            FieldIDSearch<AnnotationSet>(port.annotations))))
      return failure();
  }
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
          wire = WireOp::create(theBuilder, module.getPortType(p.portID),
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
        bounceWires[idx] = WireOp::create(theBuilder, module.getPortType(idx),
                                          module.getPortNameAttr(idx))
                               .getResult();
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
        attr.getName() != "portSymbols" && attr.getName() != "portLocations" &&
        attr.getName() != "internalPaths")
      newModuleAttrs.push_back(attr);

  SmallVector<Direction> newPortDirections;
  SmallVector<Attribute> newPortNames;
  SmallVector<Attribute> newPortTypes;
  SmallVector<Attribute> newPortSyms;
  SmallVector<Attribute> newPortLocations;
  SmallVector<Attribute, 8> newPortAnnotations;
  SmallVector<Attribute> newInternalPaths;

  bool hasInternalPaths = false;
  auto internalPaths = module->getAttrOfType<ArrayAttr>("internalPaths");
  for (auto p : newPorts) {
    newPortTypes.push_back(TypeAttr::get(p.type));
    newPortNames.push_back(p.name);
    newPortDirections.push_back(p.direction);
    newPortSyms.push_back(p.sym);
    newPortLocations.push_back(p.loc);
    newPortAnnotations.push_back(p.annotations.getArrayAttr());
    if (internalPaths) {
      auto internalPath = cast<InternalPathAttr>(internalPaths[p.portID]);
      newInternalPaths.push_back(internalPath);
      if (internalPath.getPath())
        hasInternalPaths = true;
    }
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

  assert(newInternalPaths.empty() ||
         newInternalPaths.size() == newPorts.size());
  if (hasInternalPaths) {
    newModuleAttrs.emplace_back(cache.sInternalPaths,
                                theBuilder.getArrayAttr(newInternalPaths));
  }

  // Update the module's attributes.
  module->setAttrs(newModuleAttrs);
  FModuleLike::fixupPortSymsArray(newPortSyms, theBuilder.getContext());
  module.setPortSymbols(newPortSyms);
  return success();
}

static void lowerModuleBody(FModuleOp mod,
                            const DenseMap<StringAttr, PortConversion> &ports) {
  mod->walk([&](InstanceOp inst) -> void {
    ImplicitLocOpBuilder theBuilder(inst.getLoc(), inst);
    const auto &modPorts = ports.at(inst.getModuleNameAttr().getAttr());

    // Fix up the Instance
    SmallVector<PortInfo> instPorts; // Oh I wish ArrayRef was polymorphic.
    for (auto p : modPorts) {
      p.sym = {};
      // Might need to partially copy stuff from the old instance.
      p.annotations = AnnotationSet{mod.getContext()};
      instPorts.push_back(p);
    }
    auto annos = inst.getAnnotations();
    auto newOp = InstanceOp::create(
        theBuilder, instPorts, inst.getModuleName(), inst.getName(),
        inst.getNameKind(), annos.getValue(), inst.getLayers(),
        inst.getLowerToBind(), inst.getDoNotPrint(), inst.getInnerSymAttr());

    auto oldDict = inst->getDiscardableAttrDictionary();
    auto newDict = newOp->getDiscardableAttrDictionary();
    auto oldNames = inst.getPortNamesAttr();
    SmallVector<NamedAttribute> newAttrs;
    for (auto na : oldDict)
      if (!newDict.contains(na.getName()))
        newOp->setDiscardableAttr(na.getName(), na.getValue());

    // Connect up the old instance users to the new instance
    SmallVector<WireOp> bounce(inst.getNumResults());
    for (auto p : modPorts) {
      // No change?  No bounce wire.
      if (p.fieldID == 0) {
        inst.getResult(p.portID).replaceAllUsesWith(
            newOp.getResult(p.resultID));
        continue;
      }
      if (!bounce[p.portID]) {
        bounce[p.portID] = WireOp::create(
            theBuilder, inst.getResult(p.portID).getType(),
            theBuilder.getStringAttr(
                inst.getName() + "." +
                cast<StringAttr>(oldNames[p.portID]).getValue()));
        inst.getResult(p.portID).replaceAllUsesWith(
            bounce[p.portID].getResult());
      }
      // Connect up the Instance to the bounce wires
      if (p.isInput())
        emitConnect(theBuilder, newOp.getResult(p.resultID),
                    getValueByFieldID(theBuilder, bounce[p.portID].getResult(),
                                      p.fieldID));
      else
        emitConnect(theBuilder,
                    getValueByFieldID(theBuilder, bounce[p.portID].getResult(),
                                      p.fieldID),
                    newOp.getResult(p.resultID));
    }
    // Zero Width ports may have dangling connects since they are not preserved
    // and do not have bounce wires.
    for (auto *use : llvm::make_early_inc_range(inst->getUsers())) {
      assert(isa<MatchingConnectOp>(use) || isa<ConnectOp>(use));
      use->erase();
    }
    inst->erase();
    return;
  });
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerSignaturesPass
    : public circt::firrtl::impl::LowerSignaturesBase<LowerSignaturesPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// This is the main entrypoint for the lowering pass.
void LowerSignaturesPass::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this) << "\n");
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
