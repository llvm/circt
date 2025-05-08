//===- LowerLayers.cpp - Lower Layers by Convention -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass lowers FIRRTL layers based on their specified convention.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HierPathCache.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/Utils.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Mutex.h"

#define DEBUG_TYPE "firrtl-lower-layers"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWERLAYERS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {

/// Indicates the kind of reference that was captured.
enum class ConnectKind {
  /// A normal captured value.  This is a read of a value outside the
  /// layerblock.
  NonRef,
  /// A reference.  This is a destination of a ref define.
  Ref
};

struct ConnectInfo {
  Value value;
  ConnectKind kind;
};

/// The delimiters that should be used for a given generated name.  These vary
/// for modules and files, as well as by convention.
enum class Delimiter { BindModule = '_', BindFile = '-', InlineMacro = '$' };

/// This struct contains pre-allocated "safe" names that parallel regions can
/// use to create names in the global namespace.  This is allocated per-layer
/// block.
struct LayerBlockGlobals {

  /// If the layer needs to create a module, use this name.
  StringRef moduleName;

  /// If the layer needs to create a hw::HierPathOp, use this name.
  StringRef hierPathName;
};

} // namespace

// A mapping of an old InnerRefAttr to the new inner symbol and module name that
// need to be spliced into the old InnerRefAttr.  This is used to fix
// hierarchical path operations after layers are converted to modules.
using InnerRefMap =
    DenseMap<hw::InnerRefAttr, std::pair<hw::InnerSymAttr, StringAttr>>;

//===----------------------------------------------------------------------===//
// Naming Helpers
//===----------------------------------------------------------------------===//

static void appendName(StringRef name, SmallString<32> &output,
                       bool toLower = false,
                       Delimiter delimiter = Delimiter::BindFile) {
  if (name.empty())
    return;
  if (!output.empty())
    output.push_back(static_cast<char>(delimiter));
  output.append(name);
  if (!toLower)
    return;
  auto i = output.size() - name.size();
  output[i] = llvm::toLower(output[i]);
}

static void appendName(SymbolRefAttr name, SmallString<32> &output,
                       bool toLower = false,
                       Delimiter delimiter = Delimiter::BindFile) {
  appendName(name.getRootReference(), output, toLower, delimiter);
  for (auto nested : name.getNestedReferences())
    appendName(nested.getValue(), output, toLower, delimiter);
}

/// For a layer `@A::@B::@C` in module Module,
/// the generated module is called `Module_A_B_C`.
static SmallString<32> moduleNameForLayer(StringRef moduleName,
                                          SymbolRefAttr layerName) {
  SmallString<32> result;
  appendName(moduleName, result, /*toLower=*/false,
             /*delimiter=*/Delimiter::BindModule);
  appendName(layerName, result, /*toLower=*/false,
             /*delimiter=*/Delimiter::BindModule);
  return result;
}

static SmallString<32> hierPathNameForLayer(StringRef moduleName,
                                            SymbolRefAttr layerName) {
  SmallString<32> result("__lowerLayers_path");
  appendName(moduleName, result, /*toLower=*/false,
             /*delimiter=*/Delimiter::BindModule);
  appendName(layerName, result, /*toLower=*/false,
             /*delimiter=*/Delimiter::BindModule);
  return result;
}

/// For a layerblock `@A::@B::@C`,
/// the generated instance is called `a_b_c`.
static SmallString<32> instanceNameForLayer(SymbolRefAttr layerName) {
  SmallString<32> result;
  appendName(layerName, result, /*toLower=*/true,
             /*delimiter=*/Delimiter::BindModule);
  return result;
}

/// For all layerblocks `@A::@B::@C` in a circuit called Circuit,
/// the output filename is `layers-Circuit-A-B-C.sv`.
static SmallString<32> fileNameForLayer(StringRef circuitName,
                                        SymbolRefAttr layerName) {
  SmallString<32> result;
  result.append("layers");
  appendName(circuitName, result);
  appendName(layerName, result);
  result.append(".sv");
  return result;
}

/// For a layerblock `@A::@B::@C`, the verilog macro is `A_B_C`.
static SmallString<32>
macroNameForLayer(StringRef circuitName,
                  ArrayRef<FlatSymbolRefAttr> layerName) {
  SmallString<32> result("layer_");
  result.append(circuitName);
  for (auto part : layerName)
    appendName(part, result, /*toLower=*/false,
               /*delimiter=*/Delimiter::InlineMacro);
  return result;
}

//===----------------------------------------------------------------------===//
// LowerLayersPass
//===----------------------------------------------------------------------===//

class LowerLayersPass
    : public circt::firrtl::impl::LowerLayersBase<LowerLayersPass> {
  hw::OutputFileAttr getOutputFile(SymbolRefAttr layerName) {
    auto layer = symbolToLayer.lookup(layerName);
    if (!layer)
      return nullptr;
    return layer->getAttrOfType<hw::OutputFileAttr>("output_file");
  }

  hw::OutputFileAttr outputFileForLayer(StringRef circuitName,
                                        SymbolRefAttr layerName) {
    if (auto file = getOutputFile(layerName))
      return hw::OutputFileAttr::getFromDirectoryAndFilename(
          &getContext(), file.getDirectory(),
          fileNameForLayer(circuitName, layerName),
          /*excludeFromFileList=*/true);
    return hw::OutputFileAttr::getFromFilename(
        &getContext(), fileNameForLayer(circuitName, layerName),
        /*excludeFromFileList=*/true);
  }

  /// Safely build a new module with a given namehint.  This handles geting a
  /// lock to modify the top-level circuit.
  FModuleOp buildNewModule(OpBuilder &builder, LayerBlockOp layerBlock);

  /// Strip layer colors from the module's interface.
  FailureOr<InnerRefMap> runOnModuleLike(FModuleLike moduleLike);

  /// Extract layerblocks and strip probe colors from all ops under the module.
  LogicalResult runOnModuleBody(FModuleOp moduleOp, InnerRefMap &innerRefMap);

  /// Update the module's port types to remove any explicit layer requirements
  /// from any probe types.
  void removeLayersFromPorts(FModuleLike moduleLike);

  /// Update the value's type to remove any layers from any probe types.
  void removeLayersFromValue(Value value);

  /// Lower an inline layerblock to an ifdef block.
  void lowerInlineLayerBlock(LayerOp layer, LayerBlockOp layerBlock);

  /// Preprocess layers to build macro declarations and cache information about
  /// the layers so that this can be quired later.
  void preprocessLayers(CircuitNamespace &ns, OpBuilder &b, LayerOp layer,
                        StringRef circuitName,
                        SmallVector<FlatSymbolRefAttr> &stack);
  void preprocessLayers(CircuitNamespace &ns, LayerOp layer,
                        StringRef circuitName);

  /// Entry point for the function.
  void runOnOperation() override;

  /// Indicates exclusive access to modify the circuitNamespace and the circuit.
  llvm::sys::SmartMutex<true> *circuitMutex;

  /// A map of layer blocks to "safe" global names which are fine to create in
  /// the circuit namespace.
  DenseMap<LayerBlockOp, LayerBlockGlobals> layerBlockGlobals;

  /// A map from inline layers to their macro names.
  DenseMap<LayerOp, FlatSymbolRefAttr> macroNames;

  /// A mapping of symbol name to layer operation.
  DenseMap<SymbolRefAttr, LayerOp> symbolToLayer;

  /// Utility for creating hw::HierPathOp.
  hw::HierPathCache *hierPathCache;
};

/// Multi-process safe function to build a module in the circuit and return it.
/// The name provided is only a namehint for the module---a unique name will be
/// generated if there are conflicts with the namehint in the circuit-level
/// namespace.
FModuleOp LowerLayersPass::buildNewModule(OpBuilder &builder,
                                          LayerBlockOp layerBlock) {
  auto location = layerBlock.getLoc();
  auto namehint = layerBlockGlobals.lookup(layerBlock).moduleName;
  llvm::sys::SmartScopedLock<true> instrumentationLock(*circuitMutex);
  FModuleOp newModule = builder.create<FModuleOp>(
      location, builder.getStringAttr(namehint),
      ConventionAttr::get(builder.getContext(), Convention::Internal),
      ArrayRef<PortInfo>{}, ArrayAttr{});
  if (auto dir = getOutputFile(layerBlock.getLayerNameAttr())) {
    assert(dir.isDirectory());
    newModule->setAttr("output_file", dir);
  }
  SymbolTable::setSymbolVisibility(newModule, SymbolTable::Visibility::Private);
  return newModule;
}

void LowerLayersPass::removeLayersFromValue(Value value) {
  auto type = dyn_cast<RefType>(value.getType());
  if (!type || !type.getLayer())
    return;
  value.setType(type.removeLayer());
}

void LowerLayersPass::removeLayersFromPorts(FModuleLike moduleLike) {
  auto oldTypeAttrs = moduleLike.getPortTypesAttr();
  SmallVector<Attribute> newTypeAttrs;
  newTypeAttrs.reserve(oldTypeAttrs.size());
  bool changed = false;

  for (auto typeAttr : oldTypeAttrs.getAsRange<TypeAttr>()) {
    if (auto refType = dyn_cast<RefType>(typeAttr.getValue())) {
      if (refType.getLayer()) {
        typeAttr = TypeAttr::get(refType.removeLayer());
        changed = true;
      }
    }
    newTypeAttrs.push_back(typeAttr);
  }

  if (!changed)
    return;

  moduleLike.setPortTypesAttr(
      ArrayAttr::get(moduleLike.getContext(), newTypeAttrs));

  if (auto moduleOp = dyn_cast<FModuleOp>(moduleLike.getOperation())) {
    for (auto arg : moduleOp.getBodyBlock()->getArguments())
      removeLayersFromValue(arg);
  }
}

FailureOr<InnerRefMap>
LowerLayersPass::runOnModuleLike(FModuleLike moduleLike) {
  LLVM_DEBUG({
    llvm::dbgs() << "Module: " << moduleLike.getModuleName() << "\n";
    llvm::dbgs() << "  Examining Layer Blocks:\n";
  });

  // Strip away layers from the interface of the module-like op.
  InnerRefMap innerRefMap;
  auto result =
      TypeSwitch<Operation *, LogicalResult>(moduleLike.getOperation())
          .Case<FModuleOp>([&](auto op) {
            op.setLayers({});
            removeLayersFromPorts(op);
            return runOnModuleBody(op, innerRefMap);
          })
          .Case<FExtModuleOp, FIntModuleOp, FMemModuleOp>([&](auto op) {
            op.setLayers({});
            removeLayersFromPorts(op);
            return success();
          })
          .Case<ClassOp, ExtClassOp>([](auto) { return success(); })
          .Default(
              [](auto *op) { return op->emitError("unknown module-like op"); });

  if (failed(result))
    return failure();

  return innerRefMap;
}

void LowerLayersPass::lowerInlineLayerBlock(LayerOp layer,
                                            LayerBlockOp layerBlock) {
  OpBuilder builder(layerBlock);
  auto macroName = macroNames[layer];
  auto ifDef = builder.create<sv::IfDefOp>(layerBlock.getLoc(), macroName);
  ifDef.getBodyRegion().takeBody(layerBlock.getBodyRegion());
  layerBlock.erase();
}

LogicalResult LowerLayersPass::runOnModuleBody(FModuleOp moduleOp,
                                               InnerRefMap &innerRefMap) {
  CircuitOp circuitOp = moduleOp->getParentOfType<CircuitOp>();
  StringRef circuitName = circuitOp.getName();
  hw::InnerSymbolNamespace ns(moduleOp);

  // A cache of values to nameable ops that can be used
  DenseMap<Value, Operation *> nameableCache;

  // Get or create a node op for a value captured by a layer block.
  auto getOrCreateNodeOp = [&](Value operand,
                               ImplicitLocOpBuilder &builder) -> Operation * {
    // Use the cache hit.
    auto it = nameableCache.find(operand);
    if (it != nameableCache.end())
      return it->getSecond();

    // Create a new node.  Put it in the cache and use it.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfterValue(operand);
    SmallString<16> nameHint;
    // Try to generate a "good" name hint to use for the node.
    if (auto *definingOp = operand.getDefiningOp()) {
      if (auto instanceOp = dyn_cast<InstanceOp>(definingOp)) {
        nameHint.append(instanceOp.getName());
        nameHint.push_back('_');
        nameHint.append(
            instanceOp.getPortName(cast<OpResult>(operand).getResultNumber()));
      } else if (auto opName = definingOp->getAttrOfType<StringAttr>("name")) {
        nameHint.append(opName);
      }
    }
    return nameableCache
        .insert({operand, builder.create<NodeOp>(operand.getLoc(), operand,
                                                 nameHint.empty()
                                                     ? "_layer_probe"
                                                     : StringRef(nameHint))})
        .first->getSecond();
  };

  // Determine the replacement for an operand within the current region.  Keep a
  // densemap of replacements around to avoid creating the same hardware
  // multiple times.
  DenseMap<Value, Value> replacements;
  auto getReplacement = [&](Operation *user, Value value) -> Value {
    auto it = replacements.find(value);
    if (it != replacements.end())
      return it->getSecond();

    ImplicitLocOpBuilder localBuilder(value.getLoc(), &getContext());
    Value replacement;

    auto layerBlockOp = user->getParentOfType<LayerBlockOp>();
    localBuilder.setInsertionPointToStart(layerBlockOp.getBody());

    // If the operand is "special", e.g., it has no XMR representation, then we
    // need to clone it.
    //
    // TODO: Change this to recursively clone.  This will matter once FString
    // operations have operands.
    if (type_isa<FStringType>(value.getType())) {
      localBuilder.setInsertionPoint(user);
      replacement = localBuilder.clone(*value.getDefiningOp())->getResult(0);
      replacements.insert({value, replacement});
      return replacement;
    }

    // If the operand is an XMR ref, then we _have_ to clone it.
    auto *definingOp = value.getDefiningOp();
    if (isa_and_present<XMRRefOp>(definingOp)) {
      replacement = localBuilder.clone(*definingOp)->getResult(0);
      replacements.insert({value, replacement});
      return replacement;
    }

    // Determine the replacement value for the captured operand.  There are
    // three cases that can occur:
    //
    // 1. Capturing something zero-width.  Create a zero-width constant zero.
    // 2. Capture an expression or instance port.  Drop a node and XMR deref
    //    that.
    // 3. Capture something that can handle an inner sym.  XMR deref that.
    //
    // Note: (3) can be either an operation or a _module_ port.
    auto baseType = type_cast<FIRRTLBaseType>(value.getType());
    if (baseType && baseType.getBitWidthOrSentinel() == 0) {
      OpBuilder::InsertionGuard guard(localBuilder);
      auto zeroUIntType = UIntType::get(localBuilder.getContext(), 0);
      replacement = localBuilder.createOrFold<BitCastOp>(
          value.getType(), localBuilder.create<ConstantOp>(
                               zeroUIntType, getIntZerosAttr(zeroUIntType)));
    } else {
      auto *definingOp = value.getDefiningOp();
      hw::InnerRefAttr innerRef;
      if (definingOp) {
        // Always create a node.  This is a trade-off between optimizations and
        // dead code.  By adding the node, this allows the original path to be
        // better optimized, but will leave dead code in the design.  If the
        // node is not created, then the output is less optimized.  Err on the
        // side of dead code.  This dead node _may_ be eventually inlined by
        // `ExportVerilog`.  However, this is not guaranteed.
        definingOp = getOrCreateNodeOp(value, localBuilder);
        innerRef = getInnerRefTo(
            definingOp, [&](auto) -> hw::InnerSymbolNamespace & { return ns; });
      } else {
        auto portIdx = cast<BlockArgument>(value).getArgNumber();
        innerRef = getInnerRefTo(
            cast<FModuleLike>(*moduleOp), portIdx,
            [&](auto) -> hw::InnerSymbolNamespace & { return ns; });
      }

      hw::HierPathOp hierPathOp;
      {
        // TODO: Move to before parallel region to avoid the lock.
        auto insertPoint = OpBuilder::InsertPoint(moduleOp->getBlock(),
                                                  Block::iterator(moduleOp));
        llvm::sys::SmartScopedLock<true> circuitLock(*circuitMutex);
        hierPathOp = hierPathCache->getOrCreatePath(
            localBuilder.getArrayAttr({innerRef}), localBuilder.getLoc(),
            insertPoint, layerBlockGlobals.lookup(layerBlockOp).hierPathName);
        hierPathOp.setVisibility(SymbolTable::Visibility::Private);
      }

      replacement = localBuilder.create<XMRDerefOp>(
          value.getType(), hierPathOp.getSymNameAttr());
    }

    replacements.insert({value, replacement});

    return replacement;
  };

  // A map of instance ops to modules that this pass creates.  This is used to
  // check if this was an instance that we created and to do fast module
  // dereferencing (avoiding a symbol table).
  DenseMap<InstanceOp, FModuleOp> createdInstances;

  // Check that the preconditions for this pass are met.  Reject any ops which
  // must have been removed before this runs.
  auto opPreconditionCheck = [](Operation *op) -> LogicalResult {
    // LowerXMR op removal postconditions.
    if (isa<RefCastOp, RefDefineOp, RefResolveOp, RefSendOp, RefSubOp,
            RWProbeOp>(op))
      return op->emitOpError()
             << "cannot be handled by the lower-layers pass.  This should have "
                "already been removed by the lower-xmr pass.";

    return success();
  };

  // Post-order traversal that expands a layer block into its parent. Because of
  // the pass precondition that this runs _after_ `LowerXMR`, not much has to
  // happen here.  All of the following do happen, though:
  //
  // 1. Any layer coloring is stripped.
  // 2. Layers with Inline convention are converted to SV ifdefs.
  // 3. Layers with Bind convention are converted to new modules and then
  //    instantiated at their original location.  Any captured values are either
  //    moved, cloned, or converted to XMR deref ops.
  // 4. Move instances created from earlier (3) conversions out of later (3)
  //    conversions.  This is necessary to avoid a SystemVerilog-illegal
  //    bind-under-bind.  (See Section 23.11 of 1800-2023.)
  // 5. Keep track of special ops (ops with inner symbols or verbatims) which
  //    need to have something updated because of the new instance hierarchy
  //    being created.
  //
  // Remember, this is post-order, in-order.  Child layer blocks are visited
  // before parents.  Any nested regions _within_ the layer block are also
  // visited before the outer layer block.
  auto result = moduleOp.walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
    if (failed(opPreconditionCheck(op)))
      return WalkResult::interrupt();

    // Strip layer requirements from any op that might represent a probe.
    if (auto wire = dyn_cast<WireOp>(op)) {
      removeLayersFromValue(wire.getResult());
      return WalkResult::advance();
    }
    if (auto instance = dyn_cast<InstanceOp>(op)) {
      instance.setLayers({});
      for (auto result : instance.getResults())
        removeLayersFromValue(result);
      return WalkResult::advance();
    }

    auto layerBlock = dyn_cast<LayerBlockOp>(op);
    if (!layerBlock)
      return WalkResult::advance();

    // After this point, we are dealing with a layer block.
    auto layer = symbolToLayer.lookup(layerBlock.getLayerName());

    if (layer.getConvention() == LayerConvention::Inline) {
      lowerInlineLayerBlock(layer, layerBlock);
      return WalkResult::advance();
    }

    // After this point, we are dealing with a bind convention layer block.
    assert(layer.getConvention() == LayerConvention::Bind);

    // Clear the replacements so that none are re-used across layer blocks.
    replacements.clear();
    OpBuilder builder(moduleOp);
    SmallVector<hw::InnerSymAttr> innerSyms;
    SmallVector<sv::VerbatimOp> verbatims;
    DenseSet<Operation *> spilledSubOps;
    auto layerBlockWalkResult = layerBlock.walk([&](Operation *op) {
      // Error if pass preconditions are not met.
      if (failed(opPreconditionCheck(op)))
        return WalkResult::interrupt();

      // Specialized handling of subfields, subindexes, and subaccesses which
      // need to be spilled and nodes that referred to spilled nodes.  If these
      // are kept in the module, then the XMR is going to be bidirectional.  Fix
      // this for subfield and subindex by moving these ops outside the
      // layerblock.  Try to fix this for subaccess and error if the move can't
      // be made because the index is defined inside the layerblock.  (This case
      // is exceedingly rare given that subaccesses are almost always unexepcted
      // when this pass runs.)  Additionally, if any nodes are seen that are
      // transparently referencing a spilled op, spill the node, too.  The node
      // provides an anchor for an inner symbol (which subfield, subindex, and
      // subaccess do not).
      if (isa<SubfieldOp, SubindexOp>(op)) {
        auto input = op->getOperand(0);
        if (!firrtl::type_cast<FIRRTLBaseType>(input.getType()).isPassive() &&
            !isAncestorOfValueOwner(layerBlock, input)) {
          op->moveBefore(layerBlock);
          spilledSubOps.insert(op);
        }
        return WalkResult::advance();
      }
      if (auto subOp = dyn_cast<SubaccessOp>(op)) {
        auto input = subOp.getInput();
        if (firrtl::type_cast<FIRRTLBaseType>(input.getType()).isPassive())
          return WalkResult::advance();

        if (!isAncestorOfValueOwner(layerBlock, input) &&
            !isAncestorOfValueOwner(layerBlock, subOp.getIndex())) {
          subOp->moveBefore(layerBlock);
          spilledSubOps.insert(op);
          return WalkResult::advance();
        }
        auto diag = op->emitOpError()
                    << "has a non-passive operand and captures a value defined "
                       "outside its enclosing bind-convention layerblock.  The "
                       "'LowerLayers' pass cannot lower this as it would "
                       "create an output port on the resulting module.";
        diag.attachNote(layerBlock.getLoc())
            << "the layerblock is defined here";
        return WalkResult::interrupt();
      }
      if (auto nodeOp = dyn_cast<NodeOp>(op)) {
        auto *definingOp = nodeOp.getInput().getDefiningOp();
        if (definingOp &&
            spilledSubOps.contains(nodeOp.getInput().getDefiningOp())) {
          op->moveBefore(layerBlock);
          return WalkResult::advance();
        }
      }

      // Record any operations inside the layer block which have inner symbols.
      // Theses may have symbol users which need to be updated.
      //
      // Note: this needs to _not_ index spilled NodeOps above.
      if (auto symOp = dyn_cast<hw::InnerSymbolOpInterface>(op))
        if (auto innerSym = symOp.getInnerSymAttr())
          innerSyms.push_back(innerSym);

      // Handle instance ops that were created from nested layer blocks.  These
      // ops need to be moved outside the layer block to avoid nested binds.
      // Nested binds are illegal in the SystemVerilog specification (and
      // checked by FIRRTL verification).
      //
      // For each value defined in this layer block which drives a port of one
      // of these instances, create an output reference type port on the
      // to-be-created module and drive it with the value.  Move the instance
      // outside the layer block.  We will hook it up later once we replace the
      // layer block with an instance.
      if (auto instOp = dyn_cast<InstanceOp>(op)) {
        // Ignore instances which this pass did not create.
        if (!createdInstances.contains(instOp))
          return WalkResult::advance();

        LLVM_DEBUG({
          llvm::dbgs()
              << "      Found instance created from nested layer block:\n"
              << "        module: " << instOp.getModuleName() << "\n"
              << "        instance: " << instOp.getName() << "\n";
        });
        instOp->moveBefore(layerBlock);
        return WalkResult::advance();
      }

      // Handle captures.  For any captured operands, convert them to a suitable
      // replacement value.  The `getReplacement` function will automatically
      // reuse values whenever possible.
      for (size_t i = 0, e = op->getNumOperands(); i != e; ++i) {
        auto operand = op->getOperand(i);

        // If the operand is in this layer block, do nothing.
        //
        // Note: This check is what avoids handling ConnectOp destinations.
        if (isAncestorOfValueOwner(layerBlock, operand))
          continue;

        op->setOperand(i, getReplacement(op, operand));
      }

      if (auto verbatim = dyn_cast<sv::VerbatimOp>(op))
        verbatims.push_back(verbatim);

      return WalkResult::advance();
    });

    if (layerBlockWalkResult.wasInterrupted())
      return WalkResult::interrupt();

    // Create the new module.  This grabs a lock to modify the circuit.
    FModuleOp newModule = buildNewModule(builder, layerBlock);
    SymbolTable::setSymbolVisibility(newModule,
                                     SymbolTable::Visibility::Private);
    newModule.getBody().takeBody(layerBlock.getRegion());

    LLVM_DEBUG({
      llvm::dbgs() << "      New Module: "
                   << layerBlockGlobals.lookup(layerBlock).moduleName << "\n";
    });

    // Replace the original layer block with an instance.  Hook up the
    // instance. Intentionally create instance with probe ports which do
    // not have an associated layer.  This is illegal IR that will be
    // made legal by the end of the pass.  This is done to avoid having
    // to revisit and rewrite each instance everytime it is moved into a
    // parent layer.
    builder.setInsertionPointAfter(layerBlock);
    auto instanceName = instanceNameForLayer(layerBlock.getLayerName());
    auto instanceOp = builder.create<InstanceOp>(
        layerBlock.getLoc(), /*moduleName=*/newModule,
        /*name=*/
        instanceName, NameKindEnum::DroppableName,
        /*annotations=*/ArrayRef<Attribute>{},
        /*portAnnotations=*/ArrayRef<Attribute>{}, /*lowerToBind=*/true,
        /*doNotPrint=*/false,
        /*innerSym=*/
        (innerSyms.empty() ? hw::InnerSymAttr{}
                           : hw::InnerSymAttr::get(builder.getStringAttr(
                                 ns.newName(instanceName)))));

    auto outputFile =
        outputFileForLayer(circuitName, layerBlock.getLayerName());
    instanceOp->setAttr("output_file", outputFile);

    createdInstances.try_emplace(instanceOp, newModule);

    LLVM_DEBUG(llvm::dbgs() << "        moved inner refs:\n");
    for (hw::InnerSymAttr innerSym : innerSyms) {
      auto oldInnerRef = hw::InnerRefAttr::get(moduleOp.getModuleNameAttr(),
                                               innerSym.getSymName());
      auto splice = std::make_pair(instanceOp.getInnerSymAttr(),
                                   newModule.getModuleNameAttr());
      innerRefMap.insert({oldInnerRef, splice});
      LLVM_DEBUG(llvm::dbgs() << "          - ref: " << oldInnerRef << "\n"
                              << "            splice: " << splice.first << ", "
                              << splice.second << "\n";);
    }

    // Update verbatims that target operations extracted alongside.
    if (!verbatims.empty()) {
      mlir::AttrTypeReplacer replacer;
      replacer.addReplacement(
          [&innerRefMap](hw::InnerRefAttr ref) -> std::optional<Attribute> {
            auto it = innerRefMap.find(ref);
            if (it != innerRefMap.end())
              return hw::InnerRefAttr::get(it->second.second, ref.getName());
            return std::nullopt;
          });
      for (auto verbatim : verbatims)
        replacer.replaceElementsIn(verbatim);
    }

    layerBlock.erase();

    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

void LowerLayersPass::preprocessLayers(CircuitNamespace &ns, OpBuilder &b,
                                       LayerOp layer, StringRef circuitName,
                                       SmallVector<FlatSymbolRefAttr> &stack) {
  stack.emplace_back(FlatSymbolRefAttr::get(layer.getSymNameAttr()));
  ArrayRef stackRef(stack);
  symbolToLayer.insert(
      {SymbolRefAttr::get(stackRef.front().getAttr(), stackRef.drop_front()),
       layer});
  if (layer.getConvention() == LayerConvention::Inline) {
    auto *ctx = &getContext();
    auto macName = macroNameForLayer(circuitName, stack);
    auto symName = ns.newName(macName);

    auto symNameAttr = StringAttr::get(ctx, symName);
    auto macNameAttr = StringAttr();
    if (macName != symName)
      macNameAttr = StringAttr::get(ctx, macName);

    b.create<sv::MacroDeclOp>(layer->getLoc(), symNameAttr, ArrayAttr(),
                              macNameAttr);
    macroNames[layer] = FlatSymbolRefAttr::get(&getContext(), symNameAttr);
  }
  for (auto child : layer.getOps<LayerOp>())
    preprocessLayers(ns, b, child, circuitName, stack);
  stack.pop_back();
}

void LowerLayersPass::preprocessLayers(CircuitNamespace &ns, LayerOp layer,
                                       StringRef circuitName) {
  OpBuilder b(layer);
  SmallVector<FlatSymbolRefAttr> stack;
  preprocessLayers(ns, b, layer, circuitName, stack);
}

/// Process a circuit to remove all layer blocks in each module and top-level
/// layer definition.
void LowerLayersPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "==----- Running LowerLayers "
                      "-------------------------------------------------===\n");
  CircuitOp circuitOp = getOperation();
  StringRef circuitName = circuitOp.getName();

  // Initialize members which cannot be initialized automatically.
  llvm::sys::SmartMutex<true> mutex;
  circuitMutex = &mutex;

  CircuitNamespace ns(circuitOp);
  hw::HierPathCache hpc(
      &ns, OpBuilder::InsertPoint(getOperation().getBodyBlock(),
                                  getOperation().getBodyBlock()->begin()));
  hierPathCache = &hpc;

  for (auto &op : *circuitOp.getBodyBlock()) {
    // Determine names for all modules that will be created.  Do this serially
    // to avoid non-determinism from creating these in the parallel region.
    if (auto moduleOp = dyn_cast<FModuleOp>(op)) {
      moduleOp->walk([&](LayerBlockOp layerBlockOp) {
        auto moduleName = moduleNameForLayer(moduleOp.getModuleName(),
                                             layerBlockOp.getLayerName());
        auto hierPathName = hierPathNameForLayer(moduleOp.getModuleName(),
                                                 layerBlockOp.getLayerName());
        layerBlockGlobals.insert(
            {layerBlockOp, {ns.newName(moduleName), ns.newName(hierPathName)}});
      });
      continue;
    }
    // Build verilog macro declarations for each inline layer declarations.
    // Cache layer symbol refs for lookup later.
    if (auto layerOp = dyn_cast<LayerOp>(op)) {
      preprocessLayers(ns, layerOp, circuitName);
      continue;
    }
  }

  auto mergeMaps = [](auto &&a, auto &&b) {
    if (failed(a))
      return std::forward<decltype(a)>(a);
    if (failed(b))
      return std::forward<decltype(b)>(b);

    for (auto bb : *b)
      a->insert(bb);
    return std::forward<decltype(a)>(a);
  };

  // Lower the layer blocks of each module.
  SmallVector<FModuleLike> modules(
      circuitOp.getBodyBlock()->getOps<FModuleLike>());
  auto failureOrInnerRefMap = transformReduce(
      circuitOp.getContext(), modules, FailureOr<InnerRefMap>(InnerRefMap{}),
      mergeMaps, [this](FModuleLike mod) -> FailureOr<InnerRefMap> {
        return runOnModuleLike(mod);
      });
  if (failed(failureOrInnerRefMap))
    return signalPassFailure();
  auto &innerRefMap = *failureOrInnerRefMap;

  // Rewrite any hw::HierPathOps which have namepaths that contain rewritting
  // inner refs.
  //
  // TODO: This unnecessarily computes a new namepath for every hw::HierPathOp
  // even if that namepath is not used.  It would be better to only build the
  // new namepath when a change is needed, e.g., by recording updates to the
  // namepath.
  for (hw::HierPathOp hierPathOp : circuitOp.getOps<hw::HierPathOp>()) {
    SmallVector<Attribute> newNamepath;
    bool modified = false;
    for (auto attr : hierPathOp.getNamepath()) {
      hw::InnerRefAttr innerRef = dyn_cast<hw::InnerRefAttr>(attr);
      if (!innerRef) {
        newNamepath.push_back(attr);
        continue;
      }
      auto it = innerRefMap.find(innerRef);
      if (it == innerRefMap.end()) {
        newNamepath.push_back(attr);
        continue;
      }

      auto &[inst, mod] = it->getSecond();
      newNamepath.push_back(
          hw::InnerRefAttr::get(innerRef.getModule(), inst.getSymName()));
      newNamepath.push_back(hw::InnerRefAttr::get(mod, innerRef.getName()));
      modified = true;
    }
    if (modified)
      hierPathOp.setNamepathAttr(
          ArrayAttr::get(circuitOp.getContext(), newNamepath));
  }

  // Generate the header and footer of each bindings file.  The body will be
  // populated later when binds are exported to Verilog.  This produces text
  // like:
  //
  //     `include "layers-A.sv"
  //     `include "layers-A-B.sv"
  //     `ifndef layers_A_B_C
  //     `define layers_A_B_C
  //     <body>
  //     `endif // layers_A_B_C
  //
  // TODO: This would be better handled without the use of verbatim ops.
  OpBuilder builder(circuitOp);
  SmallVector<std::pair<LayerOp, StringAttr>> layers;
  circuitOp.walk<mlir::WalkOrder::PreOrder>([&](LayerOp layerOp) {
    auto parentOp = layerOp->getParentOfType<LayerOp>();
    while (!layers.empty() && parentOp != layers.back().first)
      layers.pop_back();

    if (layerOp.getConvention() == LayerConvention::Inline) {
      layers.emplace_back(layerOp, nullptr);
      return;
    }

    builder.setInsertionPointToStart(circuitOp.getBodyBlock());

    // Save the "layers-<circuit>-<group>" and "layers_<circuit>_<group>" string
    // as this is reused a bunch.
    SmallString<32> prefixFile("layers-"), prefixMacro("layers_");
    prefixFile.append(circuitName);
    prefixFile.append("-");
    prefixMacro.append(circuitName);
    prefixMacro.append("_");
    for (auto [layer, _] : layers) {
      prefixFile.append(layer.getSymName());
      prefixFile.append("-");
      prefixMacro.append(layer.getSymName());
      prefixMacro.append("_");
    }
    prefixFile.append(layerOp.getSymName());
    prefixMacro.append(layerOp.getSymName());

    SmallString<128> includes;
    for (auto [_, strAttr] : layers) {
      if (!strAttr)
        continue;
      includes.append(strAttr);
      includes.append("\n");
    }

    hw::OutputFileAttr bindFile;
    if (auto outputFile =
            layerOp->getAttrOfType<hw::OutputFileAttr>("output_file")) {
      auto dir = outputFile.getDirectory();
      bindFile = hw::OutputFileAttr::getFromDirectoryAndFilename(
          &getContext(), dir, prefixFile + ".sv",
          /*excludeFromFileList=*/true);
    } else {
      bindFile = hw::OutputFileAttr::getFromFilename(
          &getContext(), prefixFile + ".sv", /*excludeFromFileList=*/true);
    }

    // Write header to a verbatim.
    auto header = builder.create<sv::VerbatimOp>(
        layerOp.getLoc(),
        includes + "`ifndef " + prefixMacro + "\n" + "`define " + prefixMacro);
    header->setAttr("output_file", bindFile);

    // Write footer to a verbatim.
    builder.setInsertionPointToEnd(circuitOp.getBodyBlock());
    auto footer = builder.create<sv::VerbatimOp>(layerOp.getLoc(),
                                                 "`endif // " + prefixMacro);
    footer->setAttr("output_file", bindFile);

    if (!layerOp.getBody().getOps<LayerOp>().empty())
      layers.emplace_back(
          layerOp,
          builder.getStringAttr("`include \"" +
                                bindFile.getFilename().getValue() + "\""));
  });

  // All layers definitions can now be deleted.
  for (auto layerOp :
       llvm::make_early_inc_range(circuitOp.getBodyBlock()->getOps<LayerOp>()))
    layerOp.erase();

  // Cleanup state.
  hierPathCache = nullptr;
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerLayersPass() {
  return std::make_unique<LowerLayersPass>();
}
