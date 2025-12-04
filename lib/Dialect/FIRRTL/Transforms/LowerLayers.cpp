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

#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HierPathCache.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/Utils.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
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

static void appendName(const ArrayRef<FlatSymbolRefAttr> &names,
                       SmallString<32> &output, bool toLower = false,
                       Delimiter delimiter = Delimiter::BindFile) {
  for (auto name : names)
    appendName(name.getValue(), output, toLower, delimiter);
}

static void appendName(SymbolRefAttr name, SmallString<32> &output,
                       bool toLower = false,
                       Delimiter delimiter = Delimiter::BindFile) {
  appendName(name.getRootReference(), output, toLower, delimiter);
  appendName(name.getNestedReferences(), output, toLower, delimiter);
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

static SmallString<32> fileNameForLayer(StringRef moduleName, StringAttr root,
                                        ArrayRef<FlatSymbolRefAttr> nested) {
  SmallString<32> result;
  result.append("layers");
  appendName(moduleName, result);
  appendName(root, result);
  appendName(nested, result);
  result.append(".sv");
  return result;
}

/// For all layerblocks `@A::@B::@C` in a module called Module,
/// the output filename is `layers-Module-A-B-C.sv`.
static SmallString<32> fileNameForLayer(StringRef moduleName,
                                        SymbolRefAttr layerName) {
  return fileNameForLayer(moduleName, layerName.getRootReference(),
                          layerName.getNestedReferences());
}

/// For all layerblocks `@A::@B::@C` in a module called Module,
/// the include-guard macro is `layers_Module_A_B_C`.
static SmallString<32> guardMacroNameForLayer(StringRef moduleName,
                                              SymbolRefAttr layerName) {
  SmallString<32> result;
  result.append("layers");
  appendName(moduleName, result, false, Delimiter::BindModule);
  appendName(layerName, result, false, Delimiter::BindModule);
  return result;
}

/// For a layerblock `@A::@B::@C`, the verilog macro is `A_B_C`.
static SmallString<32>
macroNameForLayer(StringRef circuitName,
                  ArrayRef<FlatSymbolRefAttr> layerName) {
  SmallString<32> result("layer");
  for (auto part : layerName)
    appendName(part, result, /*toLower=*/false,
               /*delimiter=*/Delimiter::InlineMacro);
  return result;
}

//===----------------------------------------------------------------------===//
// LowerLayersPass
//===----------------------------------------------------------------------===//

namespace {
/// Information about each bind file we are emitting. During a prepass, we walk
/// the modules to find layerblocks, creating an emit::FileOp for each bound
/// layer used under each module. As we do this, we build a table of these info
/// objects for quick lookup later.
struct BindFileInfo {
  /// The filename of the bind file _without_ a directory.
  StringAttr filename;
  /// Where to insert bind statements into the bind file.
  Block *body;
};
} // namespace

class LowerLayersPass
    : public circt::firrtl::impl::LowerLayersBase<LowerLayersPass> {
  using Base::Base;

  hw::OutputFileAttr getOutputFile(SymbolRefAttr layerName) {
    auto layer = symbolToLayer.lookup(layerName);
    if (!layer)
      return nullptr;
    return layer->getAttrOfType<hw::OutputFileAttr>("output_file");
  }

  hw::OutputFileAttr outputFileForLayer(StringRef moduleName,
                                        SymbolRefAttr layerName) {
    if (auto file = getOutputFile(layerName))
      return hw::OutputFileAttr::getFromDirectoryAndFilename(
          &getContext(), file.getDirectory(),
          fileNameForLayer(moduleName, layerName),
          /*excludeFromFileList=*/true);
    return hw::OutputFileAttr::getFromFilename(
        &getContext(), fileNameForLayer(moduleName, layerName),
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

  /// Build macro declarations and cache information about the layers.
  void preprocessLayers(CircuitNamespace &ns, OpBuilder &b, LayerOp layer,
                        StringRef circuitName,
                        SmallVector<FlatSymbolRefAttr> &stack);
  void preprocessLayers(CircuitNamespace &ns);

  /// For each module, build a bindfile for each bound-layer, if needed.
  void preprocessModules(CircuitNamespace &ns, InstanceGraph &ig);

  /// Build the bindfile skeletons for each module. Set up a table which tells
  /// us for each module/layer pair, where to insert the bind operations.
  void preprocessModuleLike(CircuitNamespace &ns, InstanceGraphNode *node);

  /// Build the bindfile skeleton for a module.
  void preprocessModule(CircuitNamespace &ns, InstanceGraphNode *node,
                        FModuleOp module);

  /// Record the supposed bindfiles for any known layers of the ext module.
  void preprocessExtModule(CircuitNamespace &ns, InstanceGraphNode *node,
                           FExtModuleOp extModule);

  /// Build a bindfile skeleton for a particular module and layer.
  void buildBindFile(CircuitNamespace &ns, InstanceGraphNode *node,
                     OpBuilder &b, SymbolRefAttr layerName, LayerOp layer);

  /// Entry point for the function.
  void runOnOperation() override;

  /// Indicates exclusive access to modify the circuitNamespace and the circuit.
  llvm::sys::SmartMutex<true> *circuitMutex;

  /// A map of layer blocks to "safe" global names which are fine to create in
  /// the circuit namespace.
  DenseMap<LayerBlockOp, LayerBlockGlobals> layerBlockGlobals;

  /// A map from inline layers to their macro names.
  DenseMap<LayerOp, FlatSymbolRefAttr> macroNames;

  /// A mapping of symbol name to layer operation. This also serves as an
  /// iterable list of all layers declared in a circuit. We use a map vector so
  /// that the iteration order matches the order of declaration in the circuit.
  /// This order is not required for correctness, it helps with legibility.
  llvm::MapVector<SymbolRefAttr, LayerOp> symbolToLayer;

  /// Utility for creating hw::HierPathOp.
  hw::HierPathCache *hierPathCache;

  /// A mapping from module*layer to bindfile name.
  DenseMap<Operation *, DenseMap<LayerOp, BindFileInfo>> bindFiles;
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
  FModuleOp newModule = FModuleOp::create(
      builder, location, builder.getStringAttr(namehint),
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
          .Case<FExtModuleOp>([&](auto op) {
            op.setKnownLayers({});
            op.setLayers({});
            removeLayersFromPorts(op);
            return success();
          })
          .Case<FIntModuleOp, FMemModuleOp>([&](auto op) {
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
  if (!layerBlock.getBody()->empty()) {
    OpBuilder builder(layerBlock);
    auto macroName = macroNames[layer];
    auto ifDef = sv::IfDefOp::create(builder, layerBlock.getLoc(), macroName);
    ifDef.getBodyRegion().takeBody(layerBlock.getBodyRegion());
  }
  layerBlock.erase();
}

LogicalResult LowerLayersPass::runOnModuleBody(FModuleOp moduleOp,
                                               InnerRefMap &innerRefMap) {
  hw::InnerSymbolNamespace ns(moduleOp);

  // A cache of values to nameable ops that can be used
  DenseMap<Value, Operation *> nodeCache;

  // Get or create a node op for a value captured by a layer block.
  auto getOrCreateNodeOp = [&](Value operand,
                               ImplicitLocOpBuilder &builder) -> Operation * {
    // Use the cache hit.
    auto *nodeOp = nodeCache.lookup(operand);
    if (nodeOp)
      return nodeOp;

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
    return nodeOp = NodeOp::create(builder, operand.getLoc(), operand,
                                   nameHint.empty() ? "_layer_probe"
                                                    : StringRef(nameHint));
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
          value.getType(), ConstantOp::create(localBuilder, zeroUIntType,
                                              getIntZerosAttr(zeroUIntType)));
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

      replacement = XMRDerefOp::create(localBuilder, value.getType(),
                                       hierPathOp.getSymNameAttr());
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
    for (auto result : op->getResults())
      removeLayersFromValue(result);

    // If the op is an instance, clear the enablelayers attribute.
    if (auto instance = dyn_cast<InstanceOp>(op))
      instance.setLayers({});

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
      auto fixSubOp = [&](auto subOp) {
        auto input = subOp.getInput();

        // If the input is defined in this layerblock, we are done.
        if (isAncestorOfValueOwner(layerBlock, input))
          return WalkResult::advance();

        // Otherwise, capture the input operand, if possible.
        if (firrtl::type_cast<FIRRTLBaseType>(input.getType()).isPassive()) {
          subOp.getInputMutable().assign(getReplacement(subOp, input));
          return WalkResult::advance();
        }

        // Otherwise, move the subfield op out of the layerblock.
        op->moveBefore(layerBlock);
        spilledSubOps.insert(op);
        return WalkResult::advance();
      };

      if (auto subOp = dyn_cast<SubfieldOp>(op))
        return fixSubOp(subOp);

      if (auto subOp = dyn_cast<SubindexOp>(op))
        return fixSubOp(subOp);

      if (auto subOp = dyn_cast<SubaccessOp>(op)) {
        auto input = subOp.getInput();
        auto index = subOp.getIndex();

        // If the input is defined in this layerblock, capture the index if
        // needed, and we are done.
        if (isAncestorOfValueOwner(layerBlock, input)) {
          if (!isAncestorOfValueOwner(layerBlock, index)) {
            subOp.getIndexMutable().assign(getReplacement(subOp, index));
          }
          return WalkResult::advance();
        }

        // Otherwise, capture the input operand, if possible.
        if (firrtl::type_cast<FIRRTLBaseType>(input.getType()).isPassive()) {
          subOp.getInputMutable().assign(getReplacement(subOp, input));
          if (!isAncestorOfValueOwner(layerBlock, index))
            subOp.getIndexMutable().assign(getReplacement(subOp, index));
          return WalkResult::advance();
        }

        // Otherwise, move the subaccess op out of the layerblock, if possible.
        if (!isAncestorOfValueOwner(layerBlock, index)) {
          subOp->moveBefore(layerBlock);
          spilledSubOps.insert(op);
          return WalkResult::advance();
        }

        // When the input is not passive, but the index is defined inside this
        // layerblock, we are out of options.
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
    auto innerSym =
        hw::InnerSymAttr::get(builder.getStringAttr(ns.newName(instanceName)));

    auto instanceOp = InstanceOp::create(
        builder, layerBlock.getLoc(), /*moduleName=*/newModule,
        /*name=*/
        instanceName, NameKindEnum::DroppableName,
        /*annotations=*/ArrayRef<Attribute>{},
        /*portAnnotations=*/ArrayRef<Attribute>{}, /*lowerToBind=*/false,
        /*doNotPrint=*/true, innerSym);

    auto outputFile = outputFileForLayer(moduleOp.getModuleNameAttr(),
                                         layerBlock.getLayerName());
    instanceOp->setAttr("output_file", outputFile);

    createdInstances.try_emplace(instanceOp, newModule);

    // create the bind op.
    {
      auto builder = OpBuilder::atBlockEnd(bindFiles[moduleOp][layer].body);
      BindOp::create(builder, layerBlock.getLoc(), moduleOp.getModuleNameAttr(),
                     instanceOp.getInnerSymAttr().getSymName());
    }

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

    sv::MacroDeclOp::create(b, layer->getLoc(), symNameAttr, ArrayAttr(),
                            macNameAttr);
    macroNames[layer] = FlatSymbolRefAttr::get(&getContext(), symNameAttr);
  }
  for (auto child : layer.getOps<LayerOp>())
    preprocessLayers(ns, b, child, circuitName, stack);
  stack.pop_back();
}

void LowerLayersPass::preprocessLayers(CircuitNamespace &ns) {
  auto circuit = getOperation();
  auto circuitName = circuit.getName();
  for (auto layer : circuit.getOps<LayerOp>()) {
    OpBuilder b(layer);
    SmallVector<FlatSymbolRefAttr> stack;
    preprocessLayers(ns, b, layer, circuitName, stack);
  }
}

void LowerLayersPass::buildBindFile(CircuitNamespace &ns,
                                    InstanceGraphNode *node, OpBuilder &b,
                                    SymbolRefAttr layerName, LayerOp layer) {
  assert(layer.getConvention() == LayerConvention::Bind);
  auto module = node->getModule<FModuleOp>();
  auto loc = module.getLoc();

  // Compute the include guard macro name.
  auto macroName = guardMacroNameForLayer(module.getModuleName(), layerName);
  auto macroSymbol = ns.newName(macroName);
  auto macroNameAttr = StringAttr::get(&getContext(), macroName);
  auto macroSymbolAttr = StringAttr::get(&getContext(), macroSymbol);
  auto macroSymbolRefAttr = FlatSymbolRefAttr::get(macroSymbolAttr);

  // Compute the base name for the bind file.
  auto bindFileName = fileNameForLayer(module.getName(), layerName);

  // Build the full output path using the filename of the bindfile and the
  // output directory of the layer, if any.
  auto dir = layer->getAttrOfType<hw::OutputFileAttr>("output_file");
  StringAttr filename = StringAttr::get(&getContext(), bindFileName);
  StringAttr path;
  if (dir)
    path = StringAttr::get(&getContext(),
                           Twine(dir.getDirectory()) + bindFileName);
  else
    path = filename;

  // Declare the macro for the include guard.
  sv::MacroDeclOp::create(b, loc, macroSymbolAttr, ArrayAttr{}, macroNameAttr);

  // Create the emit op.
  auto bindFile = emit::FileOp::create(b, loc, path);
  OpBuilder::InsertionGuard _(b);
  b.setInsertionPointToEnd(bindFile.getBody());

  // Create the #ifndef for the include guard.
  auto includeGuard = sv::IfDefOp::create(b, loc, macroSymbolRefAttr);
  b.createBlock(&includeGuard.getElseRegion());

  // Create the #define for the include guard.
  sv::MacroDefOp::create(b, loc, macroSymbolRefAttr);

  // Create IR to enable any parent layers.
  auto parent = layer->getParentOfType<LayerOp>();
  while (parent) {
    // If the parent is bound-in, we enable it by including the bindfile.
    // The parent bindfile will enable all ancestors.
    if (parent.getConvention() == LayerConvention::Bind) {
      auto target = bindFiles[module][parent].filename;
      sv::IncludeOp::create(b, loc, IncludeStyle::Local, target);
      break;
    }

    // If the parent layer is inline, we can only assert that the parent is
    // already enabled.
    if (parent.getConvention() == LayerConvention::Inline) {
      auto parentMacroSymbolRefAttr = macroNames[parent];
      auto parentGuard = sv::IfDefOp::create(b, loc, parentMacroSymbolRefAttr);
      OpBuilder::InsertionGuard guard(b);
      b.createBlock(&parentGuard.getElseRegion());
      auto message = StringAttr::get(&getContext(),
                                     Twine(parent.getName()) + " not enabled");
      sv::MacroErrorOp::create(b, loc, message);
      parent = parent->getParentOfType<LayerOp>();
      continue;
    }

    // Unknown Layer convention.
    llvm_unreachable("unknown layer convention");
  }

  // Create IR to include bind files for child modules. If a module is
  // instantiated more than once, we only need to include the bindfile once.
  SmallPtrSet<Operation *, 8> seen;
  for (auto *record : *node) {
    auto *child = record->getTarget()->getModule().getOperation();
    if (!std::get<bool>(seen.insert(child)))
      continue;
    auto files = bindFiles[child];
    auto lookup = files.find(layer);
    if (lookup != files.end())
      sv::IncludeOp::create(b, loc, IncludeStyle::Local,
                            lookup->second.filename);
  }

  // Save the bind file information for later.
  auto &info = bindFiles[module][layer];
  info.filename = filename;
  info.body = includeGuard.getElseBlock();
}

void LowerLayersPass::preprocessModule(CircuitNamespace &ns,
                                       InstanceGraphNode *node,
                                       FModuleOp module) {

  OpBuilder b(&getContext());
  b.setInsertionPointAfter(module);

  // Create a bind file only if the layer is used under the module.
  llvm::SmallDenseSet<LayerOp> layersRequiringBindFiles;

  // If the module is public, create a bind file for all layers.
  if (module.isPublic() || emitAllBindFiles)
    for (auto [_, layer] : symbolToLayer)
      if (layer.getConvention() == LayerConvention::Bind)
        layersRequiringBindFiles.insert(layer);

  // Handle layers used directly in this module.
  module->walk([&](LayerBlockOp layerBlock) {
    auto layer = symbolToLayer[layerBlock.getLayerNameAttr()];
    if (layer.getConvention() == LayerConvention::Inline)
      return;

    // Create a bindfile for any layer directly used in the module.
    layersRequiringBindFiles.insert(layer);

    // Determine names for all modules that will be created.
    auto moduleName = module.getModuleName();
    auto layerName = layerBlock.getLayerName();

    // A name hint for the module created from this layerblock.
    auto layerBlockModuleName = moduleNameForLayer(moduleName, layerName);

    // A name hint for the hier-path-op which targets the bound-in instance of
    // the module created from this layerblock.
    auto layerBlockHierPathName = hierPathNameForLayer(moduleName, layerName);

    LayerBlockGlobals globals;
    globals.moduleName = ns.newName(layerBlockModuleName);
    globals.hierPathName = ns.newName(layerBlockHierPathName);
    layerBlockGlobals.insert({layerBlock, globals});
  });

  // Create a bindfile for layers used indirectly under this module.
  for (auto *record : *node) {
    auto *child = record->getTarget()->getModule().getOperation();
    for (auto [layer, _] : bindFiles[child])
      layersRequiringBindFiles.insert(layer);
  }

  // Build the bindfiles for any layer seen under this module. The bindfiles are
  // emitted in the order which they are declared, for readability.
  for (auto [sym, layer] : symbolToLayer)
    if (layersRequiringBindFiles.contains(layer))
      buildBindFile(ns, node, b, sym, layer);
}

void LowerLayersPass::preprocessExtModule(CircuitNamespace &ns,
                                          InstanceGraphNode *node,
                                          FExtModuleOp extModule) {
  // For each known layer of the extmodule, compute and record the bindfile
  // name. When a layer is known, its parent layers are implicitly known,
  // so compute bindfiles for parent layers too. Use a set to avoid
  // repeated work, which can happen if, for example, both a child layer and
  // a parent layer are explicitly declared to be known.
  auto known = extModule.getKnownLayersAttr().getAsRange<SymbolRefAttr>();
  if (known.empty())
    return;

  auto moduleName = extModule.getExtModuleName();
  auto &files = bindFiles[extModule];
  SmallPtrSet<Operation *, 8> seen;

  for (auto name : known) {
    auto layer = symbolToLayer[name];
    auto rootLayerName = name.getRootReference();
    auto nestedLayerNames = name.getNestedReferences();
    while (layer && std::get<bool>(seen.insert(layer))) {
      if (layer.getConvention() == LayerConvention::Bind) {
        BindFileInfo info;
        auto filename =
            fileNameForLayer(moduleName, rootLayerName, nestedLayerNames);
        info.filename = StringAttr::get(&getContext(), filename);
        info.body = nullptr;
        files.insert({layer, info});
      }
      layer = layer->getParentOfType<LayerOp>();
      if (!nestedLayerNames.empty())
        nestedLayerNames = nestedLayerNames.drop_back();
    }
  }
}

void LowerLayersPass::preprocessModuleLike(CircuitNamespace &ns,
                                           InstanceGraphNode *node) {
  auto *op = node->getModule().getOperation();
  if (!op)
    return;

  if (auto module = dyn_cast<FModuleOp>(op))
    return preprocessModule(ns, node, module);

  if (auto extModule = dyn_cast<FExtModuleOp>(op))
    return preprocessExtModule(ns, node, extModule);
}

/// Create the bind file skeleton for each layer, for each module.
void LowerLayersPass::preprocessModules(CircuitNamespace &ns,
                                        InstanceGraph &ig) {
  ig.walkPostOrder([&](auto &node) { preprocessModuleLike(ns, &node); });
}

/// Process a circuit to remove all layer blocks in each module and top-level
/// layer definition.
void LowerLayersPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "==----- Running LowerLayers "
                      "-------------------------------------------------===\n");
  CircuitOp circuitOp = getOperation();

  // Initialize members which cannot be initialized automatically.
  llvm::sys::SmartMutex<true> mutex;
  circuitMutex = &mutex;

  auto *ig = &getAnalysis<InstanceGraph>();
  CircuitNamespace ns(circuitOp);
  hw::HierPathCache hpc(
      &ns, OpBuilder::InsertPoint(getOperation().getBodyBlock(),
                                  getOperation().getBodyBlock()->begin()));
  hierPathCache = &hpc;

  preprocessLayers(ns);
  preprocessModules(ns, *ig);

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

  // All layers definitions can now be deleted.
  for (auto layerOp :
       llvm::make_early_inc_range(circuitOp.getBodyBlock()->getOps<LayerOp>()))
    layerOp.erase();

  // Cleanup state.
  circuitMutex = nullptr;
  layerBlockGlobals.clear();
  macroNames.clear();
  symbolToLayer.clear();
  hierPathCache = nullptr;
  bindFiles.clear();
}
