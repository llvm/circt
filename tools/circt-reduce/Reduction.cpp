//===- Reduction.cpp - Reductions for circt-reduce ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines abstract reduction patterns for the 'circt-reduce' tool.
//
//===----------------------------------------------------------------------===//

#include "Reduction.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/InitAllDialects.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Reducer/Tester.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "circt-reduce"

using namespace llvm;
using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// A utility doing lazy construction of `SymbolTable`s and `SymbolUserMap`s,
/// which is handy for reductions that need to look up a lot of symbols.
struct SymbolCache {
  SymbolTable &getSymbolTable(Operation *op) {
    return tables.getSymbolTable(op);
  }
  SymbolTable &getNearestSymbolTable(Operation *op) {
    return getSymbolTable(SymbolTable::getNearestSymbolTable(op));
  }

  SymbolUserMap &getSymbolUserMap(Operation *op) {
    auto it = userMaps.find(op);
    if (it != userMaps.end())
      return it->second;
    return userMaps.insert({op, SymbolUserMap(tables, op)}).first->second;
  }
  SymbolUserMap &getNearestSymbolUserMap(Operation *op) {
    return getSymbolUserMap(SymbolTable::getNearestSymbolTable(op));
  }

  void clear() {
    tables = SymbolTableCollection();
    userMaps.clear();
  }

private:
  SymbolTableCollection tables;
  SmallDenseMap<Operation *, SymbolUserMap, 2> userMaps;
};

/// Check that all connections to a value are invalids.
static bool onlyInvalidated(Value arg) {
  return llvm::all_of(arg.getUses(), [](OpOperand &use) {
    auto *op = use.getOwner();
    if (!isa<firrtl::ConnectOp, firrtl::PartialConnectOp>(op))
      return false;
    if (use.getOperandNumber() != 0)
      return false;
    if (!op->getOperand(1).getDefiningOp<firrtl::InvalidValueOp>())
      return false;
    return true;
  });
}

//===----------------------------------------------------------------------===//
// Reduction
//===----------------------------------------------------------------------===//

Reduction::~Reduction() {}

//===----------------------------------------------------------------------===//
// Pass Reduction
//===----------------------------------------------------------------------===//

PassReduction::PassReduction(MLIRContext *context, std::unique_ptr<Pass> pass,
                             bool canIncreaseSize, bool oneShot)
    : context(context), canIncreaseSize(canIncreaseSize), oneShot(oneShot) {
  passName = pass->getArgument();
  if (passName.empty())
    passName = pass->getName();

  if (auto opName = pass->getOpName())
    pm = std::make_unique<PassManager>(context, *opName);
  else
    pm = std::make_unique<PassManager>(context);
  pm->addPass(std::move(pass));
}

bool PassReduction::match(Operation *op) {
  return op->getName().getStringRef() == pm->getOpName(*context);
}

LogicalResult PassReduction::rewrite(Operation *op) { return pm->run(op); }

std::string PassReduction::getName() const { return passName.str(); }

//===----------------------------------------------------------------------===//
// Concrete Sample Reductions (to later move into the dialects)
//===----------------------------------------------------------------------===//

/// A sample reduction pattern that maps `firrtl.module` to `firrtl.extmodule`.
struct ModuleExternalizer : public Reduction {
  bool match(Operation *op) override { return isa<firrtl::FModuleOp>(op); }
  LogicalResult rewrite(Operation *op) override {
    auto module = cast<firrtl::FModuleOp>(op);
    OpBuilder builder(module);
    builder.create<firrtl::FExtModuleOp>(
        module->getLoc(),
        module->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
        module.getPorts(), StringRef(), module.annotationsAttr());
    module->erase();
    return success();
  }
  std::string getName() const override { return "module-externalizer"; }
};

/// Invalidate all the leaf fields of a value with a given flippedness by
/// connecting an invalid value to them. This is useful for ensuring that all
/// output ports of an instance or memory (including those nested in bundles)
/// are properly invalidated.
static void invalidateOutputs(ImplicitLocOpBuilder &builder, Value value,
                              SmallDenseMap<Type, Value, 8> &invalidCache,
                              bool flip = false) {
  auto type = value.getType().dyn_cast<firrtl::FIRRTLType>();
  if (!type)
    return;

  // Descend into bundles by creating subfield ops.
  if (auto bundleType = type.dyn_cast<firrtl::BundleType>()) {
    for (auto &element : llvm::enumerate(bundleType.getElements())) {
      auto subfield =
          builder.createOrFold<firrtl::SubfieldOp>(value, element.index());
      invalidateOutputs(builder, subfield, invalidCache,
                        flip ^ element.value().isFlip);
      if (subfield.use_empty())
        subfield.getDefiningOp()->erase();
    }
    return;
  }

  // Descend into vectors by creating subindex ops.
  if (auto vectorType = type.dyn_cast<firrtl::FVectorType>()) {
    for (unsigned i = 0, e = vectorType.getNumElements(); i != e; ++i) {
      auto subindex = builder.createOrFold<firrtl::SubindexOp>(value, i);
      invalidateOutputs(builder, subindex, invalidCache, flip);
      if (subindex.use_empty())
        subindex.getDefiningOp()->erase();
    }
    return;
  }

  // Only drive outputs.
  if (flip)
    return;
  Value invalid = invalidCache.lookup(type);
  if (!invalid) {
    invalid = builder.create<firrtl::InvalidValueOp>(type);
    invalidCache.insert({type, invalid});
  }
  builder.create<firrtl::ConnectOp>(value, invalid);
}

/// Connect a value to every leave of a destination value.
static void connectToLeafs(ImplicitLocOpBuilder &builder, Value dest,
                           Value value) {
  auto type = dest.getType().dyn_cast<firrtl::FIRRTLType>();
  if (!type)
    return;
  if (auto bundleType = type.dyn_cast<firrtl::BundleType>()) {
    for (auto &element : llvm::enumerate(bundleType.getElements()))
      connectToLeafs(builder,
                     builder.create<firrtl::SubfieldOp>(dest, element.index()),
                     value);
    return;
  }
  if (auto vectorType = type.dyn_cast<firrtl::FVectorType>()) {
    for (unsigned i = 0, e = vectorType.getNumElements(); i != e; ++i)
      connectToLeafs(builder, builder.create<firrtl::SubindexOp>(dest, i),
                     value);
    return;
  }
  if (!type.isa<firrtl::UIntType>()) {
    if (type.isa<firrtl::SIntType>())
      value = builder.create<firrtl::AsSIntPrimOp>(value);
    else
      return;
  }
  builder.create<firrtl::ConnectOp>(dest, value);
}

/// Reduce all leaf fields of a value through an XOR tree.
static void reduceXor(ImplicitLocOpBuilder &builder, Value &into, Value value) {
  auto type = value.getType().dyn_cast<firrtl::FIRRTLType>();
  if (!type)
    return;
  if (auto bundleType = type.dyn_cast<firrtl::BundleType>()) {
    for (auto &element : llvm::enumerate(bundleType.getElements()))
      reduceXor(
          builder, into,
          builder.createOrFold<firrtl::SubfieldOp>(value, element.index()));
    return;
  }
  if (auto vectorType = type.dyn_cast<firrtl::FVectorType>()) {
    for (unsigned i = 0, e = vectorType.getNumElements(); i != e; ++i)
      reduceXor(builder, into,
                builder.createOrFold<firrtl::SubindexOp>(value, i));
    return;
  }
  if (!type.isa<firrtl::UIntType>()) {
    if (type.isa<firrtl::SIntType>())
      value = builder.create<firrtl::AsUIntPrimOp>(value);
    else
      return;
  }
  into = into ? builder.createOrFold<firrtl::XorPrimOp>(into, value) : value;
}

/// A sample reduction pattern that maps `firrtl.instance` to a set of
/// invalidated wires. This often shortcuts a long iterative process of connect
/// invalidation, module externalization, and wire stripping
struct InstanceStubber : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override { symbols.clear(); }

  bool match(Operation *op) override { return isa<firrtl::InstanceOp>(op); }

  LogicalResult rewrite(Operation *op) override {
    auto instOp = cast<firrtl::InstanceOp>(op);
    LLVM_DEBUG(llvm::dbgs() << "Stubbing instance `" << instOp.name() << "`\n");
    ImplicitLocOpBuilder builder(instOp.getLoc(), instOp);
    SmallDenseMap<Type, Value, 8> invalidCache;
    for (unsigned i = 0, e = instOp.getNumResults(); i != e; ++i) {
      auto result = instOp.getResult(i);
      auto name = builder.getStringAttr(Twine(instOp.name()) + "_" +
                                        instOp.getPortNameStr(i));
      auto wire = builder.create<firrtl::WireOp>(
          result.getType(), name, instOp.getPortAnnotation(i), StringAttr{});
      invalidateOutputs(builder, wire, invalidCache,
                        instOp.getPortDirection(i) == firrtl::Direction::In);
      result.replaceAllUsesWith(wire);
    }
    auto tableOp = SymbolTable::getNearestSymbolTable(instOp);
    auto moduleOp = instOp.getReferencedModule(symbols.getSymbolTable(tableOp));
    instOp->erase();
    if (symbols.getSymbolUserMap(tableOp).useEmpty(moduleOp)) {
      LLVM_DEBUG(llvm::dbgs() << "- Removing now unused module `"
                              << moduleOp.moduleName() << "`\n");
      moduleOp->erase();
    }
    return success();
  }

  std::string getName() const override { return "instance-stubber"; }
  bool acceptSizeIncrease() const override { return true; }

  SymbolCache symbols;
};

/// A sample reduction pattern that maps `firrtl.mem` to a set of invalidated
/// wires.
struct MemoryStubber : public Reduction {
  bool match(Operation *op) override { return isa<firrtl::MemOp>(op); }
  LogicalResult rewrite(Operation *op) override {
    auto memOp = cast<firrtl::MemOp>(op);
    LLVM_DEBUG(llvm::dbgs() << "Stubbing memory `" << memOp.name() << "`\n");
    ImplicitLocOpBuilder builder(memOp.getLoc(), memOp);
    SmallDenseMap<Type, Value, 8> invalidCache;
    Value xorInputs;
    SmallVector<Value> outputs;
    for (unsigned i = 0, e = memOp.getNumResults(); i != e; ++i) {
      auto result = memOp.getResult(i);
      auto name = builder.getStringAttr(Twine(memOp.name()) + "_" +
                                        memOp.getPortNameStr(i));
      auto wire = builder.create<firrtl::WireOp>(
          result.getType(), name, memOp.getPortAnnotation(i), StringAttr{});
      invalidateOutputs(builder, wire, invalidCache, true);
      result.replaceAllUsesWith(wire);

      // Isolate the input and output data fields of the port.
      Value input, output;
      switch (memOp.getPortKind(i)) {
      case firrtl::MemOp::PortKind::Read:
        output = builder.createOrFold<firrtl::SubfieldOp>(wire, 3);
        break;
      case firrtl::MemOp::PortKind::Write:
        input = builder.createOrFold<firrtl::SubfieldOp>(wire, 3);
        break;
      case firrtl::MemOp::PortKind::ReadWrite:
        input = builder.createOrFold<firrtl::SubfieldOp>(wire, 5);
        output = builder.createOrFold<firrtl::SubfieldOp>(wire, 3);
        break;
      }

      // Reduce all input ports to a single one through an XOR tree.
      unsigned numFields =
          wire.getType().cast<firrtl::BundleType>().getNumElements();
      for (unsigned i = 0; i != numFields; ++i) {
        if (i != 2 && i != 3 && i != 5)
          reduceXor(builder, xorInputs,
                    builder.createOrFold<firrtl::SubfieldOp>(wire, i));
      }
      if (input)
        reduceXor(builder, xorInputs, input);

      // Track the output port to hook it up to the XORd input later.
      if (output)
        outputs.push_back(output);
    }

    // Hook up the outputs.
    for (auto output : outputs)
      connectToLeafs(builder, output, xorInputs);

    memOp->erase();
    return success();
  }
  std::string getName() const override { return "memory-stubber"; }
  bool acceptSizeIncrease() const override { return true; }
};

/// Starting at the given `op`, traverse through it and its operands and erase
/// operations that have no more uses.
static void pruneUnusedOps(Operation *initialOp) {
  SmallVector<Operation *> worklist;
  SmallSet<Operation *, 4> handled;
  worklist.push_back(initialOp);
  while (!worklist.empty()) {
    auto op = worklist.pop_back_val();
    if (!op->use_empty())
      continue;
    for (auto arg : op->getOperands())
      if (auto argOp = arg.getDefiningOp())
        if (handled.insert(argOp).second)
          worklist.push_back(argOp);
    op->erase();
  }
}

/// Check whether an operation interacts with flows in any way, which can make
/// replacement and operand forwarding harder in some cases.
static bool isFlowSensitiveOp(Operation *op) {
  return isa<firrtl::WireOp, firrtl::RegOp, firrtl::RegResetOp,
             firrtl::InstanceOp, firrtl::SubfieldOp, firrtl::SubindexOp,
             firrtl::SubaccessOp>(op);
}

/// A sample reduction pattern that replaces all uses of an operation with one
/// of its operands. This can help pruning large parts of the expression tree
/// rapidly.
template <unsigned OpNum>
struct OperandForwarder : public Reduction {
  bool match(Operation *op) override {
    if (op->getNumResults() != 1 || op->getNumOperands() < 2 ||
        OpNum >= op->getNumOperands())
      return false;
    if (isFlowSensitiveOp(op))
      return false;
    auto resultTy = op->getResult(0).getType().dyn_cast<firrtl::FIRRTLType>();
    auto opTy = op->getOperand(OpNum).getType().dyn_cast<firrtl::FIRRTLType>();
    return resultTy && opTy &&
           resultTy.getWidthlessType() == opTy.getWidthlessType() &&
           (resultTy.getBitWidthOrSentinel() == -1) ==
               (opTy.getBitWidthOrSentinel() == -1) &&
           resultTy.isa<firrtl::UIntType, firrtl::SIntType>();
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    ImplicitLocOpBuilder builder(op->getLoc(), op);
    auto result = op->getResult(0);
    auto operand = op->getOperand(OpNum);
    auto resultTy = result.getType().cast<firrtl::FIRRTLType>();
    auto operandTy = operand.getType().cast<firrtl::FIRRTLType>();
    auto resultWidth = resultTy.getBitWidthOrSentinel();
    auto operandWidth = operandTy.getBitWidthOrSentinel();
    Value newOp;
    if (resultWidth < operandWidth)
      newOp =
          builder.createOrFold<firrtl::BitsPrimOp>(operand, resultWidth - 1, 0);
    else if (resultWidth > operandWidth)
      newOp = builder.createOrFold<firrtl::PadPrimOp>(operand, resultWidth);
    else
      newOp = operand;
    LLVM_DEBUG(llvm::dbgs() << "Forwarding " << newOp << " in " << *op << "\n");
    result.replaceAllUsesWith(newOp);
    pruneUnusedOps(op);
    return success();
  }
  std::string getName() const override {
    return ("operand" + Twine(OpNum) + "-forwarder").str();
  }
};

/// A sample reduction pattern that replaces operations with a constant zero of
/// their type.
struct Constantifier : public Reduction {
  bool match(Operation *op) override {
    if (op->getNumResults() != 1 || op->getNumOperands() == 0)
      return false;
    if (isFlowSensitiveOp(op))
      return false;
    auto type = op->getResult(0).getType().dyn_cast<firrtl::FIRRTLType>();
    return type && type.isa<firrtl::UIntType, firrtl::SIntType>();
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    OpBuilder builder(op);
    auto type = op->getResult(0).getType().cast<firrtl::FIRRTLType>();
    auto width = type.getBitWidthOrSentinel();
    if (width == -1)
      width = 64;
    auto newOp = builder.create<firrtl::ConstantOp>(
        op->getLoc(), type, APSInt(width, type.isa<firrtl::UIntType>()));
    op->replaceAllUsesWith(newOp);
    pruneUnusedOps(op);
    return success();
  }
  std::string getName() const override { return "constantifier"; }
};

/// A sample reduction pattern that replaces the right-hand-side of
/// `firrtl.connect` and `firrtl.partialconnect` operations with a
/// `firrtl.invalidvalue`. This removes uses from the fanin cone to these
/// connects and creates opportunities for reduction in DCE/CSE.
struct ConnectInvalidator : public Reduction {
  bool match(Operation *op) override {
    return isa<firrtl::ConnectOp, firrtl::PartialConnectOp>(op) &&
           !op->getOperand(1).getDefiningOp<firrtl::InvalidValueOp>();
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    auto rhs = op->getOperand(1);
    OpBuilder builder(op);
    auto invOp =
        builder.create<firrtl::InvalidValueOp>(rhs.getLoc(), rhs.getType());
    auto rhsOp = rhs.getDefiningOp();
    op->setOperand(1, invOp);
    if (rhsOp)
      pruneUnusedOps(rhsOp);
    return success();
  }
  std::string getName() const override { return "connect-invalidator"; }
  bool acceptSizeIncrease() const override { return true; }
};

/// A sample reduction pattern that removes operations which either produce no
/// results or their results have no users.
struct OperationPruner : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override { symbols.clear(); }
  bool match(Operation *op) override {
    return !isa<ModuleOp>(op) &&
           (op->getNumResults() == 0 || op->use_empty()) &&
           (!op->hasAttr(SymbolTable::getSymbolAttrName()) ||
            symbols.getNearestSymbolUserMap(op).useEmpty(op));
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    pruneUnusedOps(op);
    return success();
  }
  std::string getName() const override { return "operation-pruner"; }

  SymbolCache symbols;
};

/// A sample reduction pattern that removes FIRRTL annotations from ports and
/// operations.
struct AnnotationRemover : public Reduction {
  bool match(Operation *op) override {
    return op->hasAttr("annotations") || op->hasAttr("portAnnotations");
  }
  LogicalResult rewrite(Operation *op) override {
    auto emptyArray = ArrayAttr::get(op->getContext(), {});
    if (op->hasAttr("annotations"))
      op->setAttr("annotations", emptyArray);
    if (op->hasAttr("portAnnotations")) {
      auto attr = emptyArray;
      if (isa<firrtl::InstanceOp>(op))
        attr = ArrayAttr::get(
            op->getContext(),
            SmallVector<Attribute>(op->getNumResults(), emptyArray));
      op->setAttr("portAnnotations", attr);
    }
    return success();
  }
  std::string getName() const override { return "annotation-remover"; }
};

/// A sample reduction pattern that removes ports from the root `firrtl.module`
/// if the port is not used or just invalidated.
struct RootPortPruner : public Reduction {
  bool match(Operation *op) override {
    auto module = dyn_cast<firrtl::FModuleOp>(op);
    if (!module)
      return false;
    auto circuit = module->getParentOfType<firrtl::CircuitOp>();
    if (!circuit)
      return false;
    return circuit.nameAttr() == module.getNameAttr();
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    auto module = cast<firrtl::FModuleOp>(op);
    SmallVector<unsigned> dropPorts;
    for (unsigned i = 0, e = module.getNumPorts(); i != e; ++i) {
      if (onlyInvalidated(module.getArgument(i))) {
        dropPorts.push_back(i);
        for (auto user : module.getArgument(i).getUsers())
          user->erase();
      }
    }
    module.erasePorts(dropPorts);
    return success();
  }
  std::string getName() const override { return "root-port-pruner"; }
};

/// A sample reduction pattern that replaces instances of `firrtl.extmodule`
/// with wires.
struct ExtmoduleInstanceRemover : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override { symbols.clear(); }

  bool match(Operation *op) override {
    if (auto instOp = dyn_cast<firrtl::InstanceOp>(op))
      return isa<firrtl::FExtModuleOp>(
          instOp.getReferencedModule(symbols.getNearestSymbolTable(instOp)));
    return false;
  }
  LogicalResult rewrite(Operation *op) override {
    auto instOp = cast<firrtl::InstanceOp>(op);
    auto portInfo =
        instOp.getReferencedModule(symbols.getNearestSymbolTable(instOp))
            .getPorts();
    ImplicitLocOpBuilder builder(instOp.getLoc(), instOp);
    SmallVector<Value> replacementWires;
    for (firrtl::PortInfo info : portInfo) {
      auto wire = builder.create<firrtl::WireOp>(
          info.type, (Twine(instOp.name()) + "_" + info.getName()).str());
      if (info.isOutput()) {
        auto inv = builder.create<firrtl::InvalidValueOp>(info.type);
        builder.create<firrtl::ConnectOp>(wire, inv);
      }
      replacementWires.push_back(wire);
    }
    instOp.replaceAllUsesWith(std::move(replacementWires));
    instOp->erase();
    return success();
  }
  std::string getName() const override { return "extmodule-instance-remover"; }
  bool acceptSizeIncrease() const override { return true; }

  SymbolCache symbols;
};

/// A sample reduction pattern that replaces a single-use wire and register with
/// an operand of the source value of the connection.
template <unsigned OpNum>
struct ConnectSourceOperandForwarder : public Reduction {
  bool match(Operation *op) override {
    auto connect = dyn_cast<firrtl::ConnectOp>(op);
    if (!connect)
      return false;
    auto dest = connect.dest();
    auto *destOp = dest.getDefiningOp();

    // Ensure that the destination is used only once.
    if (!destOp || !destOp->hasOneUse() ||
        !isa<firrtl::WireOp, firrtl::RegOp, firrtl::RegResetOp>(destOp))
      return false;

    auto *srcOp = connect.src().getDefiningOp();
    if (!srcOp || OpNum >= srcOp->getNumOperands())
      return false;

    auto resultTy = dest.getType().dyn_cast<firrtl::FIRRTLType>();
    auto opTy =
        srcOp->getOperand(OpNum).getType().dyn_cast<firrtl::FIRRTLType>();

    return resultTy && opTy &&
           resultTy.getWidthlessType() == opTy.getWidthlessType() &&
           ((resultTy.getBitWidthOrSentinel() == -1) ==
            (opTy.getBitWidthOrSentinel() == -1)) &&
           resultTy.isa<firrtl::UIntType, firrtl::SIntType>();
  }

  LogicalResult rewrite(Operation *op) override {
    auto connect = cast<firrtl::ConnectOp>(op);
    auto *destOp = connect.dest().getDefiningOp();
    auto *srcOp = connect.src().getDefiningOp();
    auto forwardedOperand = srcOp->getOperand(OpNum);
    ImplicitLocOpBuilder builder(destOp->getLoc(), destOp);
    Value newDest;
    if (isa<firrtl::WireOp>(destOp))
      newDest = builder.create<firrtl::WireOp>(forwardedOperand.getType());
    else {
      // We can promote the register into a wire but we wouldn't do here because
      // the error might be caused by the register.
      auto clock = destOp->getOperand(0);
      newDest =
          builder.create<firrtl::RegOp>(forwardedOperand.getType(), clock);
    }

    // Create new connection between a new wire and the forwarded operand.
    builder.setInsertionPointAfter(op);
    builder.create<firrtl::ConnectOp>(newDest, forwardedOperand);

    // Remove the old connection and destination. We don't have to replace them
    // because destination has only one use.
    op->erase();
    destOp->erase();
    pruneUnusedOps(srcOp);

    return success();
  }
  std::string getName() const override {
    return ("connect-source-operand-" + Twine(OpNum) + "-forwarder").str();
  }
};

//===----------------------------------------------------------------------===//
// Reduction Registration
//===----------------------------------------------------------------------===//

static std::unique_ptr<Pass> createSimpleCanonicalizerPass() {
  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = false;
  return createCanonicalizerPass(config);
}

void circt::createAllReductions(
    MLIRContext *context,
    llvm::function_ref<void(std::unique_ptr<Reduction>)> add) {
  // Gather a list of reduction patterns that we should try. Ideally these are
  // sorted by decreasing reduction potential/benefit. For example, things that
  // can knock out entire modules while being cheap should be tried first,
  // before trying to tweak operands of individual arithmetic ops.
  add(std::make_unique<PassReduction>(context, firrtl::createLowerCHIRRTLPass(),
                                      true, true));
  add(std::make_unique<PassReduction>(context, firrtl::createInferWidthsPass(),
                                      true, true));
  add(std::make_unique<PassReduction>(context, firrtl::createInferResetsPass(),
                                      true, true));
  add(std::make_unique<PassReduction>(
      context, firrtl::createLowerFIRRTLTypesPass(), true, true));
  add(std::make_unique<PassReduction>(context, firrtl::createExpandWhensPass(),
                                      true, true));
  add(std::make_unique<PassReduction>(context, firrtl::createInlinerPass()));
  add(std::make_unique<PassReduction>(context,
                                      createSimpleCanonicalizerPass()));
  add(std::make_unique<PassReduction>(context,
                                      firrtl::createRemoveUnusedPortsPass()));
  add(std::make_unique<InstanceStubber>());
  add(std::make_unique<MemoryStubber>());
  add(std::make_unique<ModuleExternalizer>());
  add(std::make_unique<PassReduction>(context, createCSEPass()));
  add(std::make_unique<ConnectInvalidator>());
  add(std::make_unique<Constantifier>());
  add(std::make_unique<OperandForwarder<0>>());
  add(std::make_unique<OperandForwarder<1>>());
  add(std::make_unique<OperandForwarder<2>>());
  add(std::make_unique<OperationPruner>());
  add(std::make_unique<AnnotationRemover>());
  add(std::make_unique<RootPortPruner>());
  add(std::make_unique<ExtmoduleInstanceRemover>());
  add(std::make_unique<ConnectSourceOperandForwarder<0>>());
  add(std::make_unique<ConnectSourceOperandForwarder<1>>());
  add(std::make_unique<ConnectSourceOperandForwarder<2>>());
}
