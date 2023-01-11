//===- FIRRTLUndef.cpp - Infer Read Write Memory --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the undef analysis pass.  This analyzes the source to
// discover when uninitialized power-on state can impact post-reset state.
//
//===----------------------------------------------------------------------===//

#include "../Transforms/PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"

#include "mlir/IR/BuiltinOps.h"

#include <set>

using namespace mlir;
using namespace circt::firrtl;

namespace {
class LatticeValue {
  enum Kind {
    /// A value with a yet-to-be-determined value.  This is an unprocessed
    /// value.
    Unknown,

    // Value is live, but derives from an indeterminate value.
    Undefined,

    // Value is live and derived from a controlled value.
    Valid,

    // Value is live and derived from an external signal.
    External
  };

  LatticeValue(Kind k) : tag(k) {}

public:
  LatticeValue() : tag(Kind::Unknown) {}

  static LatticeValue getUndefined() { return {Kind::Undefined}; }

  bool markUndefined() {
    if (tag == Undefined)
      return false;
    tag = Undefined;
    return true;
  }
  bool markValid() {
    if (tag == Valid)
      return false;
    tag = Valid;
    return true;
  }
  bool markExternal() {
    if (tag == External)
      return false;
    tag = External;
    return true;
  }

  bool isUndefined() { return tag == Undefined; }
  bool isValid() { return tag == Valid; }
  bool isExternal() { return tag == External; }

  bool mergeIn(LatticeValue other) {
    // no information
    if (other.tag == Unknown)
      return false;

    // Undefined is a terminal state
    if (tag == Undefined) {
      return false;
    }

    // Unknown adopts rhs
    if (tag == Unknown) {
      tag = other.tag;
      return true;
    }

    // Undefined is poison
    if (other.tag == Undefined) {
      if (tag == Undefined)
        return false;
      tag = Undefined;
      return true;
    }

    // LHS is not Unkown or Undefined
    // RHS is not Unkown or Undefined

    if (other.tag == External) {
      if (tag == External)
        return false;
      tag = External;
      return true;
    }

    // External <- Valid is a no-op
    // Valid < Valid is a no-op
    return false;
  }

  raw_ostream &print(raw_ostream &os) const {
    switch (tag) {
    case Unknown:
      os << "Unknown  ";
      break;
    case Undefined:
      os << "Undefined";
      break;
    case Valid:
      os << "Valid    ";
      break;
    case External:
      os << "External ";
      break;
    }
    return os;
  }

  bool operator==(const LatticeValue &rhs) const { return tag == rhs.tag; }
  bool operator!=(const LatticeValue &rhs) const { return tag != rhs.tag; }

private:
  Kind tag;
};

struct InstPath {
  const SmallVector<InstanceOp> *path;
};

struct ValueKey {
  Value value;
  InstPath path;
};

struct BlockKey {
  Block *block;
  InstPath path;
};

bool operator<(const BlockKey &lhs, const BlockKey &rhs) {
  if (lhs.block < rhs.block)
    return true;
  if (lhs.block > rhs.block)
    return false;
  return lhs.path.path < rhs.path.path;
}

raw_ostream &operator<<(raw_ostream &os, const InstPath &path) {
  os << "[";
  for (auto i : *path.path)
    os << i.getName() << ", ";
  os << "]";
  return os;
}
raw_ostream &operator<<(raw_ostream &os, const LatticeValue &lat) {
  lat.print(os);
  return os;
}

} // namespace

namespace llvm {
template <>
struct DenseMapInfo<ValueKey> {
  static ValueKey getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return {Value::getFromOpaquePointer(pointer), {nullptr}};
  }
  static ValueKey getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return {Value::getFromOpaquePointer(pointer), {nullptr}};
  }

  static unsigned getHashValue(const ValueKey &node) {
    return hash_combine(node.value, node.path.path);
  }
  static bool isEqual(const ValueKey &lhs, const ValueKey &rhs) {
    return lhs.value == rhs.value && lhs.path.path == rhs.path.path;
  }
};

} // namespace llvm

/*
undef = undef op *
valid = valid op valid
Extern = Extern op (valid, Extern)

undef = reg no-init
undef = reg init sig x, val y (x | y is undef)
extern = reg init sig x, val y (x | y is extern and x & y is not undef)


map<Value, LatticeValue>


for each M : Modules {
    for each e : M {
        visitor(e);
    }
}
while (!WL.empty()) {
    visit(wl.pop());
}
*/

namespace {
struct UndefAnalysisPass : public UndefAnalysisBase<UndefAnalysisPass> {
  void runOnOperation() override;

  void markBlockExecutable(BlockKey block);
  void visitOperation(Operation *op, InstPath path);

  /// Returns true if the given block is executable.
  bool isBlockExecutable(BlockKey block) const {
    return executableBlocks.count(block);
  }

  LatticeValue getLatticeValue(ValueKey value) { return latticeValues[value]; }

  // Force value to undefined
  void markUndefined(ValueKey value) {
    auto &entry = latticeValues[value];
    if (!entry.isUndefined()) {
      entry.markUndefined();
      changedLatticeValueWorklist.push_back(value);
    }
  }

  // Force value to external
  void markExternal(ValueKey value) {
    auto &entry = latticeValues[value];
    if (!entry.isExternal()) {
      entry.markExternal();
      changedLatticeValueWorklist.push_back(value);
    }
  }

  // Force value to valid
  void markValid(ValueKey value) {
    auto &entry = latticeValues[value];
    if (!entry.isValid()) {
      entry.markValid();
      changedLatticeValueWorklist.push_back(value);
    }
  }

  // merge an undefined value into the existing lattice element
  void mergeUndefined(ValueKey value) {
    mergeLattice(value, LatticeValue::getUndefined());
  }

  void mergeValues(ValueKey lhs, ValueKey rhs) {
    mergeLattice(lhs, latticeValues[rhs]);
  }

  void mergeLattice(ValueKey lhs, LatticeValue rhs) {
    auto &oldLhs = latticeValues[lhs];
    if (oldLhs.mergeIn(rhs)) {
      changedLatticeValueWorklist.push_back(lhs);
    }
  }

  bool setLattice(ValueKey lhs, LatticeValue newValue) {
    auto &oldLhs = latticeValues[lhs];
    if (oldLhs != newValue) {
      oldLhs = newValue;
      return true;
    }
    return false;
  }

  InstPath ExtendPath(InstPath, InstanceOp);

  // visitors:
  void visitConnect(ConnectOp con, InstPath path);
  void visitConnect(StrictConnectOp con, InstPath path);
  void visitInstance(InstanceOp inst, InstPath path);

private:
  /// This is the current instance graph for the Circuit.
  InstanceGraph *instanceGraph = nullptr;

  /// This keeps track of the current state of each tracked value.
  DenseMap<ValueKey, LatticeValue> latticeValues;

  /// A worklist of values whose LatticeValue recently changed, indicating the
  /// users need to be reprocessed.
  SmallVector<ValueKey, 64> changedLatticeValueWorklist;

  /// The set of blocks that are known to execute, or are intrinsically live.
  std::set<BlockKey> executableBlocks;

  /// This uniqifies instance paths.
  std::set<SmallVector<InstanceOp>> uniqPaths;

  /// This keeps track of users the instance results that correspond to output
  /// ports.
  DenseMap<ValueKey, ValueKey> resultPortToInstanceResultMapping;
};
} // namespace

void UndefAnalysisPass::runOnOperation() {
  auto circuit = getOperation();
  instanceGraph = &getAnalysis<InstanceGraph>();

  // Mark the input ports of the top-level modules as being external.  We ignore
  // all other public modules.
  auto top = cast<FModuleOp>(circuit.getMainModule());
  InstPath topPath{&*uniqPaths.emplace().first};
  for (auto port : top.getBodyBlock()->getArguments())
    markExternal({(Value)port, topPath});
  markBlockExecutable({top.getBodyBlock(), topPath});

  // If a value changed lattice state then reprocess any of its users.
  while (!changedLatticeValueWorklist.empty()) {
    auto [changedVal, path] = changedLatticeValueWorklist.pop_back_val();
    for (Operation *user : changedVal.getUsers()) {
      if (isBlockExecutable({user->getBlock(), path}))
        visitOperation(user, path);
    }
  }

  for (auto i : latticeValues) {
    llvm::errs() << i.second << " : " << i.first.path << " ";
    if (auto ba = i.first.value.dyn_cast<BlockArgument>()) {
      auto mod = cast<FModuleLike>(ba.getOwner()->getParentOp());
      llvm::errs() << mod->getAttrOfType<StringAttr>(
                          ::mlir::SymbolTable::getSymbolAttrName())
                   << ":" << mod.getPortName(ba.getArgNumber()) << "\n";
    } else {
      auto mod = i.first.value.getDefiningOp()->getParentOfType<FModuleOp>();
      llvm::errs() << mod->getAttrOfType<StringAttr>(
                          ::mlir::SymbolTable::getSymbolAttrName())
                   << ":" << i.first.value << "\n";
    }
  }
}

void UndefAnalysisPass::markBlockExecutable(BlockKey block) {
  if (!executableBlocks.insert(block).second)
    return;

  for (auto &op : *block.block) {
    visitOperation(&op, block.path);
  }
}

void UndefAnalysisPass::visitOperation(Operation *op, InstPath path) {
  if (auto reg = dyn_cast<RegOp>(op))
    return mergeUndefined({reg, path});
  if (isa<ConstantOp, SpecialConstantOp, AggregateConstantOp>(op))
    return markValid({op->getResult(0), path});
  if (isa<InvalidValueOp>(op))
    return markUndefined({op->getResult(0), path});
  if (auto con = dyn_cast<ConnectOp>(op))
    return visitConnect(con, path);
  if (auto con = dyn_cast<StrictConnectOp>(op))
    return visitConnect(con, path);
  if (auto inst = dyn_cast<InstanceOp>(op))
    return visitInstance(inst, path);

  // Everything else just uses simple transfer functions
  for (auto result : op->getResults()) {
    bool changed = false;
    auto newLatticeValue = LatticeValue::getUndefined();
    for (auto operand : op->getOperands())
      newLatticeValue.mergeIn(getLatticeValue({operand, path}));
    changed |= setLattice({result, path}, newLatticeValue);
    if (changed)
      changedLatticeValueWorklist.push_back({result, path});
  }
}

void UndefAnalysisPass::visitConnect(ConnectOp con, InstPath path) {
  con.emitError("Should not have connects");
  abort();
}

void UndefAnalysisPass::visitConnect(StrictConnectOp con, InstPath path) {
  // Driving result ports propagates the value to each instance using the
  // module.
  if (auto blockArg = con.getDest().dyn_cast<BlockArgument>()) {

    if (auto userOfResultPort =
            resultPortToInstanceResultMapping.find({blockArg, path});
        userOfResultPort != resultPortToInstanceResultMapping.end())
      mergeValues(userOfResultPort->getSecond(), {con.getSrc(), path});
    // Output ports are wire-like and may have users.
    return mergeValues({con.getDest(), path}, {con.getSrc(), path});
  }

  auto dest = con.getDest().cast<mlir::OpResult>();

  // For wires and registers, we drive the value of the wire itself, which
  // automatically propagates to users.
  if (isa<RegOp, RegResetOp, WireOp>(dest.getOwner()))
    return mergeValues({con.getDest(), path}, {con.getSrc(), path});

  // Driving an instance argument port drives the corresponding argument of the
  // referenced module.
  if (auto instance = dest.getDefiningOp<InstanceOp>()) {
    // Update the dest, when its an instance op.
    mergeValues({con.getDest(), path}, {con.getSrc(), path});
    auto module =
        dyn_cast<FModuleOp>(*instanceGraph->getReferencedModule(instance));
    if (!module) // External or something
      return;

    BlockArgument modulePortVal = module.getArgument(dest.getResultNumber());
    return mergeValues({modulePortVal, ExtendPath(path, instance)},
                       {con.getSrc(), path});
  }

  // Driving a memory result is ignored because these are always treated as
  // overdefined.
  if (auto subfield = dest.getDefiningOp<SubfieldOp>()) {
    if (subfield.getOperand().getDefiningOp<MemOp>())
      return;
  }

  con.emitError("connectlike operation unhandled by FIRRTLUndef")
          .attachNote(con.getDest().getLoc())
      << "connect destination is here";

  //  mergeValues({con.getDest(), path}, {con.getSrc(), path});
}

void UndefAnalysisPass::visitInstance(InstanceOp inst, InstPath path) {
  auto modOp = inst.getReferencedModule();
  if (!modOp) {
    inst.emitError("Can't find module");
    return;
  } else if (auto mod = dyn_cast<FModuleOp>(modOp)) {
    auto newPath = ExtendPath(path, inst);
    markBlockExecutable({mod.getBodyBlock(), newPath});

    // Ok, it is a normal internal module reference.  Populate
    // resultPortToInstanceResultMapping, and forward any already-computed
    // values.
    for (size_t resultNo = 0, e = inst.getNumResults(); resultNo != e;
         ++resultNo) {
      auto instancePortVal = inst.getResult(resultNo);
      // If this is an input to the instance, it will
      // get handled when any connects to it are processed.
      if (mod.getPortDirection(resultNo) == Direction::In)
        continue;
      // We only support simple values so far.
      auto portType = instancePortVal.getType().dyn_cast<FIRRTLBaseType>();
      if (portType && !portType.isGround()) {
        inst.emitError("Cannot handle aggregates in instance");
        continue;
      }

      // Otherwise we have a result from the instance.  We need to forward
      // results from the body to this instance result's SSA value, so remember
      // it.
      BlockArgument modulePortVal = mod.getArgument(resultNo);

      resultPortToInstanceResultMapping[{modulePortVal, newPath}] = {
          instancePortVal, path};

      // If there is already a value known for modulePortVal make sure to
      // forward it here.
      mergeValues({instancePortVal, path}, {modulePortVal, newPath});
    }

  } else if (auto mod = dyn_cast<FExtModuleOp>(modOp)) {
    // output ports of external modules are External
    for (size_t resultNo = 0, e = inst.getNumResults(); resultNo != e;
         ++resultNo) {
      auto portVal = inst.getResult(resultNo);
      // If this is an input to the extmodule, we can ignore it.
      if (mod.getPortDirection(resultNo) == Direction::In)
        continue;

      // Otherwise this is a result from it or an inout, mark it as External.
      markExternal({portVal, path});
    }
  } else {
    inst.emitError("Unknown type of module");
  }
}

InstPath UndefAnalysisPass::ExtendPath(InstPath path, InstanceOp inst) {
  auto newPath = *path.path;
  newPath.push_back(inst);
  auto newUniqPath = uniqPaths.emplace(newPath).first;
  return {&*newUniqPath};
}

std::unique_ptr<mlir::Pass> circt::firrtl::createUndefAnalysisPass() {
  return std::make_unique<UndefAnalysisPass>();
}
