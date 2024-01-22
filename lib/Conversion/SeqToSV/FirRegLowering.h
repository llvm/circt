//===- FirRegLowering.h - FirReg lowering utilities ===========--*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CONVERSION_SEQTOSV_FIRREGLOWERING_H
#define CONVERSION_SEQTOSV_FIRREGLOWERING_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"
#include <variant>

namespace circt {
namespace hw {
namespace detail {

struct SCCNode {
  Operation *op;
  llvm::function_ref<bool(Operation *)> filter;
  unsigned index;
  explicit SCCNode(Operation *o, llvm::function_ref<bool(Operation *)> f,
                   unsigned index)
      : op(o), filter(f), index(index) {}
  bool operator==(const SCCNode &other) const {
    return other.op == op && other.index == index;
  }
  bool operator!=(const SCCNode &other) const { return !(*this == other); }
};

// Port iterator is not being used currently, for the purpose of FirRegLowering,
// only the FirReg are considered as roots for traversal.
class PortIterator
    : public llvm::iterator_facade_base<PortIterator, std::forward_iterator_tag,
                                        SCCNode> {
  HWModuleOp module;
  unsigned numPorts;
  SCCNode sccOp;

public:
  explicit PortIterator(SCCNode node, bool end = false)
      : module(cast<HWModuleOp>(node.op)), numPorts(module.getNumPorts()),
        sccOp(node.op, node.filter, end ? numPorts : 0) {}

  bool operator==(const PortIterator &other) const {
    return other.sccOp == this->sccOp;
  }
  bool operator!=(const PortIterator &other) const { return !(*this == other); }
  SCCNode operator*() { return sccOp; }
  PortIterator &operator++() {
    assert(sccOp.index < numPorts);
    ++sccOp.index;
    return *this;
  }
  PortIterator operator++(int) {
    PortIterator result(*this);
    ++(*this);
    return result;
  }
};

class ModOpIterator
    : public llvm::iterator_facade_base<ModOpIterator,
                                        std::forward_iterator_tag, SCCNode> {
  SCCNode sccOp;
  HWModuleOp module;
  Region::OpIterator modOpsEnd, modOpsIterator;

public:
  explicit ModOpIterator(SCCNode op, bool end = false)
      : sccOp(op), module(cast<HWModuleOp>(sccOp.op)),
        modOpsEnd(&module.getRegion(), true),
        modOpsIterator(end ? modOpsEnd
                           : module.getOps<seq::FirRegOp>().begin()) {}

  bool operator==(const ModOpIterator &other) const {
    return other.modOpsIterator == this->modOpsIterator;
  }

  bool operator!=(const ModOpIterator &other) const {
    return !(*this == other);
  }

  SCCNode operator*() {
    Operation &op = *modOpsIterator;
    return SCCNode(&op, sccOp.filter, 0);
  }

  ModOpIterator &operator++() {
    assert(modOpsIterator != modOpsEnd);
    ++modOpsIterator;
    return *this;
  }

  ModOpIterator operator++(int) {
    ModOpIterator result(*this);
    ++(*this);
    return result;
  }
};

class ModuleIterator
    : public llvm::iterator_facade_base<ModuleIterator,
                                        std::forward_iterator_tag, SCCNode> {
  SCCNode sccOp;
  PortIterator pIter, pIterEnd;
  ModOpIterator opIter, opIterEnd;

public:
  explicit ModuleIterator(SCCNode op, bool end = false,
                          bool ignorePorts = false)
      : sccOp(op), pIter(op, end || ignorePorts), pIterEnd(op, true),
        opIter(op, end || !ignorePorts), opIterEnd(op, true) {}

  bool operator==(const ModuleIterator &other) const {
    return other.pIter == this->pIter && other.opIter == this->opIter;
  }

  bool operator!=(const ModuleIterator &other) const {
    return !(*this == other);
  }

  SCCNode operator*() {
    if (pIter != pIterEnd)
      return *pIter;
    return *opIter;
  }

  ModuleIterator &operator++() {
    if (pIter == pIterEnd)
      ++opIter;
    else
      ++pIter;

    return *this;
  }

  ModuleIterator operator++(int) {
    ModuleIterator result(*this);
    ++(*this);
    return result;
  }
};

class OpUseIterator
    : public llvm::iterator_facade_base<OpUseIterator,
                                        std::forward_iterator_tag, SCCNode> {
  SCCNode sccOp;
  mlir::Value::use_iterator iterator, itEnd;

public:
  explicit OpUseIterator(SCCNode op, bool end = false) : sccOp(op) {
    iterator = itEnd = mlir::Value::use_iterator();
    if (end || sccOp.op->getNumResults() == 0)
      return;

    iterator = sccOp.op->getResult(sccOp.index).use_begin();
    getNextValid();
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  void getNextValid() {
    if (iterator == itEnd) {
      if (sccOp.index < sccOp.op->getNumResults() - 1)
        ++sccOp.index;
      else
        return;
      iterator = sccOp.op->getResult(sccOp.index).use_begin();
    }
    if (iterator != itEnd) {
      Operation *nextOp = iterator.getUser();
      if (sccOp.filter && sccOp.filter(nextOp)) {
        ++iterator;
        getNextValid();
      }
    }
  }

  bool operator==(const OpUseIterator &other) const {
    return other.iterator == this->iterator;
  }

  bool operator!=(const OpUseIterator &other) const {
    return !(*this == other);
  }

  SCCNode operator*() {
    Operation *op = iterator.getUser();
    return SCCNode(op, sccOp.filter, 0);
  }

  OpUseIterator &operator++() {
    ++iterator;
    getNextValid();

    return *this;
  }

  OpUseIterator operator++(int) {
    OpUseIterator result(*this);
    ++(*this);
    return result;
  }
};

class SCCIterator
    : public llvm::iterator_facade_base<SCCIterator, std::forward_iterator_tag,
                                        SCCNode> {
  SCCNode sccOp;

  using IteratorType = std::variant<ModuleIterator, OpUseIterator>;
  IteratorType iterator, itEnd;

public:
  explicit SCCIterator(SCCNode op, bool end = false)
      : sccOp(op), iterator(init(end)), itEnd(init(true)) {}

  IteratorType init(bool end = false) {
    if (isa<HWModuleOp>(sccOp.op))
      return ModuleIterator(sccOp, end, true);
    return OpUseIterator(sccOp, end);
  }

  bool operator==(const SCCIterator &other) const {
    return other.iterator == this->iterator;
  }

  bool operator!=(const SCCIterator &other) const { return !(*this == other); }

  SCCNode operator*() {
    switch (iterator.index()) {
    case 0:
      return *std::get<ModuleIterator>(iterator);
    case 1:
      return *std::get<OpUseIterator>(iterator);
    default:
      llvm_unreachable("invalid variant type");
    }
  }

  SCCIterator &operator++() {
    switch (iterator.index()) {
    case 0:
      return ++std::get<ModuleIterator>(iterator), *this;
    case 1:
      return ++std::get<OpUseIterator>(iterator), *this;
    default:
      llvm_unreachable("invalid variant type");
    }
  }

  SCCIterator operator++(int) {
    SCCIterator result(*this);
    ++(*this);
    return result;
  }
};
} // namespace detail
} // namespace hw
} // namespace circt
namespace llvm {
template <>
struct DenseMapInfo<circt::hw::detail::SCCNode> {
  using Node = circt::hw::detail::SCCNode;

  static Node getEmptyKey() {
    return Node(DenseMapInfo<mlir::Operation *>::getEmptyKey(), nullptr, 0);
  }

  static Node getTombstoneKey() {
    return Node(DenseMapInfo<mlir::Operation *>::getTombstoneKey(), nullptr, 0);
  }

  static unsigned getHashValue(const Node &node) {
    return detail::combineHashValue(
        DenseMapInfo<mlir::Operation *>::getHashValue(node.op), node.index);
  }

  static bool isEqual(const Node &lhs, const Node &rhs) { return lhs == rhs; }
};
} // namespace llvm

template <>
struct llvm::GraphTraits<circt::hw::detail::SCCNode> {
  using NodeType = circt::hw::detail::SCCNode;
  using NodeRef = NodeType;
  using ChildIteratorType = circt::hw::detail::SCCIterator;

  static NodeRef getEntryNode(NodeRef op) { return op; }

  // NOLINTNEXTLINE(readability-identifier-naming)
  static inline ChildIteratorType child_begin(NodeRef op) {
    return circt::hw::detail::SCCIterator(op);
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  static inline ChildIteratorType child_end(NodeRef op) {
    return circt::hw::detail::SCCIterator(op, true);
  }
};

namespace circt {

// Construct the SCC that each FirRegOp belongs to.
struct FirRegSCC {
  using SccOpType = circt::hw::detail::SCCNode;
  FirRegSCC(hw::HWModuleOp moduleOp,
            llvm::function_ref<bool(Operation *)> f = nullptr) {
    SccOpType sccOp(moduleOp, f, 0);

    for (llvm::scc_iterator<SccOpType> i = llvm::scc_begin(sccOp),
                                       e = llvm::scc_end(sccOp);
         i != e; ++i) {
      char numFirReg = 0;
      for (auto node : *i) {
        if (numFirReg < 2 && isa<seq::FirRegOp>(node.op))
          ++numFirReg;

        opSccIdMap[node.op] = sccIdGen;
      }
      if (numFirReg >= 2)
        multipleFirReginSCC.insert(sccIdGen);
      ++sccIdGen;
    }
  };
  llvm::SmallDenseSet<unsigned> multipleFirReginSCC;

  bool isCombReachable(SccOpType &root, Operation *dest,
                       llvm::SmallDenseSet<SccOpType> &visitedSet) const {
    if (root.op == dest)
      return true;
    if (!visitedSet.insert(root).second)
      return false;
    for (hw::detail::OpUseIterator it(root), end(root, true); it != end; ++it) {
      SccOpType n = *it;
      if (isCombReachable(n, dest, visitedSet))
        return true;
    }

    return false;
  };

  bool isInSameSCC(Operation *mux, Operation *reg) const {
    auto lhsIt = opSccIdMap.find(mux);
    auto rhsIt = opSccIdMap.find(reg);

    if (lhsIt != opSccIdMap.end() && rhsIt != opSccIdMap.end() &&
        lhsIt->getSecond() == rhsIt->getSecond()) {
      auto id = lhsIt->getSecond();
      if (!multipleFirReginSCC.contains(id))
        return true;

      auto filter = [&](Operation *op) {
        auto sccIt = opSccIdMap.find(op);
        return isa<seq::FirRegOp>(op) || sccIt == opSccIdMap.end() ||
               sccIt->getSecond() != id;
      };
      SccOpType root(reg, filter, 0);
      llvm::SmallDenseSet<SccOpType> visitedSet;
      return isCombReachable(root, mux, visitedSet);
    }
    return false;
  }

  void erase(Operation *op) { opSccIdMap.erase(op); }

private:
  llvm::DenseMap<Operation *, size_t> opSccIdMap;
  unsigned sccIdGen = 0;
};

/// Lower FirRegOp to `sv.reg` and `sv.always`.
class FirRegLowering {
public:
  FirRegLowering(TypeConverter &typeConverter, hw::HWModuleOp module,
                 bool disableRegRandomization = false,
                 bool emitSeparateAlwaysBlocks = false);

  void lower();
  bool needsRegRandomization() const { return needsRandom; }

  unsigned numSubaccessRestored = 0;

private:
  struct RegLowerInfo {
    sv::RegOp reg;
    IntegerAttr preset;
    Value asyncResetSignal;
    Value asyncResetValue;
    int64_t randStart;
    size_t width;
  };

  RegLowerInfo lower(seq::FirRegOp reg);

  void initialize(OpBuilder &builder, RegLowerInfo reg, ArrayRef<Value> rands);
  void initializeRegisterElements(Location loc, OpBuilder &builder, Value reg,
                                  Value rand, unsigned &pos);

  void createTree(OpBuilder &builder, Value reg, Value term, Value next);
  std::optional<std::tuple<Value, Value, Value>>
  tryRestoringSubaccess(OpBuilder &builder, Value reg, Value term,
                        hw::ArrayCreateOp nextRegValue);

  void addToAlwaysBlock(Block *block, sv::EventControl clockEdge, Value clock,
                        const std::function<void(OpBuilder &)> &body,
                        ResetType resetStyle = {},
                        sv::EventControl resetEdge = {}, Value reset = {},
                        const std::function<void(OpBuilder &)> &resetBody = {});

  void addToIfBlock(OpBuilder &builder, Value cond,
                    const std::function<void()> &trueSide,
                    const std::function<void()> &falseSide);

  hw::ConstantOp getOrCreateConstant(Location loc, const APInt &value) {
    OpBuilder builder(module.getBody());
    auto &constant = constantCache[value];
    if (constant) {
      constant->setLoc(builder.getFusedLoc({constant->getLoc(), loc}));
      return constant;
    }

    constant = builder.create<hw::ConstantOp>(loc, value);
    return constant;
  }

  using AlwaysKeyType = std::tuple<Block *, sv::EventControl, Value, ResetType,
                                   sv::EventControl, Value>;
  llvm::SmallDenseMap<AlwaysKeyType, std::pair<sv::AlwaysOp, sv::IfOp>>
      alwaysBlocks;

  using IfKeyType = std::pair<Block *, Value>;
  llvm::SmallDenseMap<IfKeyType, sv::IfOp> ifCache;

  llvm::SmallDenseMap<APInt, hw::ConstantOp> constantCache;
  llvm::SmallDenseMap<std::pair<Value, unsigned>, Value> arrayIndexCache;
  std::unique_ptr<FirRegSCC> scc;

  TypeConverter &typeConverter;
  hw::HWModuleOp module;

  bool disableRegRandomization;
  bool emitSeparateAlwaysBlocks;

  bool needsRandom = false;
};
} // namespace circt

#endif // CONVERSION_SEQTOSV_FIRREGLOWERING_H
