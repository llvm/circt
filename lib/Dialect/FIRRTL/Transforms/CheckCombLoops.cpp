//===- CheckCombLoops.cpp - FIRRTL check combinational cycles ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the FIRRTL combinational cycles detection pass.
// Terminology:
// In the context of the circt that the MLIR represents, a Value is called the
// driver of another Value, if the driver actively sets the the other Value. The
// driver Value is responsible for determining the logic level of the driven
// Value.
// This pass is a dataflow analysis that interprets the operations of the
// circt to build a connectivity graph, which represents the driver
// relationships. Each node in this connectivity graph is a FieldRef, and an
// edge exists from a source node to a desination node if the source drives the
// destination..
// Algorithm to construct the connectivity graph.
// 1. Traverse each module in the Instance Graph bottom up.
// 2. Preprocess step: Construct the circt connectivity directed graph.
// Perform a dataflow analysis, on the domain of FieldRefs. Interpret each
// operation, to record the values that can potentially drive another value.
// Each node in the graph is a fieldRef. Each edge represents a dataflow from
// the source to the sink.
// 3. Visit the nodes in topological order, to detect combinational loops and
// record the sets of ports that reach each output port.
// 4. Add those sets to the graph of each instance's parent and continue the
// analysis through the instance graph.
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/bit.h"
#include <optional>

#define DEBUG_TYPE "check-comb-loops"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_CHECKCOMBLOOPS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

using DrivenBysMapType = DenseMap<FieldRef, DenseSet<FieldRef>>;

namespace {
/// Sets of port fields. Set i is field i alone for i below the number of
/// fields, and the union of two lower sets above it.
struct PortSets {
  SmallVector<FieldRef> fields;
  SmallVector<std::pair<unsigned, unsigned>> unions;

  unsigned size() const { return fields.size() + unions.size(); }
  bool isField(unsigned set) const { return set < fields.size(); }
  /// The two lower sets a union joins.
  std::pair<unsigned, unsigned> parts(unsigned set) const {
    return unions[set - fields.size()];
  }
  /// Calls `fn` once on each field of a set.
  void forEachField(unsigned set, llvm::function_ref<void(FieldRef)> fn) const {
    SmallVector<unsigned> stack{set};
    DenseSet<unsigned> seen;
    while (!stack.empty()) {
      auto part = stack.pop_back_val();
      if (!seen.insert(part).second)
        continue;
      if (isField(part)) {
        fn(fields[part]);
        continue;
      }
      auto [lhs, rhs] = parts(part);
      stack.append({lhs, rhs});
    }
  }
};

/// The combinational paths into a module's output ports, as the set of port
/// fields that reaches each of them.
struct PortPaths {
  PortSets sets;
  /// Each output port field reached by a port field, and its set.
  SmallVector<std::pair<FieldRef, unsigned>> outputs;
  /// The forceable output ports' paths, whose RWProbes are told apart from
  /// loops by the ports they reach.
  DrivenBysMapType forceable;
};

/// Port field sets as hash-consed big-endian Patricia tries (Okasaki and
/// Gill, "Fast Mergeable Integer Maps", 1998), in `sets`: set i below the
/// number of fields is field i, and each set above is a trie branch, the
/// union of its two lower subtries. Equal sets are one set, so the union of a
/// set with one it holds is that set. Fields are numbered in the order the
/// connectivity graph meets them, so the fields a node reaches tend to share
/// their high bits.
class PortTries {
public:
  static constexpr unsigned none = ~0u;

  explicit PortTries(SmallVector<FieldRef> fields) {
    sets.fields = std::move(fields);
  }

  /// The fields and the unions made so far.
  const PortSets &getSets() const { return sets; }

  /// The union of two sets. Sets are canonical, so a pair of operands names
  /// one result and is memoized. The merges run on an explicit stack of the
  /// branches still waiting for their subtries.
  unsigned unite(unsigned a, unsigned b) {
    unsigned set = start(a, b);
    while (!stack.empty()) {
      Frame &frame = stack.back();
      if (set != pending)
        frame.subtries[frame.next++] = set;
      if (frame.next < 2) {
        auto [lhs, rhs] = frame.operands[frame.next];
        set = start(lhs, rhs);
        continue;
      }
      auto key = frame.key;
      set =
          branch(frame.prefix, frame.bit, frame.subtries[0], frame.subtries[1]);
      stack.pop_back();
      unions[key] = set;
    }
    return set;
  }

private:
  /// A union whose frame is on the stack.
  static constexpr unsigned pending = ~0u - 1;

  /// The union of `a` and `b`, or `pending` with a frame for it on the
  /// stack.
  unsigned start(unsigned a, unsigned b) {
    if (a == b || b == none)
      return a;
    if (a == none)
      return b;
    auto key = std::minmax(a, b);
    if (auto it = unions.find(key); it != unions.end())
      return it->second;
    unsigned set = merge(key);
    if (set != pending)
      unions[key] = set;
    return set;
  }

  unsigned numFields() const { return sets.fields.size(); }
  bool isField(unsigned set) const { return sets.isField(set); }
  /// The high bits shared by a set's fields, above its branching bit.
  unsigned prefix(unsigned set) const {
    return isField(set) ? set : branches[set - numFields()].prefix;
  }
  unsigned bit(unsigned set) const { return branches[set - numFields()].bit; }
  unsigned zeros(unsigned set) const { return sets.parts(set).first; }
  unsigned ones(unsigned set) const { return sets.parts(set).second; }
  static unsigned mask(unsigned key, unsigned bit) {
    return key & ~(bit | (bit - 1));
  }
  static bool matches(unsigned key, unsigned prefix, unsigned bit) {
    return mask(key, bit) == prefix;
  }

  /// The trie branching on `bit` into `zeros` and `ones`, made once. It is
  /// their union, so it is memoized with the unions. A branch is made after
  /// its subtries, so its id is above theirs.
  unsigned branch(unsigned prefix, unsigned bit, unsigned zeros,
                  unsigned ones) {
    auto [it, inserted] =
        unions.try_emplace(std::minmax(zeros, ones), sets.size());
    if (inserted) {
      sets.unions.emplace_back(zeros, ones);
      branches.push_back({prefix, bit});
    }
    return it->second;
  }

  /// Joins two disjoint tries by their highest differing prefix bit.
  unsigned join(unsigned p, unsigned s, unsigned q, unsigned t) {
    unsigned bit = llvm::bit_floor(p ^ q);
    return (p & bit) ? branch(mask(p, bit), bit, t, s)
                     : branch(mask(p, bit), bit, s, t);
  }

  /// Pushes the branch on `bit` whose subtries are the unions of the two
  /// operand pairs.
  unsigned push(std::pair<unsigned, unsigned> key, unsigned prefix,
                unsigned bit, std::pair<unsigned, unsigned> zeros,
                std::pair<unsigned, unsigned> ones) {
    stack.push_back({key, prefix, bit, {zeros, ones}, {none, none}, 0});
    return pending;
  }

  /// The union of two distinct sets, or `pending` if it waits on the
  /// unions of their subtries.
  unsigned merge(std::pair<unsigned, unsigned> key) {
    auto [s, t] = key;
    if (isField(s) && isField(t))
      return join(s, s, t, t);
    if (isField(t))
      std::swap(s, t);
    unsigned q = prefix(t), n = bit(t);
    if (isField(s)) {
      if (!matches(s, q, n))
        return join(s, s, q, t);
      unsigned t0 = zeros(t), t1 = ones(t);
      return (s & n) ? push(key, q, n, {t0, none}, {s, t1})
                     : push(key, q, n, {s, t0}, {t1, none});
    }
    unsigned p = prefix(s), m = bit(s);
    unsigned s0 = zeros(s), s1 = ones(s), t0 = zeros(t), t1 = ones(t);
    if (m == n && p == q)
      return push(key, p, m, {s0, t0}, {s1, t1});
    if (m > n && matches(q, p, m))
      return (q & m) ? push(key, p, m, {s0, none}, {s1, t})
                     : push(key, p, m, {s0, t}, {s1, none});
    if (n > m && matches(p, q, n))
      return (p & n) ? push(key, q, n, {t0, none}, {s, t1})
                     : push(key, q, n, {s, t0}, {t1, none});
    return join(p, s, q, t);
  }

  struct Branch {
    unsigned prefix, bit;
  };
  /// A union waiting for the unions of its operand pairs, its subtries.
  struct Frame {
    std::pair<unsigned, unsigned> key;
    unsigned prefix, bit;
    std::pair<unsigned, unsigned> operands[2];
    unsigned subtries[2];
    unsigned next;
  };
  PortSets sets;
  SmallVector<Branch> branches;
  SmallVector<Frame> stack;
  /// Each union and branch by its operands.
  DenseMap<std::pair<unsigned, unsigned>, unsigned> unions;
};
} // namespace

class DiscoverLoops {

  /// Adjacency list representation.
  /// Each entry is a pair, the first element is the FieldRef corresponding to
  /// the graph vertex. The second element is the list of vertices, that have an
  /// edge to this vertex.
  /// The directed edges represent a connectivity relation, of a source that
  /// drives the sink.
  using DrivenByGraphType =
      SmallVector<std::pair<FieldRef, SmallVector<unsigned>>, 64>;

public:
  DiscoverLoops(FModuleOp module, InstanceGraph &instanceGraph,
                const DenseMap<FModuleLike, PortPaths> &otherModulePortPaths,
                PortPaths &thisModulePortPaths)
      : module(module), instanceGraph(instanceGraph),
        modulePortPaths(otherModulePortPaths), portPaths(thisModulePortPaths) {}

  /// Records the module's port paths. Fails on a combinational cycle, which
  /// is reported.
  LogicalResult processModule() {
    LLVM_DEBUG(llvm::dbgs() << "\n processing module :" << module.getName());
    constructConnectivityGraph(module);
    return recordPortPaths();
  }

  /// Records the paths between ports in one pass over the nodes in
  /// topological order, drivers first; reports a cycle and fails if the graph
  /// has one. Each node gets the set of port fields that reach it.
  LogicalResult recordPortPaths() {
    SmallVector<unsigned> reach;
    PortTries tries = seedPortSets(reach);
    if (failed(propagatePortSets(tries, reach)))
      return failure();
    exportPortPaths(tries, reach);
    return success();
  }

  /// Numbers the port fields the nodes hold, field i being set i, and sets
  /// each node's `reach` to the fields it holds: itself if it is a port, and
  /// the ports holding an RWProbe of it.
  PortTries seedPortSets(SmallVectorImpl<unsigned> &reach) {
    SmallVector<FieldRef> fields;
    DenseMap<FieldRef, unsigned> fieldIds;
    SmallVector<std::pair<unsigned, unsigned>> nodeFields;
    for (unsigned node = 0, e = drivenBy.size(); node < e; ++node) {
      auto ref = drivenBy[node].first;
      if (!ref)
        continue;
      SmallSetVector<FieldRef, 4> held;
      if (isa<BlockArgument>(ref.getValue()))
        held.insert(ref);
      addToPortPathsIfRWProbe(node, held);
      for (auto field : held) {
        auto [it, inserted] = fieldIds.try_emplace(field, fields.size());
        if (inserted)
          fields.push_back(field);
        nodeFields.emplace_back(node, it->second);
      }
    }
    PortTries tries(std::move(fields));
    reach.assign(drivenBy.size(), PortTries::none);
    for (auto [node, field] : nodeFields)
      reach[node] = tries.unite(reach[node], field);
    return tries;
  }

  /// Unites each node's set with its drivers', visiting the nodes in Kahn's
  /// order. Reports a cycle and fails if the graph has one.
  LogicalResult propagatePortSets(PortTries &tries,
                                  MutableArrayRef<unsigned> reach) {
    unsigned numNodes = drivenBy.size();
    SmallVector<unsigned> pending(numNodes);
    SmallVector<SmallVector<unsigned, 2>> sinks(numNodes);
    for (unsigned node = 0; node < numNodes; ++node) {
      pending[node] = drivenBy[node].second.size();
      for (auto driver : drivenBy[node].second)
        sinks[driver].push_back(node);
    }
    SmallVector<unsigned> order;
    for (unsigned node = 0; node < numNodes; ++node)
      if (!pending[node])
        order.push_back(node);
    for (size_t head = 0; head < order.size(); ++head) {
      unsigned node = order[head];
      for (auto driver : drivenBy[node].second)
        reach[node] = tries.unite(reach[node], reach[driver]);
      for (auto sink : sinks[node])
        if (!--pending[sink])
          order.push_back(sink);
    }
    if (order.size() == numNodes)
      return success();
    reportCycle(pending);
    return failure();
  }

  /// Exports each output port's set into `portPaths`, with the sets it is
  /// made of. A forceable port's set is exported flattened.
  void exportPortPaths(PortTries &tries, ArrayRef<unsigned> reach) {
    SmallVector<std::pair<FieldRef, unsigned>> outputs;
    for (auto &[port, drivers] : drivenBy) {
      auto arg = llvm::dyn_cast_if_present<BlockArgument>(port.getValue());
      if (!arg || module.getPortDirection(arg.getArgNumber()) != Direction::Out)
        continue;
      unsigned set = PortTries::none;
      for (auto driver : drivers)
        set = tries.unite(set, reach[driver]);
      if (set == PortTries::none)
        continue;
      auto ref = type_dyn_cast<RefType>(port.getValue().getType());
      if (!ref || !ref.getForceable()) {
        outputs.emplace_back(port, set);
        continue;
      }
      auto &inputs = portPaths.forceable[port];
      tries.getSets().forEachField(
          set, [&](FieldRef field) { inputs.insert(field); });
    }

    // Keep the sets the outputs need, in order. A union's parts are lower
    // sets, so one backward pass finds them.
    const PortSets &sets = tries.getSets();
    SmallVector<bool> needed(sets.size(), false);
    for (auto [port, set] : outputs)
      needed[set] = true;
    for (unsigned set = sets.size(); set-- > sets.fields.size();)
      if (needed[set]) {
        auto [lhs, rhs] = sets.parts(set);
        needed[lhs] = needed[rhs] = true;
      }
    PortSets &out = portPaths.sets;
    SmallVector<unsigned> renumbered(sets.size());
    for (unsigned set = 0; set < sets.size(); ++set) {
      if (!needed[set])
        continue;
      renumbered[set] = out.size();
      if (sets.isField(set)) {
        out.fields.push_back(sets.fields[set]);
        continue;
      }
      auto [lhs, rhs] = sets.parts(set);
      out.unions.emplace_back(renumbered[lhs], renumbered[rhs]);
    }
    for (auto [port, set] : outputs)
      portPaths.outputs.emplace_back(port, renumbered[set]);
  }

  void constructConnectivityGraph(FModuleOp module) {
    LLVM_DEBUG(llvm::dbgs() << "\n Module :" << module.getName());

    // ALl the module output ports must be added as the initial nodes.
    for (auto port : module.getArguments()) {
      if (module.getPortDirection(port.getArgNumber()) != Direction::Out)
        continue;
      walkGroundTypes(cast<FIRRTLType>(port.getType()),
                      [&](uint64_t index, FIRRTLBaseType t, auto isFlip) {
                        getOrAddNode(FieldRef(port, index));
                      });
    }

    walk(module, [&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .Case<hw::CombDataFlow>([&](hw::CombDataFlow df) {
            // computeDataFlow returns a pair of FieldRefs, first element is the
            // destination and the second is the source.
            for (auto [dest, source] : df.computeDataFlow())
              addDrivenBy(dest, source);
          })
          .Case<Forceable>([&](Forceable forceableOp) {
            // Any declaration that can be forced.
            if (auto node = dyn_cast<NodeOp>(op))
              recordDataflow(node.getData(), node.getInput());
            if (!forceableOp.isForceable() ||
                forceableOp.getDataRef().use_empty())
              return;
            auto data = forceableOp.getData();
            auto ref = forceableOp.getDataRef();
            // Record dataflow from data to the probe.
            recordDataflow(ref, data);
            recordProbe(data, ref);
          })
          .Case<RefSendOp>([&](RefSendOp send) {
            recordDataflow(send.getResult(), send.getBase());
          })
          .Case<RefDefineOp>([&](RefDefineOp def) {
            // Dataflow from src to dst.
            recordDataflow(def.getDest(), def.getSrc());
            if (!def.getDest().getType().getForceable())
              return;
            // Dataflow from dst to src, for RWProbe.
            probesReferToSameData(def.getSrc(), def.getDest());
          })
          .Case<RefForceOp, RefForceInitialOp>(
              [&](auto ref) { handleRefForce(ref.getDest(), ref.getSrc()); })
          .Case<InstanceOp>([&](auto inst) { handleInstanceOp(inst); })
          .Case<InstanceChoiceOp>(
              [&](auto inst) { handleInstanceChoiceOp(inst); })
          .Case<SubindexOp>([&](SubindexOp sub) {
            recordValueRefersToFieldRef(
                sub.getInput(),
                sub.getInput().getType().base().getFieldID(sub.getIndex()),
                sub.getResult());
          })
          .Case<SubfieldOp>([&](SubfieldOp sub) {
            recordValueRefersToFieldRef(
                sub.getInput(),
                sub.getInput().getType().base().getFieldID(sub.getFieldIndex()),
                sub.getResult());
          })
          .Case<SubaccessOp>([&](SubaccessOp sub) {
            auto vecType = sub.getInput().getType().base();
            auto input = sub.getInput();
            auto result = sub.getResult();
            // Flatten the subaccess. The result can refer to any of the
            // elements.
            for (size_t index = 0; index < vecType.getNumElements(); ++index)
              recordValueRefersToFieldRef(input, vecType.getFieldID(index),
                                          result);
          })
          .Case<RefSubOp>([&](RefSubOp sub) {
            size_t fieldID = TypeSwitch<FIRRTLBaseType, size_t>(
                                 sub.getInput().getType().getType())
                                 .Case<FVectorType, BundleType>([&](auto type) {
                                   return type.getFieldID(sub.getIndex());
                                 });
            recordValueRefersToFieldRef(sub.getInput(), fieldID,
                                        sub.getResult());
          })
          .Case<BundleCreateOp, VectorCreateOp>([&](auto op) {
            auto type = op.getType();
            auto res = op.getResult();
            auto getFieldId = [&](unsigned index) {
              size_t fieldID =
                  TypeSwitch<FIRRTLBaseType, size_t>(type)
                      .Case<FVectorType, BundleType>(
                          [&](auto type) { return type.getFieldID(index); });
              return fieldID;
            };
            for (auto [index, v] : llvm::enumerate(op.getOperands()))
              recordValueRefersToFieldRef(res, getFieldId(index), v);
          })
          .Case<FConnectLike>([&](FConnectLike connect) {
            recordDataflow(connect.getDest(), connect.getSrc());
          })
          .Default([&](Operation *op) {
            // All other expressions are assumed to be combinational, so record
            // the dataflow between all inputs to outputs.
            for (auto res : op->getResults())
              for (auto src : op->getOperands())
                recordDataflow(res, src);
          });
    });
  }

  static std::string getName(FieldRef v) { return getFieldName(v).first; };

  unsigned getOrAddNode(Value v) {
    auto iter = valToFieldRefs.find(v);
    if (iter == valToFieldRefs.end())
      return getOrAddNode({v, 0});
    return getOrAddNode(*iter->second.begin());
  }

  // Get the node id if it exists, else add it to the graph.
  unsigned getOrAddNode(FieldRef f) {
    auto iter = nodes.find(f);
    if (iter != nodes.end())
      return iter->second;
    // Add the fieldRef to the graph.
    auto id = drivenBy.size();
    // The node id can be used to index into the graph. The entry is a pair,
    // first element is the corresponding FieldRef, and the second entry is a
    // list of adjacent nodes.
    drivenBy.push_back({f, {}});
    nodes[f] = id;
    return id;
  }

  // Construct the connectivity graph, by adding `dst` and `src` as new nodes,
  // if not already existing. Then add an edge from `src` to `dst`.
  void addDrivenBy(FieldRef dst, FieldRef src) {
    auto srcNode = getOrAddNode(src);
    auto dstNode = getOrAddNode(dst);
    drivenBy[dstNode].second.push_back(srcNode);
  }

  // Add `dstVal` as being driven by `srcVal`.
  void recordDataflow(Value dstVal, Value srcVal) {
    // Ignore connectivity from constants.
    if (auto *def = srcVal.getDefiningOp())
      if (def->hasTrait<OpTrait::ConstantLike>())
        return;
    // Check if srcVal/dstVal is a fieldRef to an aggregate. Then, there may
    // exist other values, that refer to the same fieldRef. Add a connectivity
    // from all such "aliasing" values.
    auto dstIt = valToFieldRefs.find(dstVal);
    auto srcIt = valToFieldRefs.find(srcVal);

    // Block the dataflow through registers.
    // dstVal refers to a register, If dstVal is not recorded as the fieldref of
    // an aggregate, and its either a register, or result of a sub op.
    if (dstIt == valToFieldRefs.end())
      if (auto *def = dstVal.getDefiningOp())
        if (isa<RegOp, RegResetOp, SubaccessOp, SubfieldOp, SubindexOp,
                RefSubOp>(def))
          return;

    auto dstValType = getBaseType(dstVal.getType());
    auto srcValType = getBaseType(srcVal.getType());

    // Handle Properties.
    if (!(srcValType && dstValType))
      return addDrivenBy({dstVal, 0}, {srcVal, 0});

    auto addDef = [&](FieldRef dst, FieldRef src) {
      // If the dstVal and srcVal are aggregate types, then record the dataflow
      // between each individual ground type. This is equivalent to flattening
      // the type to ensure all the contained FieldRefs are also recorded.
      if (dstValType && !dstValType.isGround())
        walkGroundTypes(dstValType, [&](uint64_t dstIndex, FIRRTLBaseType t,
                                        bool dstIsFlip) {
          // Handle the case when the dst and src are not of the same type.
          // For each dst ground type, and for each src ground type.
          if (srcValType == dstValType) {
            // This is the only case when the flip is valid. Flip is relevant
            // only for connect ops, and src and dst of a connect op must be
            // type equivalent!
            if (dstIsFlip)
              std::swap(dst, src);
            addDrivenBy(dst.getSubField(dstIndex), src.getSubField(dstIndex));
          } else if (srcValType && !srcValType.isGround())
            walkGroundTypes(srcValType, [&](uint64_t srcIndex, FIRRTLBaseType t,
                                            auto) {
              addDrivenBy(dst.getSubField(dstIndex), src.getSubField(srcIndex));
            });
          // Else, the src is ground type.
          else
            addDrivenBy(dst.getSubField(dstIndex), src);
        });

      addDrivenBy(dst, src);
    };

    // Both the dstVal and srcVal, can refer to multiple FieldRefs, ensure that
    // we capture the dataflow between each pair. Occurs when val result of
    // subaccess op.

    // Case 1: None of src and dst are fields of an aggregate. But they can be
    // aggregate values.
    if (dstIt == valToFieldRefs.end() && srcIt == valToFieldRefs.end())
      return addDef({dstVal, 0}, {srcVal, 0});
    // Case 2: Only src is the field of an aggregate. Get all the fields that
    // the src refers to.
    if (dstIt == valToFieldRefs.end() && srcIt != valToFieldRefs.end()) {
      llvm::for_each(srcIt->getSecond(), [&](const FieldRef &srcField) {
        addDef(FieldRef(dstVal, 0), srcField);
      });
      return;
    }
    // Case 3: Only dst is the field of an aggregate. Get all the fields that
    // the dst refers to.
    if (dstIt != valToFieldRefs.end() && srcIt == valToFieldRefs.end()) {
      llvm::for_each(dstIt->getSecond(), [&](const FieldRef &dstField) {
        addDef(dstField, FieldRef(srcVal, 0));
      });
      return;
    }
    // Case 4: Both src and dst are the fields of an aggregate. Get all the
    // fields that they refers to.
    llvm::for_each(srcIt->second, [&](const FieldRef &srcField) {
      llvm::for_each(dstIt->second, [&](const FieldRef &dstField) {
        addDef(dstField, srcField);
      });
    });
  }

  // Record srcVal as driving the original data value that the probe refers to.
  void handleRefForce(Value dstProbe, Value srcVal) {
    recordDataflow(dstProbe, srcVal);
    auto dstNode = getOrAddNode(dstProbe);
    // Now add srcVal as driving the data that dstProbe refers to.
    auto leader = rwProbeClasses.findLeader(dstNode);
    if (leader == rwProbeClasses.member_end())
      return;
    auto iter = rwProbeRefersTo.find(*leader);

    // This should be found, but for now may not be due to needing
    // RWProbeOp support.  May cause missed loops involving force for now.
    // https://github.com/llvm/circt/issues/6820
    if (iter == rwProbeRefersTo.end())
      return;

    assert(iter != rwProbeRefersTo.end());
    if (iter->second != dstNode)
      drivenBy[iter->second].second.push_back(getOrAddNode(srcVal));
  }

  // Helper to process instance ports for a given module and instance results.
  // This is used by both handleInstanceOp and handleInstanceChoiceOp.
  void processInstancePorts(FModuleOp refMod, ValueRange instResults) {
    auto modulePaths = modulePortPaths.find(refMod);
    if (modulePaths == modulePortPaths.end())
      return;
    const PortPaths &paths = modulePaths->second;
    auto instancePort = [&](FieldRef modPort) {
      auto argNum = cast<BlockArgument>(modPort.getValue()).getArgNumber();
      return FieldRef(instResults[argNum], modPort.getFieldID());
    };

    // Field i's node is its instance port, and each union's node is a hub
    // driven by its parts' nodes.
    SmallVector<unsigned> setNodes;
    for (auto field : paths.sets.fields)
      setNodes.push_back(getOrAddNode(instancePort(field)));
    for (auto [lhs, rhs] : paths.sets.unions) {
      drivenBy.push_back({FieldRef(), {setNodes[lhs], setNodes[rhs]}});
      setNodes.push_back(drivenBy.size() - 1);
    }
    for (auto [port, set] : paths.outputs)
      drivenBy[getOrAddNode(instancePort(port))].second.push_back(
          setNodes[set]);

    // Note: Handling RWProbes.
    // 1. For RWProbes, output ports can be source of dataflow.
    // 2. All the RWProbes that refer to the same base value form a strongly
    // connected component. Each has a dataflow from the other, including
    // itself.
    // Steps to add the instance ports to the connectivity graph:
    // 1. Find the set of RWProbes that refer to the same base value.
    // 2. Add them to the same rwProbeClasses.
    // 3. Choose the first RWProbe port from this set as a representative base
    //    value. And add it as the source driver for every other RWProbe
    //    port in the set.
    // 4. This will ensure we can detect cycles involving different RWProbes to
    //    the same base value.
    for (auto &path : paths.forceable) {
      auto modSinkPortField = path.first;
      auto sinkArgNum =
          cast<BlockArgument>(modSinkPortField.getValue()).getArgNumber();
      FieldRef sinkPort(instResults[sinkArgNum], modSinkPortField.getFieldID());
      auto sinkNode = getOrAddNode(sinkPort);
      bool sinkPortIsForceable = false;
      if (auto refResultType =
              type_dyn_cast<RefType>(instResults[sinkArgNum].getType()))
        sinkPortIsForceable = refResultType.getForceable();

      DenseSet<unsigned> setOfEquivalentRWProbes;
      unsigned minArgNum = sinkArgNum;
      unsigned basePortNode = sinkNode;
      for (auto &modSrcPortField : path.second) {
        auto srcArgNum =
            cast<BlockArgument>(modSrcPortField.getValue()).getArgNumber();
        // Self loop will exist for RWProbes, ignore them.
        if (modSrcPortField == modSinkPortField)
          continue;

        FieldRef srcPort(instResults[srcArgNum], modSrcPortField.getFieldID());
        bool srcPortIsForceable = false;
        if (auto refResultType =
                type_dyn_cast<RefType>(instResults[srcArgNum].getType()))
          srcPortIsForceable = refResultType.getForceable();
        // RWProbes can potentially refer to the same base value. Such ports
        // have a path from each other, a false loop, detect such cases.
        if (sinkPortIsForceable && srcPortIsForceable) {
          auto srcNode = getOrAddNode(srcPort);
          // If a path is already recorded, continue.
          if (rwProbeClasses.findLeader(srcNode) !=
                  rwProbeClasses.member_end() &&
              rwProbeClasses.findLeader(sinkNode) ==
                  rwProbeClasses.findLeader(srcNode))
            continue;
          // Check if sinkPort is a driver of sourcePort.
          auto drivenBysToSrcPort = paths.forceable.find(modSrcPortField);
          if (drivenBysToSrcPort != paths.forceable.end())
            if (llvm::find(drivenBysToSrcPort->second, modSinkPortField) !=
                drivenBysToSrcPort->second.end()) {
              // This case can occur when there are multiple RWProbes on the
              // port, which refer to the same base value. So, each of such
              // probes are drivers of each other. Hence the false
              // loops. Instead of recording this in the drivenByGraph,
              // record it separately with the rwProbeClasses.
              setOfEquivalentRWProbes.insert(srcNode);
              if (minArgNum > srcArgNum) {
                // Make one of the RWProbe port the base node. Use the first
                // port for deterministic error messages.
                minArgNum = srcArgNum;
                basePortNode = srcNode;
              }
              continue;
            }
        }
        addDrivenBy(sinkPort, srcPort);
      }
      if (setOfEquivalentRWProbes.empty())
        continue;

      // Add all the rwprobes to the same class.
      for (auto probe : setOfEquivalentRWProbes)
        rwProbeClasses.unionSets(probe, sinkNode);

      // Make the first port as the base value.
      // Note: this is a port and the actual reference base exists in another
      // module.
      auto leader = rwProbeClasses.getLeaderValue(sinkNode);
      rwProbeRefersTo[leader] = basePortNode;

      setOfEquivalentRWProbes.insert(sinkNode);
      // Add the base RWProbe port as a driver to all other RWProbe ports.
      for (auto probe : setOfEquivalentRWProbes)
        if (probe != basePortNode)
          drivenBy[probe].second.push_back(basePortNode);
    }
  }

  // Check the referenced module paths and add input ports as the drivers for
  // the corresponding output port. The granularity of the connectivity
  // relations is per field.
  void handleInstanceOp(InstanceOp inst) {
    auto refMod = inst.getReferencedModule<FModuleOp>(instanceGraph);
    // Skip if the instance is not a module (e.g. external module).
    if (!refMod)
      return;
    processInstancePorts(refMod, inst.getResults());
  }

  // For InstanceChoiceOp, conservatively process all possible target modules.
  // Since we cannot determine which module will be selected at runtime, we
  // must consider combinational paths through all alternatives.
  void handleInstanceChoiceOp(InstanceChoiceOp inst) {
    // Process all referenced modules (default + alternatives)
    for (auto moduleName : inst.getReferencedModuleNamesAttr()) {
      auto moduleNameStr = cast<StringAttr>(moduleName);
      auto *node = instanceGraph.lookup(moduleNameStr);
      if (!node)
        continue;

      // Skip if the instance is not a module (e.g. external module).
      if (auto refMod = dyn_cast<FModuleOp>(*node->getModule()))
        processInstancePorts(refMod, inst.getResults());
    }
  }

  // Record the FieldRef, corresponding to the result of the sub op
  // `result = base[index]`
  void recordValueRefersToFieldRef(Value base, unsigned fieldID, Value result) {

    // Check if base is itself field of an aggregate.
    auto it = valToFieldRefs.find(base);
    if (it != valToFieldRefs.end()) {
      // Rebase it to the original aggregate.
      // Because of subaccess op, each value can refer to multiple FieldRefs.
      SmallVector<FieldRef> entry;
      for (auto &sub : it->second)
        entry.emplace_back(sub.getValue(), sub.getFieldID() + fieldID);
      // Update the map at the end, to avoid invaliding the iterator.
      valToFieldRefs[result].append(entry.begin(), entry.end());
      return;
    }
    // Break cycles from registers.
    if (auto *def = base.getDefiningOp()) {
      if (isa<RegOp, RegResetOp, SubfieldOp, SubindexOp, SubaccessOp>(def))
        return;
    }
    valToFieldRefs[result].emplace_back(base, fieldID);
  }

  /// Reports a cycle among the nodes left with `pending` drivers, each of
  /// which has a driver left: from the lowest such node, step to the lowest
  /// driver left until a node repeats.
  void reportCycle(ArrayRef<unsigned> pending) {
    DenseMap<unsigned, unsigned> step;
    SmallVector<unsigned> walk;
    unsigned node =
        llvm::find_if(pending, [](unsigned n) { return n; }) - pending.begin();
    while (step.try_emplace(node, walk.size()).second) {
      walk.push_back(node);
      std::optional<unsigned> next;
      for (auto driver : drivenBy[node].second)
        if (pending[driver] && (!next || driver < *next))
          next = driver;
      node = *next;
    }
    SmallVector<FieldRef, 16> path;
    for (auto n : ArrayRef(walk).drop_front(step[node]))
      if (drivenBy[n].first)
        path.push_back(drivenBy[n].first);
    reportLoopFound(path, path.front().getLoc());
  }

  void reportLoopFound(SmallVectorImpl<FieldRef> &path, Location loc) {
    auto errorDiag = mlir::emitError(
        module.getLoc(), "detected combinational cycle in a FIRRTL module");
    // Find a value we can name
    std::string firstName;
    FieldRef *it = llvm::find_if(path, [&](FieldRef v) {
      firstName = getName(v);
      return !firstName.empty();
    });
    if (it == path.end()) {
      errorDiag.append(", but unable to find names for any involved values.");
      errorDiag.attachNote(loc) << "cycle detected here";
      return;
    }
    // Begin the path from the "smallest string".
    for (circt::FieldRef *findStartIt = it; findStartIt != path.end();
         ++findStartIt) {
      auto n = getName(*findStartIt);
      if (!n.empty() && n < firstName) {
        firstName = n;
        it = findStartIt;
      }
    }
    errorDiag.append(", sample path: ");

    bool lastWasDots = false;
    errorDiag << module.getName() << ".{" << getName(*it);
    for (auto v : llvm::concat<FieldRef>(
             llvm::make_range(std::next(it), path.end()),
             llvm::make_range(path.begin(), std::next(it)))) {
      auto name = getName(v);
      if (!name.empty()) {
        errorDiag << " <- " << name;
        lastWasDots = false;
      } else {
        if (!lastWasDots)
          errorDiag << " <- ...";
        lastWasDots = true;
      }
    }
    errorDiag << "}";
  }

  void dumpMap() {
    LLVM_DEBUG({
      auto nodeName = [](FieldRef v) { return v ? getName(v) : "<hub>"; };
      llvm::dbgs() << "\n Connectivity Graph ==>";
      for (const auto &[index, i] : llvm::enumerate(drivenBy)) {
        llvm::dbgs() << "\n ===>dst:" << nodeName(i.first)
                     << "::" << i.first.getValue();
        for (auto s : i.second)
          llvm::dbgs() << "<---" << nodeName(drivenBy[s].first)
                       << "::" << drivenBy[s].first.getValue();
      }

      llvm::dbgs() << "\n Value to FieldRef :";
      for (const auto &fields : valToFieldRefs) {
        llvm::dbgs() << "\n Val:" << fields.first;
        for (auto f : fields.second)
          llvm::dbgs() << ", FieldRef:" << getName(f) << "::" << f.getFieldID();
      }
      llvm::dbgs() << "\n Port paths:";
      for (const auto &p : portPaths.forceable) {
        llvm::dbgs() << "\n Output :" << getName(p.first);
        for (auto i : p.second)
          llvm::dbgs() << "\n Input  :" << getName(i);
      }
      llvm::dbgs() << "\n rwprobes:";
      for (auto node : rwProbeRefersTo) {
        llvm::dbgs() << "\n node:" << getName(drivenBy[node.first].first)
                     << "=> probe:" << getName(drivenBy[node.second].first);
      }
      for (const auto &i :
           rwProbeClasses) { // Iterate over all of the equivalence sets.
        if (!i->isLeader())
          continue; // Ignore non-leader sets.
        // Print members in this set.
        llvm::interleave(rwProbeClasses.members(*i), llvm::dbgs(), "\n");
        llvm::dbgs() << "\n dataflow at leader::" << i->getData() << "\n =>"
                     << rwProbeRefersTo[i->getData()];
        llvm::dbgs() << "\n Done\n"; // Finish set.
      }
    });
  }

  void recordProbe(Value data, Value ref) {
    auto refNode = getOrAddNode(ref);
    auto dataNode = getOrAddNode(data);
    rwProbeRefersTo[rwProbeClasses.getOrInsertLeaderValue(refNode)] = dataNode;
  }

  // Add both the probes to the same equivalence class, to record that they
  // refer to the same data value.
  void probesReferToSameData(Value probe1, Value probe2) {
    auto p1Node = getOrAddNode(probe1);
    auto p2Node = getOrAddNode(probe2);
    rwProbeClasses.unionSets(p1Node, p2Node);
  }

  void addToPortPathsIfRWProbe(unsigned srcNode,
                               SmallSetVector<FieldRef, 4> &inputPortPaths) {
    // Check if there exists any RWProbe for the srcNode.
    auto baseFieldRef = drivenBy[srcNode].first;
    if (auto defOp = dyn_cast_or_null<Forceable>(baseFieldRef.getDefiningOp()))
      if (defOp.isForceable() && !defOp.getDataRef().use_empty()) {
        // Assumption, the probe must exist in the equivalence classes.
        auto rwProbeNode =
            rwProbeClasses.getLeaderValue(getOrAddNode(defOp.getDataRef()));
        // For all the probes, that are in the same eqv class, i.e., refer to
        // the same value.
        for (auto probe : rwProbeClasses.members(rwProbeNode)) {
          auto probeVal = drivenBy[probe].first;
          // If the probe is a port, then record the path from the probe to the
          // input port.
          if (isa<BlockArgument>(probeVal.getValue())) {
            inputPortPaths.insert(probeVal);
          }
        }
      }
  }

private:
  FModuleOp module;
  InstanceGraph &instanceGraph;

  /// Map of values to the set of all FieldRefs (same base) that this may be
  /// directly derived from through indexing operations.
  DenseMap<Value, SmallVector<FieldRef>> valToFieldRefs;
  /// Comb paths that exist between module ports. This is maintained across
  /// modules.
  const DenseMap<FModuleLike, PortPaths> &modulePortPaths;
  /// The comb paths between the ports of this module. This is the final
  /// output of this intra-procedural analysis, that is used to construct the
  /// inter-procedural dataflow.
  PortPaths &portPaths;

  /// This is an adjacency list representation of the connectivity graph. This
  /// can be indexed by the graph node id, and each entry is the list of graph
  /// nodes that has an edge to it. Each graph node represents a FieldRef and
  /// each edge represents a source that directly drives the sink node.
  DrivenByGraphType drivenBy;
  /// Map of FieldRef to its corresponding graph node.
  DenseMap<FieldRef, size_t> nodes;

  /// The base value that the RWProbe refers to. Used to add an edge to the base
  /// value, when the probe is forced.
  DenseMap<unsigned, unsigned> rwProbeRefersTo;

  /// An eqv class of all the RWProbes that refer to the same base value.
  llvm::EquivalenceClasses<unsigned> rwProbeClasses;
};

/// This pass constructs a local graph for each module to detect
/// combinational cycles. To capture the cross-module combinational cycles,
/// this pass inlines the combinational paths between IOs of its
/// subinstances into a subgraph and encodes them in `modulePortPaths`.
class CheckCombLoopsPass
    : public circt::firrtl::impl::CheckCombLoopsBase<CheckCombLoopsPass> {
public:
  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    DenseMap<FModuleLike, PortPaths> modulePortPaths;

    // Traverse modules in a post order to make sure the combinational paths
    // between IOs of a module have been detected and recorded in
    // `modulePortPaths` before we handle its parent modules.
    for (auto *igNode : llvm::post_order<InstanceGraph *>(&instanceGraph)) {
      if (auto module = dyn_cast<FModuleOp>(*igNode->getModule())) {
        DiscoverLoops rdf(module, instanceGraph, modulePortPaths,
                          modulePortPaths[module]);
        if (rdf.processModule().failed()) {
          return signalPassFailure();
        }
      }
    }
    markAllAnalysesPreserved();
  }
};
