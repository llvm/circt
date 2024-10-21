//===- DCPrintDot.cpp - Analysis Pass -----------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the DCPrintDot pass.
//
//===----------------------------------------------------------------------===//
#include "circt/Dialect/DC/DCOps.h"
#include "circt/Dialect/DC/DCPasses.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/DC/DCOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace dc {
#define GEN_PASS_DEF_DCDOTPRINT
#include "circt/Dialect/DC/DCPasses.h.inc"
} // namespace dc
} // namespace circt

using namespace circt;
using namespace dc;
using namespace mlir;

using ValueMap = llvm::ScopedHashTable<mlir::Value, std::string>;

namespace {

/// all Comb and DC nodetypes
enum class NodeType {
  AddOp,
  AndOp,
  XorOp,
  MulOp,
  OrOp,
  MuxOp,
  EqOp,
  NeOp,
  GtOp,
  GeOp,
  LtOp,
  LeOp,
  BranchOp,
  BufferOp,
  ForkOp,
  FromESIOp,
  JoinOp,
  MergeOp,
  PackOp,
  SelectOp,
  SinkOp,
  SourceOp,
  ToESIOp,
  UnpackOp,
  Null
};

std::string stringify(NodeType type) {
  switch (type) {
  case NodeType::AddOp:
    return "+";
  case NodeType::AndOp:
    return "&&";
  case NodeType::XorOp:
    return "^";
  case NodeType::OrOp:
    return "||";
  case NodeType::MulOp:
    return "x";
  case NodeType::MuxOp:
    return "mux";
  case NodeType::EqOp:
    return "==";
  case NodeType::NeOp:
    return "!=";
  case NodeType::GtOp:
    return ">";
  case NodeType::GeOp:
    return ">=";
  case NodeType::LtOp:
    return "<";
  case NodeType::LeOp:
    return "<=";
  case NodeType::BranchOp:
    return "branch";
  case NodeType::BufferOp:
    return "buffer";
  case NodeType::ForkOp:
    return "fork";
  case NodeType::FromESIOp:
    return "fromESI";
  case NodeType::JoinOp:
    return "join";
  case NodeType::MergeOp:
    return "merge";
  case NodeType::PackOp:
    return "pack";
  case NodeType::SelectOp:
    return "select";
  case NodeType::SinkOp:
    return "sink";
  case NodeType::SourceOp:
    return "source";
  case NodeType::ToESIOp:
    return "toESI";
  case NodeType::UnpackOp:
    return "unpack";
  case NodeType::Null:
    return "null";
  }
}

/// stores operand and results for each node in the dot graph
struct DotNode {
  NodeType nodeType;
  SmallVector<std::pair<mlir::Value, std::string>> incoming;
  SmallVector<std::pair<mlir::Value, std::string>> outgoing;
};
} // namespace

/// gives a unique name to each value in the graph
static SmallVector<std::pair<mlir::Value, std::string>>
valueToName(const SmallVector<mlir::Value> &values,
            SmallVector<std::pair<mlir::Value, std::string>> &currentMap,
            bool tokenFlag) {
  SmallVector<std::pair<mlir::Value, std::string>> res;
  for (auto [i, v] : llvm::enumerate(values)) {
    auto found = false;
    for (const auto &[key, value] : currentMap) {
      if (v == key) {
        res.push_back({v, value});
        found = true;
      }
    }
    if (!found) {
      std::string name = "in_" + std::to_string(currentMap.size());
      if (tokenFlag && i % 2 == 0)
        name = "token_" + std::to_string(currentMap.size());
      res.push_back({v, name});
      currentMap.push_back({v, name});
    }
  }
  return res;
}

/// creates node in the dataflow graph for DC operations
static DotNode
createDCNode(Operation &op,
             SmallVector<std::pair<mlir::Value, std::string>> &valuesMap) {

  auto tokenFlag = false;
  if (isa<dc::UnpackOp>(op))
    tokenFlag = true;

  DotNode n = {NodeType::Null, valueToName(op.getOperands(), valuesMap, false),
               valueToName(op.getOperands(), valuesMap, tokenFlag)};

  TypeSwitch<mlir::Operation *>(&op)
      .Case([&](dc::BranchOp) { n.nodeType = NodeType::BranchOp; })
      .Case([&](dc::BufferOp) { n.nodeType = NodeType::BufferOp; })
      .Case([&](dc::ForkOp) { n.nodeType = NodeType::ForkOp; })
      .Case([&](dc::FromESIOp) { n.nodeType = NodeType::FromESIOp; })
      .Case([&](dc::JoinOp) { n.nodeType = NodeType::JoinOp; })
      .Case([&](dc::MergeOp) { n.nodeType = NodeType::MergeOp; })
      .Case([&](dc::PackOp) { n.nodeType = NodeType::PackOp; })
      .Case([&](dc::SelectOp) { n.nodeType = NodeType::SelectOp; })
      .Case([&](dc::SinkOp) { n.nodeType = NodeType::SinkOp; })
      .Case([&](dc::SourceOp) { n.nodeType = NodeType::SourceOp; })
      .Case([&](dc::ToESIOp) { n.nodeType = NodeType::ToESIOp; })
      .Case([&](dc::UnpackOp) { n.nodeType = NodeType::UnpackOp; });

  return n;
}

/// creates node in the dataflow graph for Comb operations
static DotNode
createCombNode(Operation &op,
               SmallVector<std::pair<mlir::Value, std::string>> &valuesMap) {

  DotNode n = {NodeType::Null, valueToName(op.getOperands(), valuesMap, false),
               valueToName(op.getOperands(), valuesMap, false)};

  TypeSwitch<mlir::Operation *>(&op)
      .Case([&](comb::AddOp) { n.nodeType = NodeType::AddOp; })
      .Case([&](comb::AndOp) { n.nodeType = NodeType::AndOp; })
      .Case([&](comb::XorOp) { n.nodeType = NodeType::XorOp; })
      .Case([&](comb::OrOp) { n.nodeType = NodeType::OrOp; })
      .Case([&](comb::MulOp) { n.nodeType = NodeType::MulOp; })
      .Case([&](comb::MuxOp) { n.nodeType = NodeType::MuxOp; })
      .Case<comb::ICmpOp>([&](auto op) {
        switch (op.getPredicate()) {
        case circt::comb::ICmpPredicate::eq: {
          n.nodeType = NodeType::EqOp;
          break;
        }
        case circt::comb::ICmpPredicate::ne: {
          n.nodeType = NodeType::NeOp;
          break;
        }
        case circt::comb::ICmpPredicate::sgt: {
          n.nodeType = NodeType::GtOp;
          break;
        }
        case circt::comb::ICmpPredicate::sge: {
          n.nodeType = NodeType::GeOp;
          break;
        }
        case circt::comb::ICmpPredicate::slt: {
          n.nodeType = NodeType::LtOp;
          break;
        }
        case circt::comb::ICmpPredicate::sle: {
          n.nodeType = NodeType::LeOp;
          break;
        }
        case circt::comb::ICmpPredicate::ugt: {
          n.nodeType = NodeType::GtOp;
          break;
        }
        case circt::comb::ICmpPredicate::uge: {
          n.nodeType = NodeType::GeOp;
          break;
        }
        case circt::comb::ICmpPredicate::ult: {
          n.nodeType = NodeType::LeOp;
          break;
        }
        case circt::comb::ICmpPredicate::ule: {
          n.nodeType = NodeType::LtOp;
          break;
        }
        case circt::comb::ICmpPredicate::ceq: {
          n.nodeType = NodeType::EqOp;
          break;
        }
        case circt::comb::ICmpPredicate::cne: {
          n.nodeType = NodeType::NeOp;
          break;
        }
        case circt::comb::ICmpPredicate::weq: {
          n.nodeType = NodeType::EqOp;
          break;
        }
        case circt::comb::ICmpPredicate::wne: {
          n.nodeType = NodeType::NeOp;
          break;
        }
        }
      });
  return n;
}

namespace {
/// Emit the dot nodes
struct DCDotPrintPass : public circt::dc::impl::DCDotPrintBase<DCDotPrintPass> {
  explicit DCDotPrintPass(raw_ostream &os) : os(os) {}
  void runOnOperation() override {

    ModuleOp op = getOperation();

    std::error_code ec;

    SmallVector<std::pair<mlir::Value, std::string>> valuesMap;
    SmallVector<DotNode> nodes;

    auto &moduleOps = op->getRegion(0).getBlocks();
    for (auto &moduleOp : moduleOps) {
      auto hwModuleOp = moduleOp.getOps<hw::HWModuleOp>();
      for (auto hmo : hwModuleOp) {
        for (auto &op : hmo->getRegion(0).getOps()) {
          // either dc or comb operation
          if (op.getDialect()->getNamespace() == "comb") {
            // create new node
            auto node = createCombNode(op, valuesMap);
            nodes.push_back(node);
          } else if (op.getDialect()->getNamespace() == "dc") {
            auto node = createDCNode(op, valuesMap);
            nodes.push_back(node);
          }
        }
      }
    }
    os << "digraph{\n";
    // print all nodes first
    for (auto [i, n] : llvm::enumerate(nodes)) {
      os << i << " [shape = polygon, label = \"" << stringify(n.nodeType)
         << "\"]\n";
    }
    for (auto [id, n] : llvm::enumerate(nodes)) {
      if (n.nodeType == NodeType::UnpackOp)
        for (auto ic : n.incoming)
          os << "token_" << ic.second << " -> " << id << R"( [label = ")"
             << ic.second << "\"]\n";
    }
    for (auto [id1, n1] : llvm::enumerate(nodes)) {
      for (auto [id2, n2] : llvm::enumerate(nodes)) {
        if (id1 != id2) {
          for (const auto &n1Out : n1.outgoing) {
            for (const auto &[i, n2In] : llvm::enumerate(n2.incoming)) {
              if (n1Out.first == n2In.first) {
                os << id1 << " -> " << id2 << " [label = \"" << n1Out.second
                   << "\"]\n";
              }
            }
          }
        }
      }
    }
    for (auto [id, n] : llvm::enumerate(nodes)) {
      if (n.nodeType == NodeType::PackOp)
        for (const auto &ic : n.outgoing)
          os << id << " -> " << ic.second << " [label = \"" << ic.second
             << "\"]\n";
    }

    os << "}\n";
  }
  raw_ostream &os;
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
circt::dc::createDCDotPrintPass() {
  return std::make_unique<DCDotPrintPass>(llvm::errs());
}
