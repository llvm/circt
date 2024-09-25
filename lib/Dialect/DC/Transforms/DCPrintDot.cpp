//===- DCPrintDot.cpp - Analysis Pass -----------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the Analysis pass.
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
enum NodeType {
  addOp,
  andOp,
  xorOp,
  orOp,
  mulOp,
  muxOp,
  eqOp,
  neOp,
  gtOp,
  geOp,
  ltOp,
  leOp,
  branchOp,
  bufferOp,
  forkOp,
  fromESIOp,
  joinOp,
  mergeOp,
  packOp,
  selectOp,
  sinkOp,
  sourceOp,
  toESIOp,
  unpackOp,
  null
};

std::string stringify(NodeType type) {
  switch (type) {
  case addOp:
    return "+";
  case andOp:
    return "&&";
  case xorOp:
    return "^";
  case orOp:
    return "||";
  case mulOp:
    return "x";
  case muxOp:
    return "mux";
  case eqOp:
    return "==";
  case neOp:
    return "!=";
  case gtOp:
    return ">";
  case geOp:
    return ">=";
  case ltOp:
    return "<";
  case leOp:
    return "<=";
  case branchOp:
    return "branch";
  case bufferOp:
    return "buffer";
  case forkOp:
    return "fork";
  case fromESIOp:
    return "fromESI";
  case joinOp:
    return "join";
  case mergeOp:
    return "merge";
  case packOp:
    return "pack";
  case selectOp:
    return "select";
  case sinkOp:
    return "sink";
  case sourceOp:
    return "source";
  case toESIOp:
    return "toESI";
  case unpackOp:
    return "unpack";
  case null:
    return "null";
  default:
    return "Unknown NodeType";
  }
}

/// stores operand and results for each node in the dot graph
struct DotNode {
  NodeType nodeType;
  llvm::SmallVector<std::pair<mlir::Value, std::string>> incoming;
  llvm::SmallVector<std::pair<mlir::Value, std::string>> outgoing;
};
} // namespace

/// gives a unique name to each value in the graph
llvm::SmallVector<std::pair<mlir::Value, std::string>> static valueToName(
    const llvm::SmallVector<mlir::Value> &values,
    llvm::SmallVector<std::pair<mlir::Value, std::string>> &currentMap,
    bool tokenFlag) {
  SmallVector<std::pair<mlir::Value, std::string>> res;
  for (auto [i, v] : llvm::enumerate(values)) {
    auto found = false;
    for (const auto &cm : currentMap) {
      if (v == cm.first) {
        res.push_back({v, cm.second});
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
DotNode createDCNode(
    Operation &op,
    llvm::SmallVector<std::pair<mlir::Value, std::string>> &valuesMap) {

  auto tokenFlag = false;
  if (isa<dc::UnpackOp>(op))
    tokenFlag = true;

  DotNode n = {null, valueToName(op.getOperands(), valuesMap, false),
               valueToName(op.getOperands(), valuesMap, tokenFlag)};

  TypeSwitch<mlir::Operation *>(&op)
      .Case([&](dc::BranchOp) { n.nodeType = branchOp; })
      .Case([&](dc::BufferOp) { n.nodeType = bufferOp; })
      .Case([&](dc::ForkOp) { n.nodeType = forkOp; })
      .Case([&](dc::FromESIOp) { n.nodeType = fromESIOp; })
      .Case([&](dc::JoinOp) { n.nodeType = joinOp; })
      .Case([&](dc::MergeOp) { n.nodeType = mergeOp; })
      .Case([&](dc::PackOp) { n.nodeType = packOp; })
      .Case([&](dc::SelectOp) { n.nodeType = selectOp; })
      .Case([&](dc::SinkOp) { n.nodeType = sinkOp; })
      .Case([&](dc::SourceOp) { n.nodeType = sourceOp; })
      .Case([&](dc::ToESIOp) { n.nodeType = toESIOp; })
      .Case([&](dc::UnpackOp) { n.nodeType = unpackOp; });

  return n;
}

/// creates node in the dataflow graph for Comb operations
DotNode createCombNode(
    Operation &op,
    llvm::SmallVector<std::pair<mlir::Value, std::string>> &valuesMap) {

  DotNode n = {null, valueToName(op.getOperands(), valuesMap, false),
               valueToName(op.getOperands(), valuesMap, false)};

  TypeSwitch<mlir::Operation *>(&op)
      .Case([&](comb::AddOp) { n.nodeType = addOp; })
      .Case([&](comb::AndOp) { n.nodeType = andOp; })
      .Case([&](comb::XorOp) { n.nodeType = xorOp; })
      .Case([&](comb::OrOp) { n.nodeType = orOp; })
      .Case([&](comb::MulOp) { n.nodeType = mulOp; })
      .Case([&](comb::MuxOp) { n.nodeType = muxOp; })
      .Case<comb::ICmpOp>([&](auto op) {
        switch (op.getPredicate()) {
        case circt::comb::ICmpPredicate::eq: {
          n.nodeType = eqOp;
          break;
        }
        case circt::comb::ICmpPredicate::ne: {
          n.nodeType = neOp;
          break;
        }
        case circt::comb::ICmpPredicate::sgt: {
          n.nodeType = gtOp;
          break;
        }
        case circt::comb::ICmpPredicate::sge: {
          n.nodeType = geOp;
          break;
        }
        case circt::comb::ICmpPredicate::slt: {
          n.nodeType = ltOp;
          break;
        }
        case circt::comb::ICmpPredicate::sle: {
          n.nodeType = leOp;
          break;
        }
        case circt::comb::ICmpPredicate::ugt: {
          n.nodeType = gtOp;
          break;
        }
        case circt::comb::ICmpPredicate::uge: {
          n.nodeType = geOp;
          break;
        }
        case circt::comb::ICmpPredicate::ult: {
          n.nodeType = leOp;
          break;
        }
        case circt::comb::ICmpPredicate::ule: {
          n.nodeType = ltOp;
          break;
        }
        case circt::comb::ICmpPredicate::ceq: {
          n.nodeType = eqOp;
          break;
        }
        case circt::comb::ICmpPredicate::cne: {
          n.nodeType = neOp;
          break;
        }
        case circt::comb::ICmpPredicate::weq: {
          n.nodeType = eqOp;
          break;
        }
        case circt::comb::ICmpPredicate::wne: {
          n.nodeType = neOp;
          break;
        }
        }
      });
  return n;
}

namespace {
/// Emit the dot nodes
struct DCDotPrintPass : public circt::dc::impl::DCDotPrintBase<DCDotPrintPass> {
  DCDotPrintPass(llvm::raw_ostream &os) : os(os) {}
  void runOnOperation() override {

    ModuleOp op = getOperation();

    std::error_code ec;

    llvm::SmallVector<std::pair<mlir::Value, std::string>> valuesMap;
    llvm::SmallVector<DotNode> nodes;

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
      if (n.nodeType == unpackOp)
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
      if (n.nodeType == packOp)
        for (const auto &ic : n.outgoing)
          os << id << " -> " << ic.second << " [label = \"" << ic.second
             << "\"]\n";
    }

    os << "}\n";
  }
  llvm::raw_ostream &os;
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
circt::dc::createDCDotPrintPass() {
  return std::make_unique<DCDotPrintPass>(llvm::errs());
}
