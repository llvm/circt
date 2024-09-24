//===- Analysis.cpp - Analysis Pass -----------------------------*- C++ -*-===//
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
#include "circt/Dialect/HW/HWDialect.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/DC/DCOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

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

// stores operand and results for each node in the dot graph
struct DotNode {
  std::string nodeType;
  llvm::SmallVector<std::pair<mlir::Value, std::string>> incoming;
  llvm::SmallVector<std::pair<mlir::Value, std::string>> outgoing;
};

// gives unique name to each value in the graph
llvm::SmallVector<std::pair<mlir::Value, std::string>>
valueToName(const llvm::SmallVector<mlir::Value> &values,
            llvm::SmallVector<std::pair<mlir::Value, std::string>> &currentMap,
            bool tokenFlag) {
  llvm::SmallVector<std::pair<mlir::Value, std::string>> res;
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

DotNode createDCNode(
    Operation &op,
    llvm::SmallVector<std::pair<mlir::Value, std::string>> &valuesMap) {
  if (auto castOp = llvm::dyn_cast<dc::ForkOp>(op))
    return DotNode{.nodeType = "fork",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto castOp = llvm::dyn_cast<dc::JoinOp>(op))
    return DotNode{.nodeType = "join",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto castOp = llvm::dyn_cast<dc::BranchOp>(op))
    return DotNode{.nodeType = "branch",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto castOp = llvm::dyn_cast<dc::MergeOp>(op))
    return DotNode{.nodeType = "merge",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto castOp = llvm::dyn_cast<dc::SourceOp>(op))
    return DotNode{.nodeType = "source",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto castOp = llvm::dyn_cast<dc::SinkOp>(op))
    return DotNode{.nodeType = "sink",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto castOp = llvm::dyn_cast<dc::SelectOp>(op))
    return DotNode{.nodeType = "select",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto castOp = llvm::dyn_cast<dc::BufferOp>(op))
    return DotNode{.nodeType = "buffer",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto castOp = llvm::dyn_cast<dc::FromESIOp>(op))
    return DotNode{.nodeType = "fromESI",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto castOp = llvm::dyn_cast<dc::PackOp>(op))
    return DotNode{.nodeType = "pack",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto castOp = llvm::dyn_cast<dc::ToESIOp>(op))
    return DotNode{.nodeType = "toESI",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto castOp = llvm::dyn_cast<dc::UnpackOp>(op))
    return DotNode{.nodeType = "unpack",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, true)};
}

DotNode createCombNode(
    Operation &op,
    llvm::SmallVector<std::pair<mlir::Value, std::string>> &valuesMap) {
  if (auto addOp = llvm::dyn_cast<comb::AddOp>(op))
    return DotNode{.nodeType = "+",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto andOp = llvm::dyn_cast<comb::AndOp>(op))
    return DotNode{.nodeType = "&&",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto xorOp = llvm::dyn_cast<comb::XorOp>(op))
    return DotNode{.nodeType = "^",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto orOp = llvm::dyn_cast<comb::OrOp>(op))
    return DotNode{.nodeType = "||",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto mulOp = llvm::dyn_cast<comb::MulOp>(op))
    return DotNode{.nodeType = "x",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto muxOp = llvm::dyn_cast<comb::MuxOp>(op))
    return DotNode{.nodeType = "mux",
                   .incoming = valueToName(op.getOperands(), valuesMap, false),
                   .outgoing = valueToName(op.getResults(), valuesMap, false)};
  if (auto icmp = llvm::dyn_cast<comb::ICmpOp>(op)) {
    if (icmp.getPredicate() == circt::comb::ICmpPredicate::eq)
      return DotNode{
          .nodeType = "==",
          .incoming = valueToName(op.getOperands(), valuesMap, false),
          .outgoing = valueToName(op.getResults(), valuesMap, false)};
    if (icmp.getPredicate() == circt::comb::ICmpPredicate::ne)
      return DotNode{
          .nodeType = "!=",
          .incoming = valueToName(op.getOperands(), valuesMap, false),
          .outgoing = valueToName(op.getResults(), valuesMap, false)};
    if (icmp.getPredicate() == circt::comb::ICmpPredicate::sgt)
      return DotNode{
          .nodeType = ">",
          .incoming = valueToName(op.getOperands(), valuesMap, false),
          .outgoing = valueToName(op.getResults(), valuesMap, false)};
    if (icmp.getPredicate() == circt::comb::ICmpPredicate::sge)
      return DotNode{
          .nodeType = ">=",
          .incoming = valueToName(op.getOperands(), valuesMap, false),
          .outgoing = valueToName(op.getResults(), valuesMap, false)};
    if (icmp.getPredicate() == circt::comb::ICmpPredicate::slt)
      return DotNode{
          .nodeType = "<",
          .incoming = valueToName(op.getOperands(), valuesMap, false),
          .outgoing = valueToName(op.getResults(), valuesMap, false)};
    if (icmp.getPredicate() == circt::comb::ICmpPredicate::sle)
      return DotNode{
          .nodeType = "<=",
          .incoming = valueToName(op.getOperands(), valuesMap, false),
          .outgoing = valueToName(op.getResults(), valuesMap, false)};
    if (icmp.getPredicate() == circt::comb::ICmpPredicate::ugt)
      return DotNode{
          .nodeType = ">",
          .incoming = valueToName(op.getOperands(), valuesMap, false),
          .outgoing = valueToName(op.getResults(), valuesMap, false)};
    if (icmp.getPredicate() == circt::comb::ICmpPredicate::uge)
      return DotNode{
          .nodeType = ">=",
          .incoming = valueToName(op.getOperands(), valuesMap, false),
          .outgoing = valueToName(op.getResults(), valuesMap, false)};
    if (icmp.getPredicate() == circt::comb::ICmpPredicate::ult)
      return DotNode{
          .nodeType = "<",
          .incoming = valueToName(op.getOperands(), valuesMap, false),
          .outgoing = valueToName(op.getResults(), valuesMap, false)};
    if (icmp.getPredicate() == circt::comb::ICmpPredicate::ule)
      return DotNode{
          .nodeType = "<=",
          .incoming = valueToName(op.getOperands(), valuesMap, false),
          .outgoing = valueToName(op.getResults(), valuesMap, false)};
  }
  return DotNode{.nodeType = "combNull",
                 .incoming = valueToName(op.getOperands(), valuesMap, false),
                 .outgoing = valueToName(op.getResults(), valuesMap, false)};
}

namespace {
/// Emit the dot nodes
struct DCDotPrintPass : public circt::dc::impl::DCDotPrintBase<DCDotPrintPass> {
  void runOnOperation() override {

    ModuleOp op = getOperation();

    std::error_code ec;
    llvm::raw_fd_ostream outfile("dc-out.dot", ec);
    mlir::raw_indented_ostream os(outfile);

    llvm::SmallVector<std::pair<mlir::Value, std::string>> valuesMap;
    llvm::SmallVector<DotNode> nodes;

    auto &moduleOps = op->getRegion(0).getBlocks();
    // getBlock()->getOps<hw::HWModuleOp>();
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
      os << i << " [shape = polygon, label = \"" << n.nodeType << "\"]\n";
    }
    for (auto [id, n] : llvm::enumerate(nodes)) {
      if (n.nodeType == "unpack")
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
      if (n.nodeType == "pack")
        for (const auto &ic : n.outgoing)
          os << id << " -> " << ic.second << " [label = \"" << ic.second
             << "\"]\n";
    }

    os << "}";
    outfile.close();
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
circt::dc::createDCDotPrintPass() {
  return std::make_unique<DCDotPrintPass>();
}