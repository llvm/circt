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

  auto tokenFlag = false;
  if (isa<dc::UnpackOp>(op))
    tokenFlag = true;

  DotNode n = {op.getName().stripDialect().str(),
               valueToName(op.getOperands(), valuesMap, false),
               valueToName(op.getOperands(), valuesMap, tokenFlag)};

  return n;
}

DotNode createCombNode(
    Operation &op,
    llvm::SmallVector<std::pair<mlir::Value, std::string>> &valuesMap) {

  DotNode n = {"", valueToName(op.getOperands(), valuesMap, false),
               valueToName(op.getOperands(), valuesMap, false)};

  TypeSwitch<mlir::Operation *>(&op)
      .Case<comb::AddOp>([&](auto op) { n.nodeType = "+"; })
      .Case<comb::AndOp>([&](auto op) { n.nodeType = "&&"; })
      .Case<comb::XorOp>([&](auto op) { n.nodeType = "^"; })
      .Case<comb::OrOp>([&](auto op) { n.nodeType = "||"; })
      .Case<comb::MulOp>([&](auto op) { n.nodeType = "x"; })
      .Case<comb::MuxOp>([&](auto op) { n.nodeType = "mux"; })
      .Case<comb::ICmpOp>([&](auto op) {
        switch (op.getPredicate()) {
        case circt::comb::ICmpPredicate::eq: {
          n.nodeType = "==";
          break;
        }
        case circt::comb::ICmpPredicate::ne: {
          n.nodeType = "!=";
          break;
        }
        case circt::comb::ICmpPredicate::sgt: {
          n.nodeType = ">";
          break;
        }
        case circt::comb::ICmpPredicate::sge: {
          n.nodeType = ">=";
          break;
        }
        case circt::comb::ICmpPredicate::slt: {
          n.nodeType = "<";
          break;
        }
        case circt::comb::ICmpPredicate::sle: {
          n.nodeType = "<=";
          break;
        }
        case circt::comb::ICmpPredicate::ugt: {
          n.nodeType = ">";
          break;
        }
        case circt::comb::ICmpPredicate::uge: {
          n.nodeType = ">=";
          break;
        }
        case circt::comb::ICmpPredicate::ult: {
          n.nodeType = "<=";
          break;
        }
        case circt::comb::ICmpPredicate::ule: {
          n.nodeType = "<";
          break;
        }
        case circt::comb::ICmpPredicate::ceq: {
          n.nodeType = "==";
          break;
        }
        case circt::comb::ICmpPredicate::cne: {
          n.nodeType = "!=";
          break;
        }
        case circt::comb::ICmpPredicate::weq: {
          n.nodeType = "==";
          break;
        }
        case circt::comb::ICmpPredicate::wne: {
          n.nodeType = "!=";
          break;
        }
        }
      });
  return n;
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