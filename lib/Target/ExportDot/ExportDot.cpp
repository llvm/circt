//===- ExportDot.cpp - SMT-LIB Emitter -----=---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Dot emitter implementation.
//
//===----------------------------------------------------------------------===//
#include "circt/Target/ExportDot.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/DC/DCOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace circt;
using namespace dc;

using namespace ExportDot;

using ValueMap = llvm::ScopedHashTable<mlir::Value, std::string>;

#define DEBUG_TYPE "export-dot"


//===----------------------------------------------------------------------===//
// Unified Emitter implementation
//===----------------------------------------------------------------------===//




namespace {

/// Contains the informations passed to the ExpressionVisitor methods. Makes it
/// easier to add more information.
struct DotNode {
  std::string nodeType;
  llvm::SmallVector<std::pair<mlir::Value, std::string>> incoming;
  llvm::SmallVector<std::pair<mlir::Value, std::string>> outgoing;
};

llvm::SmallVector<std::pair<mlir::Value, std::string>> valueToName(const llvm::SmallVector<mlir::Value>& values, llvm::SmallVector<std::pair<mlir::Value, std::string>> &currentMap, bool tokenFlag, bool inputFlag){
  llvm::SmallVector<std::pair<mlir::Value, std::string>> res;
  for (auto [i, v] : llvm::enumerate(values)){
    auto found = false;
    for (const auto& cm : currentMap){
      if (v == cm.first){
        res.push_back({v, cm.second});
        found = true;
      }
    }
    if (!found){
      std::string name = "data_"+std::to_string(currentMap.size());
      if (tokenFlag && i%2==0)
        name = "token_"+std::to_string(currentMap.size());
      else if (inputFlag)
        name = "input_"+std::to_string(currentMap.size());
      res.push_back({v, name});
      currentMap.push_back({v, name});
    }
  }
  return res;
}

DotNode createDCNode(Operation &op, llvm::SmallVector<std::pair<mlir::Value, std::string>> &valuesMap){
  if (auto castOp = llvm::dyn_cast<dc::ForkOp>(op))
    return DotNode{.nodeType="fork", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
  if (auto castOp = llvm::dyn_cast<dc::JoinOp>(op))
    return DotNode{.nodeType="join", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
  if (auto castOp = llvm::dyn_cast<dc::BranchOp>(op))
    return DotNode{.nodeType="branch", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
  if (auto castOp = llvm::dyn_cast<dc::MergeOp>(op))
    return DotNode{.nodeType="merge", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
  if (auto castOp = llvm::dyn_cast<dc::SourceOp>(op))
    return DotNode{.nodeType="source", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
  if (auto castOp = llvm::dyn_cast<dc::SinkOp>(op))
    return DotNode{.nodeType="sink", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
  if (auto castOp = llvm::dyn_cast<dc::SelectOp>(op))
    return DotNode{.nodeType="select", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
  return DotNode{.nodeType="nullDC", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
}

DotNode createCombNode(Operation &op, llvm::SmallVector<std::pair<mlir::Value, std::string>> &valuesMap){
  if (auto addOp = llvm::dyn_cast<comb::AddOp>(op))
    return DotNode{.nodeType="combAdd", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
  if (auto andOp = llvm::dyn_cast<comb::AndOp>(op))
    return DotNode{.nodeType="combAnd", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
  if (auto xorOp = llvm::dyn_cast<comb::XorOp>(op))
    return DotNode{.nodeType="combXor", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
  if (auto orOp = llvm::dyn_cast<comb::OrOp>(op))
    return DotNode{.nodeType="combOr", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
  if (auto mulOp = llvm::dyn_cast<comb::MulOp>(op))
    return DotNode{.nodeType="combMult", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
  if (auto muxOp = llvm::dyn_cast<comb::MuxOp>(op))
    return DotNode{.nodeType="combMux", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
  if (auto icmp = llvm::dyn_cast<comb::ICmpOp>(op)){
    if(icmp.getPredicate() == circt::comb::ICmpPredicate::eq)
      return DotNode{.nodeType="combEq", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
    if(icmp.getPredicate() == circt::comb::ICmpPredicate::ne)
      return DotNode{.nodeType="combNe", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
    if(icmp.getPredicate() == circt::comb::ICmpPredicate::sgt)
      return DotNode{.nodeType="combSgt", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
    if(icmp.getPredicate() == circt::comb::ICmpPredicate::sge)
      return DotNode{.nodeType="combSge", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
    if(icmp.getPredicate() == circt::comb::ICmpPredicate::slt)
      return DotNode{.nodeType="combSlt", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
    if(icmp.getPredicate() == circt::comb::ICmpPredicate::sle)
      return DotNode{.nodeType="combSle", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
    if(icmp.getPredicate() == circt::comb::ICmpPredicate::ugt)
      return DotNode{.nodeType="combUgt", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
    if(icmp.getPredicate() == circt::comb::ICmpPredicate::uge)
      return DotNode{.nodeType="combUge", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
    if(icmp.getPredicate() == circt::comb::ICmpPredicate::ult)
      return DotNode{.nodeType="combUlt", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
    if(icmp.getPredicate() == circt::comb::ICmpPredicate::ule)
      return DotNode{.nodeType="combUle", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
  }
  return DotNode{.nodeType="combNull", .incoming=valueToName(op.getOperands(), valuesMap, false, false), .outgoing=valueToName(op.getResults(), valuesMap, false, false)};
}

/// A visitor to print SMT dialect operations with zero result values or
/// ones that have to initialize some global state.

} // namespace



/// Emit the SMT operations in the given 'solver' to the 'stream'.
static LogicalResult emitDot(Operation *module,
                          mlir::raw_indented_ostream &stream) {

  llvm::SmallVector<std::pair<mlir::Value, std::string>> valuesMap;
  llvm::SmallVector<DotNode> nodes;

  auto &moduleOps = module->getRegion(0).getBlocks();
  // getBlock()->getOps<hw::HWModuleOp>();
  for (auto &moduleOp: moduleOps){
    auto hwModuleOp = moduleOp.getOps<hw::HWModuleOp>();
    for(auto hmo : hwModuleOp){
      for (auto &op: hmo->getRegion(0).getOps())
        if(auto unpackOp = llvm::dyn_cast<dc::UnpackOp>(op)){
          DotNode node = {.nodeType = "unpack", .incoming =valueToName(unpackOp->getOperands(), valuesMap, false, true), .outgoing = valueToName(unpackOp->getResults(), valuesMap, true, false)};
          nodes.push_back(node);
        } else if (auto packOp = llvm::dyn_cast<dc::PackOp>(op)) {
          DotNode node = {.nodeType="pack", .incoming=valueToName(packOp.getOperands(), valuesMap, false, false), .outgoing = valueToName(packOp->getResults(), valuesMap, false, false)};
          nodes.push_back(node);
        } else {
          // either dc or comb operation
          if (op.getDialect()->getNamespace() == "comb"){
            // create new node
            auto node = createCombNode(op, valuesMap);
            nodes.push_back(node);
          } else if (op.getDialect()->getNamespace() == "dc"){
            auto node = createDCNode(op, valuesMap);
            nodes.push_back(node);
          }
        }
    }
  }

  
  stream << "digraph{\n";
  // print all nodes first 
  for (auto [i, n] : llvm::enumerate(nodes)){
    stream<<n.nodeType<<i<<" [shape = polygon, mod = \""<<n.nodeType<<"\"]\n";
  }
  for(auto [id, n] : llvm::enumerate(nodes)){
    if (n.nodeType == "unpack")
      for (auto ic: n.incoming)
        stream << ic.second<<" -> "<<n.nodeType<<id<<" [inp = \""<<ic.second<<"\", out = out, label = \""<<ic.second<<"\"]\n";
  }
  for(auto [id1, n1] : llvm::enumerate(nodes)){
    for(auto [id2, n2] : llvm::enumerate(nodes)){
      if(id1 != id2) {
        for(const auto& n1Out: n1.outgoing){
          for(const auto& n2In : n2.incoming){
            if (n1Out.first == n2In.first){
              // print transition wiht label 
              // combSge20 -> combMux21 [inp = "%12", out = "out", label = "%12"];

                stream << n1.nodeType<<id1<<" -> "<<n2.nodeType<<id2<<" [inp = \""<<n1Out.second<<"\", out = out, label = \""<<n1Out.second<<"\"]\n";
            }
          }
        }
      }
    }
  }
  for(auto [id, n] : llvm::enumerate(nodes)){
    if (n.nodeType == "pack")
      for (auto ic: n.outgoing)
        stream << n.nodeType<<id<<" -> "<<ic.second<<id<<" [inp = \""<<ic.second<<"\", out = out, label = \""<<ic.second<<"\"]\n";
  }

  // }

  // print transitions and label them correctly 
  stream << "}";
  return success();
}


LogicalResult ExportDot::exportDot(Operation *module,
                                         llvm::raw_ostream &os) {
  if (module->getNumRegions() != 1)
    return module->emitError("must have exactly one region");
  if (!module->getRegion(0).hasOneBlock())
    return module->emitError("op region must have exactly one block");

  mlir::raw_indented_ostream ios(os);
  if (!failed(emitDot(module, ios)))
  return success();
}

//===----------------------------------------------------------------------===//
// circt-translate registration
//===----------------------------------------------------------------------===//

void ExportDot::registerExportDotTranslation() {
  static llvm::cl::opt<bool> inlineSingleUseValues(
      "dotexport-inline-single-use-values",
      llvm::cl::desc("Inline expressions that are used only once rather than "
                     "generating a let-binding"),
      llvm::cl::init(false));

  static mlir::TranslateFromMLIRRegistration toDot(
      "export-dot", "export Dot",
      [=](Operation *module, raw_ostream &output) {
        return ExportDot::exportDot(module, output);
      },
      [](mlir::DialectRegistry &registry) {
        // Register the 'func' and 'HW' dialects to support printing solver
        // scopes nested in functions and modules.
        registry
            .insert<mlir::func::FuncDialect, hw::HWDialect, dc::DCDialect, comb::CombDialect>();
      });
}
