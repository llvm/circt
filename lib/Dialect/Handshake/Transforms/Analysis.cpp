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

#include "PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace handshake;
using namespace mlir;

static bool isControlOp(Operation *op) {
  return op->hasAttr("control") &&
         op->getAttrOfType<BoolAttr>("control").getValue();
}

static void dotPrintNode(llvm::raw_fd_ostream &outfile, Operation *op,
                         DenseMap<Operation *, unsigned> &opIDs) {
  outfile << "\t\t";
  outfile << "\"" + op->getName().getStringRef().str() + "_" +
                 std::to_string(opIDs[op]) + "\"";
  outfile << " [";

  /// Fill color
  outfile << "fillcolor = ";
  outfile
      << llvm::TypeSwitch<Operation *, std::string>(op)
             .Case<handshake::ForkOp, handshake::LazyForkOp, handshake::MuxOp,
                   handshake::JoinOp>([&](auto) { return "lavender"; })
             .Case<handshake::BufferOp>([&](auto) { return "lightgreen"; })
             .Case<handshake::ReturnOp>([&](auto) { return "gold"; })
             .Case<handshake::SinkOp, handshake::ConstantOp>(
                 [&](auto) { return "gainsboro"; })
             .Case<handshake::MemoryOp, handshake::LoadOp, handshake::StoreOp>(
                 [&](auto) { return "coral"; })
             .Case<handshake::MergeOp, handshake::ControlMergeOp,
                   handshake::BranchOp, handshake::ConditionalBranchOp>(
                 [&](auto) { return "lightblue"; })
             .Default([&](auto) { return "moccasin"; });

  /// Shape
  outfile << ", shape=";
  if (op->getDialect()->getNamespace() == "handshake")
    outfile << "box";
  else
    outfile << "oval";

  /// Label
  outfile << ", label=\"";
  outfile << llvm::TypeSwitch<Operation *, std::string>(op)
                 .Case<handshake::ConstantOp>([&](auto op) {
                   return std::to_string(
                       op->template getAttrOfType<mlir::IntegerAttr>("value")
                           .getValue()
                           .getSExtValue());
                 })
                 .Case<handshake::ControlMergeOp>(
                     [&](auto) { return "cmerge"; })
                 .Case<handshake::ConditionalBranchOp>(
                     [&](auto) { return "cbranch"; })
                 .Case<arith::AddIOp>([&](auto) { return "+"; })
                 .Case<arith::SubIOp>([&](auto) { return "-"; })
                 .Case<arith::AndIOp>([&](auto) { return "&"; })
                 .Case<arith::OrIOp>([&](auto) { return "|"; })
                 .Case<arith::XOrIOp>([&](auto) { return "^"; })
                 .Case<arith::MulIOp>([&](auto) { return "*"; })
                 .Case<arith::ShRSIOp, arith::ShRUIOp>(
                     [&](auto) { return ">>"; })
                 .Case<arith::ShLIOp>([&](auto) { return "<<"; })
                 .Case<arith::CmpIOp>([&](arith::CmpIOp op) {
                   switch (op.predicate()) {
                   case arith::CmpIPredicate::eq:
                     return "==";
                   case arith::CmpIPredicate::ne:
                     return "!=";
                   case arith::CmpIPredicate::uge:
                   case arith::CmpIPredicate::sge:
                     return ">=";
                   case arith::CmpIPredicate::ugt:
                   case arith::CmpIPredicate::sgt:
                     return ">";
                   case arith::CmpIPredicate::ule:
                   case arith::CmpIPredicate::sle:
                     return "<=";
                   case arith::CmpIPredicate::ult:
                   case arith::CmpIPredicate::slt:
                     return "<";
                   }
                   llvm_unreachable("unhandled cmpi predicate");
                 })
                 .Default([&](auto op) {
                   auto opDialect = op->getDialect()->getNamespace();
                   std::string label = op->getName().getStringRef().str();
                   if (opDialect == "handshake")
                     label.erase(0, StringLiteral("handshake.").size());

                   return label;
                 });
  outfile << "\"";

  /// Style; add dashed border for control nodes
  outfile << ", style=\"filled";
  if (isControlOp(op))
    outfile << ", dashed";

  outfile << "\"";

  outfile << "]\n";
}

/// Returns true if v is used as a control operand in op
static bool isControlOperand(Operation *op, Value v) {
  if (isControlOp(op))
    return true;

  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<handshake::MuxOp, handshake::ConditionalBranchOp>(
          [&](auto op) { return v == op.getOperand(0); })
      .Case<handshake::ControlMergeOp>([&](auto) { return true; })
      .Default([](auto) { return false; });
}

template <typename FuncOp>
void dotPrint(FuncOp f, StringRef name) {
  // Prints DOT representation of the dataflow graph, used for debugging.
  DenseMap<Block *, unsigned> blockIDs;
  DenseMap<Operation *, unsigned> opIDs;
  unsigned i = 0;
  unsigned j = 0;

  for (Block &block : f) {
    blockIDs[&block] = i++;
    for (Operation &op : block)
      opIDs[&op] = j++;
  }

  std::error_code ec;
  llvm::raw_fd_ostream outfile(name.str() + ".dot", ec);

  outfile << "Digraph G {\n\tsplines=spline;\n";

  for (Block &block : f) {
    outfile << "\tsubgraph cluster_" + std::to_string(blockIDs[&block]) +
                   " {\n";
    outfile << "\tnode [shape=box style=filled fillcolor=\"white\"]\n";
    outfile << "\tcolor = \"darkgreen\"\n";
    outfile << "\t\tlabel = \" block " + std::to_string(blockIDs[&block]) +
                   "\"\n";

    for (Operation &op : block)
      dotPrintNode(outfile, &op, opIDs);

    for (Operation &op : block) {
      if (op.getNumResults() == 0)
        continue;

      for (auto result : op.getResults()) {
        for (auto &u : result.getUses()) {
          Operation *useOp = u.getOwner();
          if (useOp->getBlock() == &block) {
            outfile << "\t\t";
            outfile << "\"" + op.getName().getStringRef().str() + "_" +
                           std::to_string(opIDs[&op]) + "\"";
            outfile << " -> ";
            outfile << "\"" + useOp->getName().getStringRef().str() + "_" +
                           std::to_string(opIDs[useOp]) + "\"";

            if (isControlOp(&op) || isControlOperand(useOp, result))
              outfile << " [style=\"dashed\"]\n";

            outfile << "\n";
          }
        }
      }
    }

    /// Annotate block argument uses
    for (auto barg : enumerate(block.getArguments())) {
      std::string argName = "arg" + std::to_string(barg.index());
      outfile << "\t\"" << argName << "\" [shape=diamond";
      if (barg.index() == block.getNumArguments() - 1)
        outfile << ", style=dashed";
      outfile << "]\n";
      for (auto useOp : barg.value().getUsers()) {
        outfile << argName << " -> \""
                << useOp->getName().getStringRef().str() + "_" +
                       std::to_string(opIDs[useOp]) + "\"";
        if (isControlOperand(useOp, barg.value()))
          outfile << " [style=\"dashed\"]";
        outfile << "\n";
      }
    }

    outfile << "\t}\n";

    for (Operation &op : block) {
      if (op.getNumResults() == 0)
        continue;

      for (auto result : op.getResults()) {
        for (auto &u : result.getUses()) {
          Operation *useOp = u.getOwner();
          if (useOp->getBlock() != &block) {
            outfile << "\t\t";
            outfile << "\"" + op.getName().getStringRef().str() + "_" +
                           std::to_string(opIDs[&op]) + "\"";
            outfile << " -> ";
            outfile << "\"" + useOp->getName().getStringRef().str() + "_" +
                           std::to_string(opIDs[useOp]) + "\"";
            outfile << " [minlen = 3]\n";
          }
        }
      }
    }
  }

  outfile << "\t}\n";
  outfile.close();
}

namespace {
struct HandshakeDotPrintPass
    : public HandshakeDotPrintBase<HandshakeDotPrintPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    for (auto func : m.getOps<handshake::FuncOp>())
      dotPrint(func, func.getName());
  };
};

struct HandshakeOpCountPass
    : public HandshakeOpCountBase<HandshakeOpCountPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    for (auto func : m.getOps<handshake::FuncOp>()) {
      int count = 0;
      int fork_count = 0;
      int merge_count = 0;
      int branch_count = 0;
      int join_count = 0;
      for (Operation &op : func.getOps()) {
        if (isa<ForkOp>(op))
          fork_count++;
        else if (isa<MergeLikeOpInterface>(op))
          merge_count++;
        else if (isa<ConditionalBranchOp>(op))
          branch_count++;
        else if (isa<JoinOp>(op))
          join_count++;
        else if (!isa<handshake::BranchOp>(op) && !isa<SinkOp>(op) &&
                 !isa<TerminatorOp>(op))
          count++;
      }

      llvm::outs() << "// Fork count: " << fork_count << "\n";
      llvm::outs() << "// Merge count: " << merge_count << "\n";
      llvm::outs() << "// Branch count: " << branch_count << "\n";
      llvm::outs() << "// Join count: " << join_count << "\n";
      int total = count + fork_count + merge_count + branch_count;
      llvm::outs() << "// Total op count: " << total << "\n";
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
circt::handshake::createHandshakeDotPrintPass() {
  return std::make_unique<HandshakeDotPrintPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
circt::handshake::createHandshakeOpCountPass() {
  return std::make_unique<HandshakeOpCountPass>();
}
