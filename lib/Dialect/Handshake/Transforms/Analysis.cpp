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
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace handshake;
using namespace mlir;

static bool isControlOp(Operation *op) {
  return op->hasAttr("control") &&
         op->getAttrOfType<BoolAttr>("control").getValue();
}

namespace {
struct HandshakeDotPrintPass
    : public HandshakeDotPrintBase<HandshakeDotPrintPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Resolve the instance graph to get a top-level module.
    std::string topLevel;
    handshake::InstanceGraph uses;
    SmallVector<std::string> sortedFuncs;
    if (resolveInstanceGraph(m, uses, topLevel, sortedFuncs).failed()) {
      signalPassFailure();
      return;
    }

    handshake::FuncOp topLevelOp =
        cast<handshake::FuncOp>(m.lookupSymbol(topLevel));

    // Create top-level graph.
    std::error_code ec;
    llvm::raw_fd_ostream outfile(topLevel + ".dot", ec);
    mlir::raw_indented_ostream os(outfile);

    os << "Digraph G {\n";
    os.indent();
    os << "splines=spline;\n";
    dotPrint(os, topLevelOp);
    os.unindent();
    os << "}\n";
    outfile.close();
  };

private:
  /// Prints an instance of a handshake.func to the graph. Returns the unique
  /// name that was assigned to the instance.
  std::string dotPrint(mlir::raw_indented_ostream &os, handshake::FuncOp f);

  /// Maintain a mapping of module names and the number of times one of those
  /// modules have been instantiated in the design. This is used to generate
  /// unique names in the output graph.
  std::map<std::string, unsigned> instanceIdMap;

  /// A mapping between operations and their unique name in the .dot file.
  DenseMap<Operation *, std::string> opNameMap;

  /// A mapping between block arguments and their unique name in the .dot file.
  DenseMap<Value, std::string> argNameMap;

  void setUsedByMapping(Value v, Operation *op, StringRef node);
  void setProducedByMapping(Value v, Operation *op, StringRef node);

  /// Returns the name of the vertex using 'v' through 'consumer'.
  std::string getUsedByNode(Value v, Operation *consumer);
  /// Returns the name of the vertex producing 'v' through 'producer'.
  std::string getProducedByNode(Value v, Operation *producer);

  /// Maintain mappings between a value, the operation which (uses/produces) it,
  /// and the node name which the (tail/head) of an edge should refer to. This
  /// is used to resolve edges across handshake.instance's.
  // "'value' used by 'operation*' is used by the 'string' vertex"
  DenseMap<Value, std::map<Operation *, std::string>> usedByMapping;
  // "'value' produced by 'operation*' is produced from the 'string' vertex"
  DenseMap<Value, std::map<Operation *, std::string>> producedByMapping;
};

struct HandshakeOpCountPass
    : public HandshakeOpCountBase<HandshakeOpCountPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    for (auto func : m.getOps<handshake::FuncOp>()) {
      int count = 0;
      int forkCount = 0;
      int mergeCount = 0;
      int branchCount = 0;
      int joinCount = 0;
      for (Operation &op : func.getOps()) {
        if (isa<ForkOp>(op))
          forkCount++;
        else if (isa<MergeLikeOpInterface>(op))
          mergeCount++;
        else if (isa<ConditionalBranchOp>(op))
          branchCount++;
        else if (isa<JoinOp>(op))
          joinCount++;
        else if (!isa<handshake::BranchOp>(op) && !isa<SinkOp>(op) &&
                 !isa<TerminatorOp>(op))
          count++;
      }

      llvm::outs() << "// Fork count: " << forkCount << "\n";
      llvm::outs() << "// Merge count: " << mergeCount << "\n";
      llvm::outs() << "// Branch count: " << branchCount << "\n";
      llvm::outs() << "// Join count: " << joinCount << "\n";
      int total = count + forkCount + mergeCount + branchCount;
      llvm::outs() << "// Total op count: " << total << "\n";
    }
  }
};

} // namespace

/// Prints an operation to the dot file and returns the unique name for the
/// operation within the graph.
static std::string dotPrintNode(mlir::raw_indented_ostream &outfile,
                                StringRef instanceName, Operation *op,
                                DenseMap<Operation *, unsigned> &opIDs) {
  std::string opName =
      ("\"" + instanceName + "_" + op->getName().getStringRef().str() + "_" +
       std::to_string(opIDs[op]) + "\"")
          .str();

  outfile << opName << " [";

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

  return opName;
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

static std::string getLocalName(StringRef instanceName, StringRef suffix) {
  return (instanceName + "_" + suffix).str();
}

static std::string getArgName(handshake::FuncOp op, unsigned index) {
  return op.getArgName(index).getValue().str();
}

static std::string getUniqueArgName(StringRef instanceName,
                                    handshake::FuncOp op, unsigned index) {
  return getLocalName(instanceName, getArgName(op, index));
}

static std::string getResName(handshake::FuncOp op, unsigned index) {
  return op.getResName(index).getValue().str();
}

static std::string getUniqueResName(StringRef instanceName,
                                    handshake::FuncOp op, unsigned index) {
  return getLocalName(instanceName, getResName(op, index));
}

void HandshakeDotPrintPass::setUsedByMapping(Value v, Operation *op,
                                             StringRef node) {
  usedByMapping[v][op] = node;
}
void HandshakeDotPrintPass::setProducedByMapping(Value v, Operation *op,
                                                 StringRef node) {
  producedByMapping[v][op] = node;
}

std::string HandshakeDotPrintPass::getUsedByNode(Value v, Operation *consumer) {
  // Check if there is any mapping registerred for the value-use relation.
  auto it = usedByMapping.find(v);
  if (it != usedByMapping.end()) {
    auto it2 = it->second.find(consumer);
    if (it2 != it->second.end())
      return it2->second;
  }

  // fallback to the registerred name for the operation
  auto opNameIt = opNameMap.find(consumer);
  assert(opNameIt != opNameMap.end() &&
         "No name registered for the operation!");
  return opNameIt->second;
}

std::string HandshakeDotPrintPass::getProducedByNode(Value v,
                                                     Operation *producer) {
  // Check if there is any mapping registerred for the value-produce relation.
  auto it = producedByMapping.find(v);
  if (it != producedByMapping.end()) {
    auto it2 = it->second.find(producer);
    if (it2 != it->second.end())
      return it2->second;
  }

  // fallback to the registerred name for the operation
  auto opNameIt = opNameMap.find(producer);
  assert(opNameIt != opNameMap.end() &&
         "No name registered for the operation!");
  return opNameIt->second;
}

std::string HandshakeDotPrintPass::dotPrint(mlir::raw_indented_ostream &os,
                                            handshake::FuncOp f) {
  // Prints DOT representation of the dataflow graph, used for debugging.
  DenseMap<Block *, unsigned> blockIDs;
  DenseMap<Operation *, unsigned> opIDs;
  auto name = f.getName();
  unsigned thisId = instanceIdMap[name.str()]++;
  std::string instanceName = name.str() + std::to_string(thisId);
  unsigned i = 0;
  unsigned j = 0;

  for (Block &block : f) {
    blockIDs[&block] = i++;
    for (Operation &op : block)
      opIDs[&op] = j++;
  }

  os << "// Subgraph for instance of " << name << "\n";
  os << "subgraph cluster_" << instanceName << " {\n";
  os.indent();
  os << "labeljust=\"l\"\n";
  os << "node [shape=box style=filled fillcolor=\"white\"]\n";
  os << "color = \"darkgreen\"\n";
  os << "label = \"" << instanceName << "\"\n";

  Block *bodyBlock = &f.getBody().front();

  /// Print function arg and res nodes.
  os << "// Function argument nodes\n";
  os << "subgraph cluster_" << instanceName << "_args {\n";
  os.indent();
  // No label or border; the subgraph just forces args to stay together in the
  // diagram.
  os << "label=\"\"\n";
  os << "peripheries=0\n";
  for (auto barg : enumerate(bodyBlock->getArguments())) {
    auto argName = getArgName(f, barg.index());
    os << "\"" << getLocalName(instanceName, argName) << "\" [shape=diamond";
    if (barg.index() == bodyBlock->getNumArguments() - 1) // ctrl
      os << ", style=dashed";
    os << " label=\"" << argName << "\"";
    os << "]\n";
  }
  os.unindent();
  os << "}\n";

  os << "// Function return nodes\n";
  os << "subgraph cluster_" << instanceName << "_res {\n";
  os.indent();
  // No label or border; the subgraph just forces args to stay together in the
  // diagram.
  os << "label=\"\"\n";
  os << "peripheries=0\n";
  // Get the return op; a handshake.func always has a terminator, making this
  // safe.
  auto returnOp = *f.getBody().getOps<handshake::ReturnOp>().begin();
  for (auto res : llvm::enumerate(returnOp.getOperands())) {
    auto resName = getResName(f, res.index());
    auto uniqueResName = getUniqueResName(instanceName, f, res.index());
    os << "\"" << uniqueResName << "\" [shape=diamond";
    if (res.index() == bodyBlock->getNumArguments() - 1) // ctrl
      os << ", style=dashed";
    os << " label=\"" << resName << "\"";
    os << "]\n";

    // Create a mapping between the return op argument uses and the return
    // nodes.
    setUsedByMapping(res.value(), returnOp, uniqueResName);
  }
  os.unindent();
  os << "}\n";

  /// Print operation nodes.
  os << "// Function operation nodes\n";
  for (Operation &op : *bodyBlock) {
    if (!isa<handshake::InstanceOp, handshake::ReturnOp>(op)) {
      // Regular node in the diagram.
      opNameMap[&op] = dotPrintNode(os, instanceName, &op, opIDs);
      continue;
    }
    auto instOp = dyn_cast<handshake::InstanceOp>(op);
    if (instOp) {
      // Recurse into instantiated submodule.
      auto calledFuncOp =
          instOp->getParentOfType<ModuleOp>().lookupSymbol<handshake::FuncOp>(
              instOp.getModule());
      assert(calledFuncOp);
      auto subInstanceName = dotPrint(os, calledFuncOp);

      // Create a mapping between the instance arguments and the arguments to
      // the module which it instantiated.
      for (auto arg : llvm::enumerate(instOp.getOperands())) {
        setUsedByMapping(
            arg.value(), instOp,
            getUniqueArgName(subInstanceName, calledFuncOp, arg.index()));
      }
      // Create a  mapping between the instance results and the results from the
      // module which it instantiated.
      for (auto res : llvm::enumerate(instOp.getResults())) {
        setProducedByMapping(
            res.value(), instOp,
            getUniqueResName(subInstanceName, calledFuncOp, res.index()));
      }
    }
  }

  /// Print operation result edges.
  os << "// Operation result edges\n";
  for (Operation &op : *bodyBlock) {
    for (auto result : op.getResults()) {
      for (auto &u : result.getUses()) {
        Operation *useOp = u.getOwner();
        if (useOp->getBlock() == bodyBlock) {
          os << getProducedByNode(result, &op);
          os << " -> ";
          os << getUsedByNode(result, useOp);
          if (isControlOp(&op) || isControlOperand(useOp, result))
            os << " [style=\"dashed\"]";

          os << "\n";
        }
      }
    }
  }

  os << "}\n";

  /// Print edges for function argument uses.
  os << "// Function argument edges\n";
  for (auto barg : enumerate(bodyBlock->getArguments())) {
    auto argName = getArgName(f, barg.index());
    os << "\"" << getLocalName(instanceName, argName) << "\" [shape=diamond";
    if (barg.index() == bodyBlock->getNumArguments() - 1)
      os << ", style=dashed";
    os << "]\n";
    for (auto *useOp : barg.value().getUsers()) {
      os << "" << getLocalName(instanceName, argName) << " -> "
         << getUsedByNode(barg.value(), useOp);
      if (isControlOperand(useOp, barg.value()))
        os << " [style=\"dashed\"]";
      os << "\n";
    }
  }

  os.unindent();
  return instanceName;
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
circt::handshake::createHandshakeDotPrintPass() {
  return std::make_unique<HandshakeDotPrintPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
circt::handshake::createHandshakeOpCountPass() {
  return std::make_unique<HandshakeOpCountPass>();
}
