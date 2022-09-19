//===- PrintHWModuleGraph.cpp - Print the instance graph --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Prints an HW module as a .dot graph.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWModuleGraph.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace hw;

namespace {

template <>
struct llvm::DOTGraphTraits<circt::hw::HWModuleOp>
    : public llvm::DefaultDOTGraphTraits {
  using DefaultDOTGraphTraits::DefaultDOTGraphTraits;

  static std::string getNodeLabel(circt::hw::detail::HWOperation *node,
                                  circt::hw::HWModuleOp) {
    return llvm::TypeSwitch<mlir::Operation *, std::string>(node)
        .Case<circt::comb::AddOp>([&](auto) { return "+"; })
        .Case<circt::comb::SubOp>([&](auto) { return "-"; })
        .Case<circt::comb::AndOp>([&](auto) { return "&"; })
        .Case<circt::comb::OrOp>([&](auto) { return "|"; })
        .Case<circt::comb::XorOp>([&](auto) { return "^"; })
        .Case<circt::comb::MulOp>([&](auto) { return "*"; })
        .Case<circt::comb::MuxOp>([&](auto) { return "mux"; })
        .Case<circt::comb::ShrSOp, circt::comb::ShrUOp>(
            [&](auto) { return ">>"; })
        .Case<circt::comb::ShlOp>([&](auto) { return "<<"; })
        .Case<circt::comb::ICmpOp>([&](auto op) {
          switch (op.getPredicate()) {
          case circt::comb::ICmpPredicate::eq:
          case circt::comb::ICmpPredicate::ceq:
          case circt::comb::ICmpPredicate::weq:
            return "==";
          case circt::comb::ICmpPredicate::wne:
          case circt::comb::ICmpPredicate::cne:
          case circt::comb::ICmpPredicate::ne:
            return "!=";
          case circt::comb::ICmpPredicate::uge:
          case circt::comb::ICmpPredicate::sge:
            return ">=";
          case circt::comb::ICmpPredicate::ugt:
          case circt::comb::ICmpPredicate::sgt:
            return ">";
          case circt::comb::ICmpPredicate::ule:
          case circt::comb::ICmpPredicate::sle:
            return "<=";
          case circt::comb::ICmpPredicate::ult:
          case circt::comb::ICmpPredicate::slt:
            return "<";
          }
          llvm_unreachable("unhandled ICmp predicate");
        })
        .Case<circt::seq::CompRegOp, circt::seq::FirReg>(
            [&](auto) { return "reg"; })
        .Case<circt::hw::ConstantOp>([&](auto op) {
          llvm::SmallString<64> valueString;
          op.getValue().toString(valueString, 10, false);
          return valueString.str().str();
        })
        .Default([&](auto op) { return op->getName().getStringRef().str(); });
  }

  std::string getNodeAttributes(circt::hw::detail::HWOperation *node,
                                circt::hw::HWModuleOp) {
    return llvm::TypeSwitch<mlir::Operation *, std::string>(node)
        .Case<circt::hw::ConstantOp>(
            [&](auto) { return "fillcolor=darkgoldenrod1,style=filled"; })
        .Case<circt::comb::MuxOp>([&](auto) {
          return "shape=invtrapezium,fillcolor=bisque,style=filled";
        })
        .Case<circt::hw::OutputOp>(
            [&](auto) { return "fillcolor=lightblue,style=filled"; })
        .Default([&](auto op) {
          llvm::TypeSwitch<mlir::Dialect *>(op->getDialect())
              .Case<circt::comb::CombDialect>([&](auto) {
                return "shape=oval,fillcolor=bisque,style=filled";
              })
              .template Case<circt::seq::SeqDialect>([&](auto) {
                return "shape=folder,fillcolor=gainsboro,style=filled";
              })
              .Default([&](auto) { return ""; });
        });
  }

  static void
  addCustomGraphFeatures(circt::hw::HWModuleOp mod,
                         llvm::GraphWriter<circt::hw::HWModuleOp> &g) {

    // Add module input args.
    auto &os = g.getOStream();
    os << "subgraph cluster_entry_args {\n";
    os << "label=\"Input arguments\";\n";
    for (auto arg : mod.getPorts().inputs) {
      g.emitSimpleNode(reinterpret_cast<void *>(arg.getId()), "",
                       arg.getName().str());
    }
    os << "}\n";
    for (auto [info, arg] :
         llvm::zip(mod.getPorts().inputs, mod.getArguments())) {
      for (auto user : arg.getUsers()) {
        g.emitEdge(reinterpret_cast<void *>(info.getId()), 0, user, -1, "");
      }
    }
  }

  template <typename Iterator>
  static std::string getEdgeAttributes(circt::hw::detail::HWOperation *node,
                                       Iterator it, circt::hw::HWModuleOp mod) {

    mlir::OpOperand &operand = *it.getCurrent();
    mlir::Value v = operand.get();
    std::string str;
    llvm::raw_string_ostream os(str);
    auto verboseEdges = mod->getAttrOfType<mlir::BoolAttr>("dot_verboseEdges");
    if (verboseEdges.getValue()) {
      os << "label=\"" << operand.getOperandNumber() << " (" << v.getType()
         << ")\"";
    }

    int64_t width = circt::hw::getBitWidth(v.getType());
    if (width > 1)
      os << " style=bold";

    return os.str();
  };
};

struct PrintHWModuleGraphPass
    : public PrintHWModuleGraphBase<PrintHWModuleGraphPass> {
  PrintHWModuleGraphPass() {}
  void runOnOperation() override {
    getOperation().walk([&](hw::HWModuleOp module) {
      // We don't really have any other way of forwarding draw arguments to the
      // DOTGraphTraits for HWModule except through the module itself - as an
      // attribute.
      module->setAttr("dot_verboseEdges",
                      BoolAttr::get(module.getContext(), verboseEdges));

      llvm::WriteGraph(module, module.getName(), /*ShortNames=*/false);
    });
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::hw::createPrintHWModuleGraphPass() {
  return std::make_unique<PrintHWModuleGraphPass>();
}
