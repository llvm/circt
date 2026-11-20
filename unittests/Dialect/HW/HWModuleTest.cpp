//===- HWModuleTest.cpp - HW Module utility tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace hw;

using Direction = ModulePort::Direction;

/// Render the port list of `mod` as "in a, out x, ...".
static std::string portsToString(HWModuleLike mod) {
  std::string result;
  llvm::raw_string_ostream os(result);
  llvm::interleaveComma(mod.getHWModuleType().getPorts(), os,
                        [&](const ModulePort &port) {
                          os << (port.dir == Direction::Output ? "out " : "in ")
                             << port.name.getValue();
                        });
  return result;
}

namespace {

TEST(HWModuleOpTest, AddOutputs) {
  // Create a hw.module with no ports.
  MLIRContext context;
  context.loadDialect<HWDialect>();
  LocationAttr loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());
  auto top = HWModuleOp::create(builder, StringAttr::get(&context, "Top"),
                                ArrayRef<PortInfo>{});

  builder.setInsertionPointToStart(top.getBodyBlock());
  auto wireTy = builder.getIntegerType(2);

  // Add two ports.
  SmallVector<std::pair<StringAttr, Value>> appendPorts;
  auto wireA = ConstantOp::create(builder, wireTy, 0);
  appendPorts.emplace_back(builder.getStringAttr("a"), wireA);
  auto wireD = ConstantOp::create(builder, wireTy, 1);
  appendPorts.emplace_back(builder.getStringAttr("d"), wireD);
  top.appendOutputs(appendPorts);

  SmallVector<std::pair<StringAttr, Value>> insertPorts;
  auto wireB = ConstantOp::create(builder, wireTy, 2);
  insertPorts.emplace_back(builder.getStringAttr("b"), wireB);
  auto wireC = ConstantOp::create(builder, wireTy, 3);
  insertPorts.emplace_back(builder.getStringAttr("c"), wireC);
  top.insertOutputs(1, insertPorts);

  // Convenience methods.
  auto wireF = hw::ConstantOp::create(builder, APInt(2, 0));
  top.appendOutput("f", wireF);
  auto wireQ = hw::ConstantOp::create(builder, APInt(2, 0));
  top.prependOutput("q", wireQ);

  auto ports = top.getPortList();
  ASSERT_EQ(ports.size(), 6u);

  EXPECT_EQ(ports[0].name, builder.getStringAttr("q"));
  EXPECT_EQ(ports[0].dir, ModulePort::Direction::Output);
  EXPECT_EQ(ports[0].type, wireTy);

  EXPECT_EQ(ports[1].name, builder.getStringAttr("a"));
  EXPECT_EQ(ports[1].dir, ModulePort::Direction::Output);
  EXPECT_EQ(ports[1].type, wireTy);

  EXPECT_EQ(ports[2].name, builder.getStringAttr("b"));
  EXPECT_EQ(ports[2].dir, ModulePort::Direction::Output);
  EXPECT_EQ(ports[2].type, wireTy);

  EXPECT_EQ(ports[3].name, builder.getStringAttr("c"));
  EXPECT_EQ(ports[3].dir, ModulePort::Direction::Output);
  EXPECT_EQ(ports[3].type, wireTy);

  EXPECT_EQ(ports[4].name, builder.getStringAttr("d"));
  EXPECT_EQ(ports[4].dir, ModulePort::Direction::Output);
  EXPECT_EQ(ports[4].type, wireTy);

  EXPECT_EQ(ports[5].name, builder.getStringAttr("f"));
  EXPECT_EQ(ports[5].dir, ModulePort::Direction::Output);
  EXPECT_EQ(ports[5].type, wireTy);

  auto output = cast<OutputOp>(top.getBodyBlock()->getTerminator());
  ASSERT_EQ(output->getNumOperands(), 6u);

  EXPECT_EQ(output->getOperand(0), wireQ.getResult());
  EXPECT_EQ(output->getOperand(1), wireA.getResult());
  EXPECT_EQ(output->getOperand(2), wireB.getResult());
  EXPECT_EQ(output->getOperand(3), wireC.getResult());
  EXPECT_EQ(output->getOperand(4), wireD.getResult());
  EXPECT_EQ(output->getOperand(5), wireF.getResult());
}

TEST(HWModuleOpTest, AddInputs) {
  // Create a hw.module with no ports.
  MLIRContext context;
  context.loadDialect<HWDialect>();
  LocationAttr loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());
  auto top = HWModuleOp::create(builder, StringAttr::get(&context, "Top"),
                                ArrayRef<PortInfo>{});

  builder.setInsertionPointToStart(top.getBodyBlock());

  // Convenience methods.
  auto tyF = builder.getIntegerType(6);
  top.appendInput("f", tyF);
  auto tyQ = builder.getIntegerType(1);
  top.prependInput("q", tyQ);

  auto ports = top.getPortList();
  ASSERT_EQ(ports.size(), 2u);

  EXPECT_EQ(ports[0].name, builder.getStringAttr("q"));
  EXPECT_EQ(ports[0].dir, ModulePort::Direction::Input);
  EXPECT_EQ(ports[0].type, tyQ);

  EXPECT_EQ(ports[1].name, builder.getStringAttr("f"));
  EXPECT_EQ(ports[1].dir, ModulePort::Direction::Input);
  EXPECT_EQ(ports[1].type, tyF);
}

/// Builds hw.module and hw.module.extern modules with i1 ports.
/// Each port is located at a NameLoc of its name, so locations are checkable.
class ModifyPortsTest : public testing::Test {
protected:
  enum Kind { Module, Extern };
  static constexpr Kind allKinds[] = {Module, Extern};

  ModifyPortsTest() : builder(UnknownLoc::get(&context), &context) {
    context.loadDialect<HWDialect>();
    top = ModuleOp::create(builder.getLoc());
    builder.setInsertionPointToEnd(top->getBody());
  }

  static const char *kindName(Kind kind) {
    switch (kind) {
    case Module:
      return "hw.module";
    case Extern:
      return "hw.module.extern";
    }
    llvm_unreachable("unknown kind");
  }

  /// Make a port from "in name" or "out name".
  PortInfo makePort(StringRef spec) {
    auto [dir, name] = spec.split(' ');
    PortInfo port;
    port.name = builder.getStringAttr(name);
    port.type = builder.getI1Type();
    port.dir = dir == "out" ? Direction::Output : Direction::Input;
    port.loc = NameLoc::get(port.name);
    return port;
  }

  /// Make a port as `makePort` does, carrying a "tag" attribute naming it.
  PortInfo makeTaggedPort(StringRef spec) {
    auto port = makePort(spec);
    port.attrs =
        builder.getDictionaryAttr(builder.getNamedAttr("tag", port.name));
    return port;
  }

  HWModuleLike build(Kind kind, ArrayRef<PortInfo> portList) {
    ModulePortInfo ports(portList);
    auto name = builder.getStringAttr("M" + Twine(numModules++));
    switch (kind) {
    case Module:
      return HWModuleOp::create(
          builder, name, ports, [&](OpBuilder &b, HWModulePortAccessor &acc) {
            for (auto [idx, port] : llvm::enumerate(ports.getOutputs()))
              acc.setOutput(idx, ConstantOp::create(b, port.loc, port.type, 0));
          });
    case Extern:
      return HWModuleExternOp::create(builder, name, ports);
    }
    llvm_unreachable("unknown kind");
  }

  HWModuleLike build(Kind kind, ArrayRef<StringRef> specs) {
    return build(kind, llvm::map_to_vector(
                           specs, [&](StringRef s) { return makePort(s); }));
  }

  /// Edit the ports of `mod`.
  /// On a hw.module, edit the body around the call as modifyPorts callers do.
  void modify(HWModuleLike mod,
              ArrayRef<std::pair<unsigned, PortInfo>> insertInputs,
              ArrayRef<std::pair<unsigned, PortInfo>> insertOutputs,
              ArrayRef<unsigned> eraseInputs, ArrayRef<unsigned> eraseOutputs) {
    auto hwModule = dyn_cast<HWModuleOp>(mod.getOperation());
    if (!hwModule) {
      cast<HWMutableModuleLike>(mod.getOperation())
          .modifyPorts(insertInputs, insertOutputs, eraseInputs, eraseOutputs);
      return;
    }

    Block *body = hwModule.getBodyBlock();
    for (auto [idx, insert] : llvm::enumerate(insertInputs))
      body->insertArgument(insert.first + idx, insert.second.type,
                           insert.second.loc);
    hwModule.modifyPorts(insertInputs, insertOutputs, eraseInputs,
                         eraseOutputs);
    BitVector erasedArgs(body->getNumArguments());
    for (auto idx : eraseInputs)
      erasedArgs.set(idx + llvm::count_if(insertInputs, [&](auto &insert) {
                       return insert.first <= idx;
                     }));
    body->eraseArguments(erasedArgs);

    auto output = cast<OutputOp>(body->getTerminator());
    OpBuilder b(output);
    SmallVector<Value> outputs;
    for (auto type : hwModule.getOutputTypes())
      outputs.push_back(ConstantOp::create(b, output.getLoc(), type, 0));
    output->setOperands(outputs);
  }

  /// Expect each port to be located at a NameLoc of its name, and exactly the
  /// `tagged` ports to carry a "tag" attribute naming them.
  void expectAttrsAndLocsFollowPorts(HWModuleLike mod,
                                     ArrayRef<StringRef> tagged) {
    for (auto port : mod.getPortList()) {
      SCOPED_TRACE(port.getName().str());
      EXPECT_EQ(port.loc, NameLoc::get(port.name));
      if (llvm::is_contained(tagged, port.getName()))
        EXPECT_EQ(port.attrs.get("tag"), port.name);
      else
        EXPECT_FALSE(port.attrs.get("tag"));
    }
  }

  MLIRContext context;
  ImplicitLocOpBuilder builder;
  OwningOpRef<ModuleOp> top;
  unsigned numModules = 0;
};

TEST_F(ModifyPortsTest, InsertKeepsInterleavedOrder) {
  for (auto kind : allKinds) {
    SCOPED_TRACE(kindName(kind));
    auto mod = build(kind, {"in a", "out x", "in b", "out y", "in c"});
    modify(mod, {{1, makePort("in n")}}, {{1, makePort("out z")}}, {}, {});
    EXPECT_EQ(portsToString(mod),
              "in a, out x, in n, in b, out z, out y, in c");
    expectAttrsAndLocsFollowPorts(mod, {});
  }
  EXPECT_TRUE(succeeded(verify(*top)));
}

TEST_F(ModifyPortsTest, EraseKeepsInterleavedOrder) {
  for (auto kind : allKinds) {
    SCOPED_TRACE(kindName(kind));
    auto mod = build(kind, {"in a", "out x", "in b", "out y", "in c"});
    modify(mod, {}, {}, {1}, {0});
    EXPECT_EQ(portsToString(mod), "in a, out y, in c");
    expectAttrsAndLocsFollowPorts(mod, {});
  }
  EXPECT_TRUE(succeeded(verify(*top)));
}

TEST_F(ModifyPortsTest, InsertAndEraseInOneCall) {
  for (auto kind : allKinds) {
    SCOPED_TRACE(kindName(kind));
    // An erased port still anchors inserts before it and, if it was the last
    // port of its direction, inserts past the end.
    auto mod = build(kind, {"out w", "in a", "out x", "in b", "out y", "in c"});
    modify(mod, {{0, makePort("in m")}, {1, makePort("in n")}},
           {{3, makePort("out z")}}, {0, 1}, {2});
    EXPECT_EQ(portsToString(mod), "out w, in m, out x, in n, out z, in c");
    expectAttrsAndLocsFollowPorts(mod, {});
  }
  EXPECT_TRUE(succeeded(verify(*top)));
}

TEST_F(ModifyPortsTest, SameIndexInsertsKeepListOrder) {
  for (auto kind : allKinds) {
    SCOPED_TRACE(kindName(kind));
    auto mod = build(kind, {"in a", "out x", "in b", "out y", "in c"});
    modify(mod,
           {{1, makePort("in p")},
            {1, makePort("in q")},
            {3, makePort("in r")},
            {3, makePort("in s")}},
           {{0, makePort("out u")}, {0, makePort("out v")}}, {}, {});
    EXPECT_EQ(portsToString(mod),
              "in a, out u, out v, out x, in p, in q, in b, "
              "out y, in c, in r, in s");
    expectAttrsAndLocsFollowPorts(mod, {});
  }
  EXPECT_TRUE(succeeded(verify(*top)));
}

TEST_F(ModifyPortsTest, InsertIntoEmptyDirection) {
  for (auto kind : allKinds) {
    SCOPED_TRACE(kindName(kind));
    auto noInputs = build(kind, {"out x", "out y"});
    modify(noInputs, {{0, makePort("in n")}}, {}, {}, {});
    EXPECT_EQ(portsToString(noInputs), "in n, out x, out y");

    auto noOutputs = build(kind, {"in a", "in b"});
    modify(noOutputs, {}, {{0, makePort("out z")}}, {}, {});
    EXPECT_EQ(portsToString(noOutputs), "in a, in b, out z");

    auto empty = build(kind, ArrayRef<StringRef>{});
    modify(empty, {{0, makePort("in n")}}, {{0, makePort("out z")}}, {}, {});
    EXPECT_EQ(portsToString(empty), "in n, out z");
    expectAttrsAndLocsFollowPorts(empty, {});
  }
  EXPECT_TRUE(succeeded(verify(*top)));
}

TEST_F(ModifyPortsTest, InputsFirstStaysInputsFirst) {
  for (auto kind : allKinds) {
    SCOPED_TRACE(kindName(kind));
    // Input appends and output inserts at index 0 meet between the last input
    // and the first output, inputs first.
    auto mod = build(kind, {"in a", "in b", "out x", "out y"});
    modify(mod, {{0, makePort("in m")}, {2, makePort("in n")}},
           {{0, makePort("out w")}, {2, makePort("out z")}}, {}, {1});
    EXPECT_EQ(portsToString(mod),
              "in m, in a, in b, in n, out w, out x, out z");
    expectAttrsAndLocsFollowPorts(mod, {});
  }
  EXPECT_TRUE(succeeded(verify(*top)));
}

TEST_F(ModifyPortsTest, AttrsAndLocsFollowPorts) {
  for (auto kind : allKinds) {
    SCOPED_TRACE(kindName(kind));
    auto mod = build(kind, {makePort("in a"), makePort("out x"),
                            makeTaggedPort("in b"), makeTaggedPort("out y"),
                            makePort("in c")});
    auto sym = InnerSymAttr::get(builder.getStringAttr("c_sym"));
    mod.setPortAttr(
        4, builder.getStringAttr(HWModuleLike::getPortSymbolAttrName()), sym);

    modify(mod, {{1, makeTaggedPort("in n")}, {3, makePort("in m")}},
           {{0, makePort("out z")}}, {0}, {1});
    EXPECT_EQ(portsToString(mod), "out z, out x, in n, in b, in c, in m");
    expectAttrsAndLocsFollowPorts(mod, {"b", "n"});
    for (auto port : mod.getPortList())
      EXPECT_EQ(port.getSym(), port.getName() == "c" ? sym : InnerSymAttr());
  }
  EXPECT_TRUE(succeeded(verify(*top)));
}

TEST_F(ModifyPortsTest, NoPortAttrsStaysEmpty) {
  for (auto kind : allKinds) {
    SCOPED_TRACE(kindName(kind));
    auto mod = build(kind, {makePort("in a"), makeTaggedPort("out x")});
    modify(mod, {{0, makePort("in n")}}, {}, {}, {});
    EXPECT_FALSE(mod.getAllPortAttrs().empty());
    modify(mod, {}, {}, {}, {0});
    EXPECT_TRUE(mod.getAllPortAttrs().empty());
  }
  EXPECT_TRUE(succeeded(verify(*top)));
}

TEST_F(ModifyPortsTest, InsertInputAndOutputsUpdateBody) {
  auto mod =
      cast<HWModuleOp>(build(Module, {"in a", "out x", "in b", "out y"}));
  Block *body = mod.getBodyBlock();
  auto output = cast<OutputOp>(body->getTerminator());
  Value a = body->getArgument(0), b = body->getArgument(1);
  Value x = output.getOperand(0), y = output.getOperand(1);

  auto [name, n] = mod.insertInput(1, "n", builder.getI8Type());
  EXPECT_EQ(portsToString(mod), "in a, out x, in n, in b, out y");
  EXPECT_EQ(name, builder.getStringAttr("n"));
  ASSERT_EQ(body->getNumArguments(), 3u);
  EXPECT_EQ(body->getArgument(0), a);
  EXPECT_EQ(body->getArgument(1), n);
  EXPECT_EQ(body->getArgument(2), b);
  EXPECT_EQ(n.getType(), builder.getI8Type());
  EXPECT_EQ(mod.getInputTypes()[1], builder.getI8Type());

  mod.insertOutputs(1, {{builder.getStringAttr("z"), n}});
  EXPECT_EQ(portsToString(mod), "in a, out x, in n, in b, out z, out y");
  EXPECT_EQ(SmallVector<Value>(output->getOperands()),
            (SmallVector<Value>{x, n, y}));
  EXPECT_EQ(mod.getOutputTypes()[1], builder.getI8Type());
  EXPECT_TRUE(succeeded(verify(*top)));
}

} // namespace
