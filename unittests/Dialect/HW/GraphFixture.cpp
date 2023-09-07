//===- GraphFixutre.cpp - A fixture for instance graph unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GraphFixture.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
using namespace circt;
using namespace hw;

mlir::ModuleOp fixtures::createModule(MLIRContext *context) {
  context->loadDialect<HWDialect>();

  // Build the following graph:
  // hw.module @Top() {
  //   hw.instance "alligator" @Alligator() -> ()
  //   hw.instance "cat" @Cat() -> ()
  // }
  // hw.module private @Alligator() {
  //   hw.instance "bear" @Bear() -> ()
  // }
  // hw.module private @Bear() {
  //   hw.instance "cat" @Cat() -> ()
  // }
  // hw.module private @Cat() { }

  LocationAttr loc = UnknownLoc::get(context);
  auto circuit = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, circuit.getBody());

  auto top = builder.create<HWModuleOp>(StringAttr::get(context, "Top"),
                                        ArrayRef<PortInfo>{});
  auto alligator = builder.create<HWModuleOp>(
      StringAttr::get(context, "Alligator"), ArrayRef<PortInfo>{});
  auto bear = builder.create<HWModuleOp>(StringAttr::get(context, "Bear"),
                                         ArrayRef<PortInfo>{});
  auto cat = builder.create<HWModuleOp>(StringAttr::get(context, "Cat"),
                                        ArrayRef<PortInfo>{});

  alligator.setVisibility(SymbolTable::Visibility::Private);
  bear.setVisibility(SymbolTable::Visibility::Private);
  cat.setVisibility(SymbolTable::Visibility::Private);

  builder.setInsertionPointToStart(top.getBodyBlock());
  builder.create<InstanceOp>(alligator, "alligator", ArrayRef<Value>{});
  builder.create<InstanceOp>(cat, "cat", ArrayRef<Value>{});

  builder.setInsertionPointToStart(alligator.getBodyBlock());
  builder.create<InstanceOp>(bear, "bear", ArrayRef<Value>{});

  builder.setInsertionPointToStart(bear.getBodyBlock());
  builder.create<InstanceOp>(cat, "cat", ArrayRef<Value>{});

  return circuit;
}
