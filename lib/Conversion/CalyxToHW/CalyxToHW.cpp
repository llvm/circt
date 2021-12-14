//===- CalyxToHW.cpp - Translate Calyx into HW ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Calyx to HW Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CalyxToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;
using namespace circt;
using namespace circt::calyx;
using namespace circt::hw;
using namespace circt::sv;

hw::HWModuleExternOp circt::calyx::getExternHWModule(OpBuilder &builder,
                                                     ComponentOp op) {
  SmallVector<hw::PortInfo, 8> hwPortInfos;
  auto addHWPortInfo = [&](auto portInfos, hw::PortDirection direction) {
    for (auto portInfo : enumerate(portInfos)) {
      hw::PortInfo hwPortInfo;
      hwPortInfo.direction = direction;
      hwPortInfo.type = portInfo.value().type;
      hwPortInfo.argNum = portInfo.index();
      hwPortInfo.name = portInfo.value().name;
      hwPortInfos.push_back(hwPortInfo);
    }
  };

  addHWPortInfo(op.getInputPortInfo(), hw::PortDirection::INPUT);
  addHWPortInfo(op.getOutputPortInfo(), hw::PortDirection::OUTPUT);

  return builder.create<hw::HWModuleExternOp>(
      op->getLoc(), builder.getStringAttr(op.getName()), hwPortInfos);
}

namespace {
class CalyxToHWPass : public CalyxToHWBase<CalyxToHWPass> {
public:
  void runOnOperation() override;
};
} // end anonymous namespace

void CalyxToHWPass::runOnOperation() {
  // auto op = getOperation();
}

std::unique_ptr<mlir::Pass> circt::createCalyxToHWPass() {
  return std::make_unique<CalyxToHWPass>();
}
