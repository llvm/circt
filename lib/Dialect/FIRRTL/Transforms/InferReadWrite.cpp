//===- InferReadWrite.cpp - Infer Read Write Memory -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InferReadWrite pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"
#include <set>

#define DEBUG_TYPE "firrtl-infer-read-write"

using namespace circt;
using namespace firrtl;

namespace {
struct InferReadWritePass : public InferReadWriteBase<InferReadWritePass> {

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "\n Running Infer Read Write on module:"
                            << getOperation().getName());
    ModuleNamespace modNamespace(getOperation());
    for (MemOp mem : llvm::make_early_inc_range(
             getOperation().getBody()->getOps<MemOp>())) {
      size_t nReads, nWrites, nRWs;
      mem.getNumPorts(nReads, nWrites, nRWs);
      // Run the analysis only for Seq memories (latency=1) and a single read
      // and write ports.
      if (!(nReads == 1 && nWrites == 1 && nRWs == 0) ||
          !(mem.readLatency() == 1 && mem.writeLatency() == 1))
        continue;
      Value rClock, wClock;
      // The memory has exactly two ports.
      SmallVector<Value> portTerms[2];
      for (auto portIt : llvm::enumerate(mem.results())) {
        // Get the port value.
        Value portVal = portIt.value();
        // Get the port kind.
        bool readPort =
            mem.getPortKind(portIt.index()) == MemOp::PortKind::Read;
        // Iterate over all users of the port.
        for (Operation *u : portVal.getUsers())
          if (auto sf = dyn_cast<SubfieldOp>(u)) {
            // Get the field name.
            auto fName = sf.input().getType().cast<BundleType>().getElementName(
                sf.fieldIndex());
            // If this is the enable field, record the product terms(the And
            // expression tree).
            if (fName.equals("en"))
              getProductTerms(sf, portTerms[portIt.index()]);
            else if (fName.equals("clk")) {
              if (readPort)
                rClock = getConnectSrc(sf);
              else
                wClock = getConnectSrc(sf);
            }
          }
        // End of loop for getting MemOp port users.
      }
      if (rClock != wClock) {
        DenseSet<Value> rClocks, wClocks;
        while (rClock) {
          rClocks.insert(rClock);
          rClock = getConnectSrc(rClock);
        }

        bool sameClock = false;
        while (wClock) {
          if (rClocks.find(wClock) != rClocks.end()) {
            sameClock = true;
            break;
          }
          wClock = getConnectSrc(wClock);
        }
        if (!sameClock)
          continue;
        rClock = wClock;
      }
      LLVM_DEBUG(
          llvm::dbgs() << "\n read clock:" << rClock
                       << " --- write clock:" << wClock;
          llvm::dbgs() << "\n Read terms==>"; for (auto t
                                                   : portTerms[0]) llvm::dbgs()
                                              << "\n term::" << t;

          llvm::dbgs() << "\n Write terms==>"; for (auto t
                                                    : portTerms[1]) llvm::dbgs()
                                               << "\n term::" << t;

      );
      // If the read and write clocks are the same and the product terms have a
      // complement of each other.
      if (checkComplement(portTerms)) {
        SmallVector<Attribute, 4> resultNames;
        SmallVector<Type, 4> resultTypes;
        SmallVector<Attribute, 4> portAnnotations;
        // There is only a single port, the rw.
        resultNames.push_back(
            StringAttr::get(modNamespace.newName("rw"), mem.getContext()));
        // Set the type of the rw port.
        resultTypes.push_back(MemOp::getTypeForPort(
            mem.depth(), mem.getDataType(), MemOp::PortKind::ReadWrite,
            mem.getMaskBits()));
        ImplicitLocOpBuilder builder(mem.getLoc(), mem);
        SmallVector<Attribute> portAtts;
        // Append the annotations from the two ports.
        if (!mem.portAnnotations()[0].cast<ArrayAttr>().empty())
          portAtts.push_back(mem.portAnnotations()[0]);
        if (!mem.portAnnotations()[1].cast<ArrayAttr>().empty())
          portAtts.push_back(mem.portAnnotations()[1]);
        portAnnotations.push_back(builder.getArrayAttr(portAtts));
        // Create the new rw memory.
        auto rwMem = builder.create<MemOp>(
            resultTypes, mem.readLatency(), mem.writeLatency(), mem.depth(),
            RUWAttr::Undefined, builder.getArrayAttr(resultNames),
            mem.nameAttr(), mem.annotations(),
            builder.getArrayAttr(portAnnotations), mem.inner_symAttr());
        auto rwPort = rwMem.getResult(0);
        // Create the subfield access to all fields of the port.
        // The addr should be connected to read/write address depending on the
        // read/write mode.
        auto addr = builder.create<SubfieldOp>(rwPort, "addr");
        // Enable is high whenever the memory is written or read.
        auto enb = builder.create<SubfieldOp>(rwPort, "en");
        // Read/Write clock.
        auto clk = builder.create<SubfieldOp>(rwPort, "clk");
        auto readData = builder.create<SubfieldOp>(rwPort, "rdata");
        // wmode is high when the port is in write mode. That is this can be
        // connected to the write enable.
        auto wmode = builder.create<SubfieldOp>(rwPort, "wmode");
        auto writeData = builder.create<SubfieldOp>(rwPort, "wdata");
        auto mask = builder.create<SubfieldOp>(rwPort, "wmask");
        // Temp wires to replace the original memory connects.
        auto rAddr = builder.create<WireOp>(addr.getType(), "readAddr");
        auto wAddr = builder.create<WireOp>(addr.getType(), "writeAddr");
        auto wEnWire = builder.create<WireOp>(enb.getType(), "writeEnable");
        auto rEnWire = builder.create<WireOp>(enb.getType(), "readEnable");
        // addr = Mux(WriteEnable, WriteAddress, ReadAddress).
        builder.create<ConnectOp>(
            addr, builder.create<MuxPrimOp>(wEnWire, wAddr, rAddr));
        // Enable = Or(WriteEnable, ReadEnable).
        builder.create<ConnectOp>(enb,
                                  builder.create<OrPrimOp>(rEnWire, wEnWire));
        // WriteMode = WriteEnable.
        builder.create<ConnectOp>(wmode, wEnWire);
        // Now iterate over the original memory read and write ports.
        for (auto portIt : llvm::enumerate(mem.results())) {
          // Get the port value.
          Value portVal = portIt.value();
          // Get the port kind.
          bool readPort =
              mem.getPortKind(portIt.index()) == MemOp::PortKind::Read;
          // Iterate over all users of the port, which are the subfield ops, and
          // replace them.
          for (Operation *u : portVal.getUsers())
            if (auto sf = dyn_cast<SubfieldOp>(u)) {
              auto fName =
                  sf.input().getType().cast<BundleType>().getElementName(
                      sf.fieldIndex());
              if (fName.equals("en")) {
                if (readPort)
                  sf->replaceAllUsesWith(rEnWire);
                else
                  sf->replaceAllUsesWith(wEnWire);
              } else if (fName.equals("clk")) {
                if (readPort)
                  sf->replaceAllUsesWith(clk);
                else {
                  auto dummyWire = builder.create<WireOp>(sf.getType());
                  sf->replaceAllUsesWith(dummyWire);
                }
              } else if (fName.equals("addr")) {
                if (readPort)
                  sf->replaceAllUsesWith(rAddr);
                else
                  sf->replaceAllUsesWith(wAddr);
              } else if (fName.equals("data")) {
                if (readPort)
                  sf->replaceAllUsesWith(readData);
                else
                  sf->replaceAllUsesWith(writeData);
              } else if (fName.equals("mask")) {
                if (!readPort)
                  sf->replaceAllUsesWith(mask);
              }
              // Once all the uses of the subfield op replaced, delete it.
              sf.erase();
            }
        }
        // All uses for all results of mem removed, now erase the mem.
        mem.erase();
      }
    }
  }

private:
  Value getConnectSrc(Value dst) {
    for (auto c : dst.getUsers())
      if (auto connect = dyn_cast<ConnectOp>(c))
        if (connect.dest() == dst)
          return connect.src();

    return nullptr;
  }

  void getProductTerms(Value a, SmallVector<Value> &terms) {
    if (!a)
      return;
    terms.push_back(a);
    if (a.isa<BlockArgument>())
      return;
    if (auto node = dyn_cast<NodeOp>(a.getDefiningOp()))
      return getProductTerms(node.input(), terms);

    if (auto src = getConnectSrc(a))
      return getProductTerms(src, terms);

    if (auto op = a.getDefiningOp()) {
      if (auto andOp = dyn_cast<AndPrimOp>(op)) {
        for (auto i : op->getOperands())
          getProductTerms(i, terms);
        return;
      } else if (auto muxOp = dyn_cast<MuxPrimOp>(op)) {
        if (ConstantOp cLow =
                dyn_cast_or_null<ConstantOp>(muxOp.low().getDefiningOp()))
          if (cLow.value().isZero()) {
            getProductTerms(muxOp.sel(), terms);
            getProductTerms(muxOp.high(), terms);
            return;
          }
      }
    }
  }

  bool checkComplement(SmallVector<Value> prodTerms[2]) {
    bool isComplement = false;
    for (auto t1 : prodTerms[0])
      for (auto t2 : prodTerms[1]) {
        if (!t1.isa<BlockArgument>() && isa<NotPrimOp>(t1.getDefiningOp()))
          if (cast<NotPrimOp>(t1.getDefiningOp()).input() == t2)
            return true;

        if (!t2.isa<BlockArgument>() && isa<NotPrimOp>(t2.getDefiningOp()))
          if (cast<NotPrimOp>(t2.getDefiningOp()).input() == t1)
            return true;
      }

    return isComplement;
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createInferReadWritePass() {
  return std::make_unique<InferReadWritePass>();
}
