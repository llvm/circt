//===- InferReadWrite.cpp - Infer Read Write Memory -----------------------===//
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
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

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
      if (!sameDriver(rClock, wClock))
        continue;

      rClock = wClock;
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
      // If the read and write clocks are the same, check if any of the product
      // terms are a complement of each other.
      if (!checkComplement(portTerms))
        continue;

      SmallVector<Attribute, 4> resultNames;
      SmallVector<Type, 4> resultTypes;
      SmallVector<Attribute, 4> portAnnotations;
      // Create the merged rw port for the new memory.
      resultNames.push_back(
          StringAttr::get(mem.getContext(), modNamespace.newName("rw")));
      // Set the type of the rw port.
      resultTypes.push_back(
          MemOp::getTypeForPort(mem.depth(), mem.getDataType(),
                                MemOp::PortKind::ReadWrite, mem.getMaskBits()));
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
          RUWAttr::Undefined, builder.getArrayAttr(resultNames), mem.nameAttr(),
          mem.annotations(), builder.getArrayAttr(portAnnotations),
          mem.inner_symAttr());
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
      auto writeClock =
          builder.create<WireOp>(ClockType::get(enb.getContext()));
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
            StringRef fName =
                sf.input().getType().cast<BundleType>().getElementName(
                    sf.fieldIndex());
            Value repl;
            if (readPort)
              repl = llvm::StringSwitch<Value>(fName)
                         .Case("en", rEnWire)
                         .Case("clk", clk)
                         .Case("addr", rAddr)
                         .Case("data", readData);
            else
              repl = llvm::StringSwitch<Value>(fName)
                         .Case("en", wEnWire)
                         .Case("clk", writeClock)
                         .Case("addr", wAddr)
                         .Case("data", writeData)
                         .Case("mask", mask);
            sf.replaceAllUsesWith(repl);
            // Once all the uses of the subfield op replaced, delete it.
            sf.erase();
          }
      }
      // All uses for all results of mem removed, now erase the mem.
      mem.erase();
    }
  }

private:
  // Get the source value which is connected to the dst.
  Value getConnectSrc(Value dst) {
    for (auto *c : dst.getUsers())
      if (auto connect = dyn_cast<ConnectOp>(c))
        if (connect.dest() == dst)
          return connect.src();

    return nullptr;
  }

  /// If the ports are not directly connected to the same clock, then check
  /// if indirectly connected to the same clock.
  bool sameDriver(Value rClock, Value wClock) {
    if (rClock == wClock)
      return true;
    DenseSet<Value> rClocks, wClocks;
    while (rClock) {
      // Record all the values which are indirectly connected to the clock
      // port.
      rClocks.insert(rClock);
      rClock = getConnectSrc(rClock);
    }

    bool sameClock = false;
    // Now check all the indirect connections to the write memory clock
    // port.
    while (wClock) {
      if (rClocks.find(wClock) != rClocks.end()) {
        sameClock = true;
        break;
      }
      wClock = getConnectSrc(wClock);
    }
    return sameClock;
  }

  void getProductTerms(Value enValue, SmallVector<Value> &terms) {
    if (!enValue)
      return;
    SmallVector<Value> worklist;
    worklist.push_back(enValue);
    while (!worklist.empty()) {
      auto term = worklist.back();
      worklist.pop_back();
      terms.push_back(term);
      if (term.isa<BlockArgument>())
        continue;
      TypeSwitch<Operation *>(term.getDefiningOp())
          .Case<NodeOp>([&](auto n) { worklist.push_back(n.input()); })
          .Case<AndPrimOp>([&](AndPrimOp andOp) {
            worklist.push_back(andOp.getOperand(0));
            worklist.push_back(andOp.getOperand(1));
          })
          .Case<MuxPrimOp>([&](auto muxOp) {
            // Check for the pattern when low is 0, which is equivalent to (sel
            // & high)
            // term = mux (sel, high, 0) => term = sel & high
            if (ConstantOp cLow =
                    dyn_cast_or_null<ConstantOp>(muxOp.low().getDefiningOp()))
              if (cLow.value().isZero()) {
                worklist.push_back(muxOp.sel());
                worklist.push_back(muxOp.high());
              }
          })
          .Default([&](auto) {
            if (auto src = getConnectSrc(term))
              worklist.push_back(src);
          });
    }
  }

  /// Check if any of the terms in the prodTerms[0] is a complement of any of
  /// the terms in prodTerms[1]. prodTerms[0], prodTerms[1] is a vector of
  /// Value, each of which correspond to the two product terms of read/write
  /// enable.
  bool checkComplement(SmallVector<Value> prodTerms[2]) {
    bool isComplement = false;
    // Foreach Value in first term, check if it is the complement of any of the
    // Value in second term.
    for (auto t1 : prodTerms[0])
      for (auto t2 : prodTerms[1]) {
        // Return true if t1 is a Not of t2.
        if (!t1.isa<BlockArgument>() && isa<NotPrimOp>(t1.getDefiningOp()))
          if (cast<NotPrimOp>(t1.getDefiningOp()).input() == t2)
            return true;
        // Else Return true if t2 is a Not of t1.
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
