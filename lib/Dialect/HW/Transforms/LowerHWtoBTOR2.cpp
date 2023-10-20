//===- LowerHWtoBTOR2.cpp - Lowers a hw module to btor2 --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Converts a hw module to a btor2 format and prints it out
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
// The goal here is to traverse the operations in order and convert them one by one into btor2
struct LowerHWtoBTOR2Pass : public LowerHWtoBTOR2Base<LowerHWtoBTOR2Pass> {

  // Executes the pass
  void runOnOperation() override;

};
} // end anonymous namespace

void LowerHWtoBTOR2Pass::runOnOperation() {
  // Create a counter that attributes a unique id to each generated btor2 line
  size_t lid = 1; // btor2 line identifiers usually start at 1

  // Start by checking for each module in the circt, for now we only consider the 1st one
  // As we are not support multi-modules yet. We assume that no nested modules exist at this point.
  // This simplifies lowering to btor as btor does not support the concept of module instanciation.
  getOperation().walk([&](hw::HWModuleOp module) {
    
    // Start by extracting the inputs and outputs from the module
    SmallVector<PortInfo> inputs, outputs;
    for(auto &port : module.getPortList()) {
      // Seperate the inputs from outputs using their directions
      if (port.dir == ModulePort::Direction::Input) {
        inputs.push_back(port);
      } else if (port.dir == ModulePort::Direction::Output){
        outputs.push_back(port);
      }
      // Show the port that was just added
      llvm::errs() << port << "\n";
    }

    // Go over all operations in our module
    module.walk([&](Operation* op) {
      // Pattern match the operation to figure out what type it is
      llvm::TypeSwitch<Operation*, void>(op) 
        // We start out by simply printing the types to make sure 
        // that we are understanding llvm pattern matching correctly
        .Case<hw::ConstantOp>([&](hw::ConstantOp op) {
          llvm::errs() << "This is a constant" << "\n";
        })
        .Case<hw::WireOp>([&](hw::WireOp op) {
          llvm::errs() << "This is the wire's result: ";
          llvm::errs() << op.getResult();
          llvm::errs() << "This is a wire" << "\n";
        })
        // Supported Comb operations
        .Case<comb::ConcatOp>([&](comb::ConcatOp op) {
          llvm::errs() << "This is a concat" << "\n";
        })
        .Case<comb::AddOp>([&](comb::AddOp op) {
          llvm::errs() << "This is an add" << "\n";
        })
        .Case<comb::ExtractOp>([&](comb::ExtractOp op) {
          llvm::errs() << "This is an extract" << "\n";
        })
        .Case<comb::ICmpOp>([&](comb::ICmpOp op) {
          llvm::errs() << "This is an icmp" << "\n";
        })
        // Supported SV Operations
        /*.Case<sv::AssertOp>([&](auto op) {
          llvm::errs() << "This is an assertion" << "\n";
        })*/
        // All other operations should be ignored for the time being
        .Default([](auto) {
          llvm::errs() << "Ignore" << "\n";
        });
    });
  });
}

// Basic constructor for the pass
std::unique_ptr<mlir::Pass> circt::hw::createLowerHWtoBTOR2Pass() {
  return std::make_unique<LowerHWtoBTOR2Pass>();
}
