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

// Macros for C-style error handling
#define NO_LID -1UL

namespace {
// The goal here is to traverse the operations in order and convert them one by one into btor2
struct LowerHWtoBTOR2Pass : public LowerHWtoBTOR2Base<LowerHWtoBTOR2Pass> {
  public:
    // Executes the pass
    void runOnOperation() override;

  private:
    // Create a counter that attributes a unique id to each generated btor2 line
    size_t lid = 1; // btor2 line identifiers usually start at 1

    // Create maps to keep track of lid associations
    // Proper maps wouldn't work for some reason so I'll use a vector of pairs instead
    llvm::SmallVector<std::pair<size_t, size_t>> sortToLIDMap; // Keeps track of the ids associated to each declared sort 
    llvm::SmallVector<std::pair<std::string, size_t>> nameLIDMap; // Connects a variable name to it's most recent update line

    // Set of often reused strings in btor2 emission (to avoid typos and enable auto-complete)
    const std::string SORT    = "sort";
    const std::string BITVEC  = "bitvec";
    const std::string INPUT   = "input";
    const std::string OUTPUT  = "output";
    const std::string ZERO    = "zero";
    const std::string ONE     = "one";
    const std::string CONST   = "const";
    const std::string CONSTD  = "constd";
    const std::string CONSTH  = "consth";
    const std::string SLICE   = "slice";
    const std::string UEXT    = "uext";
    const std::string ADD     = "add";
    const std::string CONCAT  = "concat";
    const std::string NOT     = "not";
    const std::string BAD     = "bad";
    const std::string WS      = " "; // WhiteSpace
    const std::string NL      = "\n"; // NewLine

    /// Field helper functions

    // Checks if a sort was declared with the given width
    // If so, its lid will be returned
    // Otherwise -1 will be returned
    size_t getSortLID(size_t w) {
      // Look for presence of the width in a pair
      for(auto p : sortToLIDMap) {
        // If the width was declared, return it's lid
        if(w == p.first) {
          return p.second;
        }
      }
      // If no lid was found return -1
      return NO_LID;
    }

    // Updates or creates an entry for the given width
    // associating it with the current lid
    void setSortLID(size_t w) {
      // Check for the existence of the sort
      for(size_t i = 0; i < sortToLIDMap.size(); ++i) {
        auto p = sortToLIDMap[i];

        // If the width was declared, update it's lid
        if(w == p.first) {
          sortToLIDMap[i] = std::make_pair<size_t, size_t>(
            std::forward<size_t>(w), 
            std::forward<size_t>(lid)
          );
          return;
        }
      }

        // Otherwise simply create a new entry 
        sortToLIDMap.push_back(std::make_pair<size_t, size_t>(
          std::forward<size_t>(w), 
          std::forward<size_t>(lid)
        ));
    }

    // Checks if a name was declared with the given width
    // If so, its lid will be returned
    // Otherwise -1 will be returned
    size_t getNameLID(std::string n) {
      // Look for presence of the width in a pair
      for(auto p : nameLIDMap) {
        // If the width was declared, return it's lid
        if(n == p.first) {
          return p.second;
        }
      }
      // If no lid was found return -1
      return NO_LID;
    }

    // Updates or creates an entry for the given name
    // associating it with the current lid
    void setNameLID(std::string n) {
      // Check for the existence of the sort
      for(size_t i = 0; i < nameLIDMap.size(); ++i) {
        auto p = nameLIDMap[i];

        // If the width was declared, update it's lid
        if(n == p.first) {
          nameLIDMap[i] = std::make_pair<std::string, size_t>(n.c_str(), std::forward<size_t>(lid));
          return;
        }
      }

        // Otherwise simply create a new entry 
        nameLIDMap.push_back(std::make_pair<std::string, size_t>(n.c_str(), std::forward<size_t>(lid)));
    }

    /// String generation helper functions

    // Generates a sort declaration instruction given a type (bitvec or array) and a width
    std::string genSort(std::string type, size_t width) {
      // Check that the sort wasn't already declared
      if(getSortLID(width) != NO_LID) {
        return ""; // If it has already been declared then return an empty string
      }

      // Add the width to the declared sorts along with the associated line id
      setSortLID(width);

      // Build and return a sort declaration
      return std::to_string(lid++) + WS + SORT + WS + BITVEC + WS + std::to_string(width) + NL; 
    }

    // Generates an input declaration given a sort lid and a name
    std::string genInput(size_t sortLid, std::string name) {
      // Sanity check: The given lid must be associated to a sort declaration
      bool found = false;
      for(auto entry : sortToLIDMap) {
        // Iterate until we've found our association
        if((found = (entry.second == sortLid))) break; 
      }

      // Check that a result was found before continuing
      assert(found);

      // Register the input name
      setNameLID(name);

      // Generate input declaration
      return std::to_string(lid++) + WS + INPUT + WS + std::to_string(sortLid) + WS + name + NL;
    }

    // Generates a constant declaration given a value, a width and a name
    std::string genConst(uint64_t value, size_t width, std::string name) {
      // For now we're going to assume that the name isn't taken, given that hw is already in SSA form
      setNameLID(name);

      // Retrieve the lid associated with the sort (sid)
      size_t sid = getSortLID(width);

      // Check that a result was found before continuing
      assert(sid != NO_LID);

      return std::to_string(lid++) + WS + CONSTD + WS + std::to_string(sid) + WS + std::to_string(value) + NL;
    }


};
} // end anonymous namespace

void LowerHWtoBTOR2Pass::runOnOperation() {
  // String used to build out our emitted btor2
  std::string btor2Res; 

  // Start by checking for each module in the circt, for now we only consider the 1st one
  // As we are not support multi-modules yet. We assume that no nested modules exist at this point.
  // This simplifies lowering to btor as btor does not support the concept of module instanciation.
  getOperation().walk([&](hw::HWModuleOp module) {
    
    // Start by extracting the inputs and outputs from the module
    SmallVector<PortInfo> inputs, outputs;
    for(auto &port : module.getPortList()) {
      // Separate the inputs from outputs and generate the first btor2 lines for input declaration
      // We only consider ports with an explicit bit-width for now (so ignore clocks)
      if (port.isInput() && port.type.isIntOrFloat()) {
        inputs.push_back(port);
        
        // Generate the associated btor declaration for the inputs
        std::string iName = port.getName().str();  // Start by retrieving the name

        // Then only retrieve the width if the input is not a clock
        size_t width = port.type.getIntOrFloatBitWidth();

        // Generate the input and sort declaration using the extracted information
        btor2Res += genSort(BITVEC, width); // We assume all sorts are bitvectors for now
        btor2Res += genInput(getSortLID(width), iName);
      
      } else if (port.isOutput()) {
        outputs.push_back(port);
      }

    }

    // Go over all operations in our module
    module.walk([&](Operation* op) {
      // Pattern match the operation to figure out what type it is
      llvm::TypeSwitch<Operation*, void>(op) 
        // Constants are directly mapped to the btor2 constants
        .Case<hw::ConstantOp>([&](hw::ConstantOp op) {
          // Start by figuring out what sort needs to be generated
          int64_t width = hw::getBitWidth(op.getType());

          // Generate the sort (nothing will be added if the sort already exists)
          btor2Res += genSort(BITVEC, width);

          // Extract the associated constant value
          uint64_t value = dyn_cast<uint64_t, APInt>(op.getValue());

          // Extract the name from the op
          auto name = op.getResult();

          //TODO: Figure out how to extract the string used to define the result
          // e.g. extract "result" from  %result = hw.constant 42 : t1

        })
        // Wires can generally be ignored in bto2, however we do need
        // to keep track of the new alias it creates
        .Case<hw::WireOp>([&](hw::WireOp op) {
        })
        // Outputs map to btor2 outputs on their final assignment
        .Case<hw::OutputOp>([&](hw::OutputOp op) {
        })
        // Supported Comb operations
        // Concat operations can directly be mapped to btor2 concats
        .Case<comb::ConcatOp>([&](comb::ConcatOp op) {
        })
        .Case<comb::AddOp>([&](comb::AddOp op) {
        })
        .Case<comb::ExtractOp>([&](comb::ExtractOp op) {
        })
        .Case<comb::ICmpOp>([&](comb::ICmpOp op) {
        })
        // Supported SV Operations
        /*.Case<sv::AssertOp>([&](auto op) {
          llvm::errs() << "This is an assertion" << "\n";
        })*/
        // All other operations should be ignored for the time being
        .Default([](auto) {});
    });

    // Print out the resuling btor2
    llvm::errs() << "==========BTOR2 FORM:==========\n" 
                 << btor2Res 
                 << "\n===============================\n";

  });
}

// Basic constructor for the pass
std::unique_ptr<mlir::Pass> circt::hw::createLowerHWtoBTOR2Pass() {
  return std::make_unique<LowerHWtoBTOR2Pass>();
}
