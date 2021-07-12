//===- HandshakeInterfaces.td - Handshake interfaces -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the structured interface for Handshake ops.
//
//===----------------------------------------------------------------------===//

#ifdef HANDSHAKE_INTERFACES
#else
#define HANDSHAKE_INTERFACES

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def MergeLikeOpInterface : OpInterface<"MergeLikeOpInterface"> {
  let description = [{
     Some handshake operations can have predecessors in other
     blocks. This is primarily useful for verification purposes during
     lowering from other dialect, such as the standard CDFG dialect.
  }];

  let methods = [
    InterfaceMethod<[{
        Returns an operand range over the data signals being merged.
      }],
      "OperandRange", "dataOperands", (ins)
    >,
  ];

  let verify = [{
    auto concreteOp = cast<ConcreteOp>($_op);

    auto operands = concreteOp.dataOperands();

    if (!operands.size())
      return concreteOp.emitOpError("must have at least one data operand");

    Type resultType = $_op->getResult(0).getType();

    for (auto operand : operands)
      if (operand.getType() != resultType)
        return concreteOp.emitOpError("operand has type ") << operand.getType()
            << ", but result has type " << resultType;

    return success();
  }];
}

def GeneralOpInterface : OpInterface<"GeneralOpInterface"> {
  let description =
      [{"Simulate the Execution of ops. The op takes a set of input values and "
        "returns the corresponding outputs assuming the precondition to "
        "execute holds."}];

  let methods = [
    InterfaceMethod<
        "Simulate the Execution of the general op with given inputs", "void",
        "execute",
        (ins "std::vector<llvm::Any> &" : $ins,
           "std::vector<llvm::Any> &" : $outs)>,
  ];
}

def ExecutableOpInterface : OpInterface<"ExecutableOpInterface"> {
  let description = [{"Simulate the Execution of ops"}];

  let methods = [
    InterfaceMethod<
        "The op checks its precondition to execute. If the precondition holds, "
        "the op executes with the given input values and returns zero. "
        "Otherwise, it returns a false value to suggest the simulator to "
        "reschedule. The execution interface of hpx ops take the following "
        "arguments: "
        "bool = returns whether the op has been executed;"
        "valueMap = a map of all values in the code;"
        "memoryMap = a map of memory ops to simualte;"
        "timeMap = a map of the last arrival time of all values;"
        "store = The store associates each allocation in the program"
        "(represented by a int) with a vector of values which can be"
        "accessed by it."
        "scheduleList = a list of values to be scheduled.",
        "bool", "tryExecute",
        (ins "llvm::DenseMap<mlir::Value, llvm::Any> &" : $valueMap,
           "llvm::DenseMap<unsigned, unsigned> &" : $memoryMap,
           "llvm::DenseMap<mlir::Value, double> &" : $timeMap,
           "std::vector<std::vector<llvm::Any>> &" : $store,
           "std::vector<mlir::Value> &" : $scheduleList)>,
  ];
}

def MemoryOpInterface : OpInterface<"MemoryOpInterface"> {
  let description =
      [{"Allocate the memory to the meory map in the simulation. "}];

  let methods = [
    InterfaceMethod<
        "Simulate the memory allocation in the memoryMap", "bool",
        "allocateMemory",
        (ins "llvm::DenseMap<unsigned, unsigned> &" : $memoryMap,
    "std::vector<std::vector<llvm::Any>> &" : $store,
    "std::vector<double> &" : $storeTimes)>,
  ];
}

def HasClock : NativeOpTrait<"HasClock">;

#endif // HANDSHAKE_INTERFACES
