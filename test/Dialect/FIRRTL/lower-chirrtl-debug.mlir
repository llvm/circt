// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-chirrtl)))'  %s | FileCheck %s

// Verify that dbg.* ops whose operands reference chirrtl.memoryport data
// results survive lower-chirrtl with those operands remapped to the
// corresponding firrtl.mem subfield results.

// ---- Test 1: dbg.variable referencing a CHIRRTL memoryport data result ----

// CHECK-LABEL: firrtl.module @DebugMemPort
firrtl.circuit "DebugMemPort" {
  firrtl.module @DebugMemPort(in %clock: !firrtl.clock,
                               in %addr: !firrtl.uint<8>,
                               in %en: !firrtl.uint<1>) {
    %ram = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 256>

    %rdata, %rport = chirrtl.memoryport Infer %ram {name = "rport"}
        : (!chirrtl.cmemory<uint<8>, 256>) -> (!firrtl.uint<8>, !chirrtl.cmemoryport)
    chirrtl.memoryport.access %rport[%addr], %clock
        : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock

    // dbg.variable references the chirrtl memoryport data result (%rdata).
    // After the pass, the operand must be remapped to the firrtl.mem data
    // subfield -- the variable must still be present.
    // CHECK: dbg.variable "rdata", %{{.*}} : !firrtl.uint<8>
    dbg.variable "rdata", %rdata : !firrtl.uint<8>
  }
}

// ---- Test 2: dbg.subfield referencing a CHIRRTL memoryport data result ----
// dbg.subfield (like dbg.variable) consumes an SSA value; its operand must
// also be remapped when it references a CHIRRTL port data result.

// CHECK-LABEL: firrtl.module @DebugSubfieldMemPort
firrtl.circuit "DebugSubfieldMemPort" {
  firrtl.module @DebugSubfieldMemPort(in %clock: !firrtl.clock,
                                       in %addr: !firrtl.uint<8>) {
    %ram = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 256>

    %rdata, %rport = chirrtl.memoryport Infer %ram {name = "rport"}
        : (!chirrtl.cmemory<uint<8>, 256>) -> (!firrtl.uint<8>, !chirrtl.cmemoryport)
    chirrtl.memoryport.access %rport[%addr], %clock
        : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock

    // dbg.subfield references the chirrtl memoryport data result (%rdata).
    // After the pass, the operand must be remapped; the subfield op must
    // survive with a valid (non-CHIRRTL) operand.
    // CHECK: %[[SF:.+]] = dbg.subfield "rdata", %{{.*}} : !firrtl.uint<8>
    // CHECK: dbg.struct {"rdata": %[[SF]]}
    %sf = dbg.subfield "rdata", %rdata : !firrtl.uint<8>
    %st = dbg.struct {"rdata": %sf} : !dbg.subfield
    dbg.variable "mem", %st : !dbg.struct
  }
}

// ---- Test 3: non-Debug dialect op alongside a CHIRRTL memory ----
// visitInvalidOp must not touch ops from dialects other than Debug.
// The hw.constant below is from the HW dialect; even though a CHIRRTL memory
// exists in the same module, visitInvalidOp must leave it completely alone.

// CHECK-LABEL: firrtl.module @NonDebugHwOp
firrtl.circuit "NonDebugHwOp" {
  firrtl.module @NonDebugHwOp(in %clock: !firrtl.clock,
                               in %addr: !firrtl.uint<8>) {
    %ram = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 256>

    %rdata, %rport = chirrtl.memoryport Infer %ram {name = "rport"}
        : (!chirrtl.cmemory<uint<8>, 256>) -> (!firrtl.uint<8>, !chirrtl.cmemoryport)
    chirrtl.memoryport.access %rport[%addr], %clock
        : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock

    // hw.constant is dispatched through visitInvalidOp (not FIRRTL, not Debug).
    // The new branch must skip it; the constant must survive unchanged.
    // CHECK: hw.constant 42 : i8
    %c = hw.constant 42 : i8
  }
}
