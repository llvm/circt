// RUN: circt-verilog --import-only --top=top %s | FileCheck %s
// RUN: circt-verilog --ir-moore --top=top %s > /dev/null
// RUN: circt-verilog --import-only --top=MinimalTop %s | FileCheck %s --check-prefix=MINIMAL
// REQUIRES: slang
// UNSUPPORTED: valgrind

interface bus;
  logic valid;
  wire ready;
  struct packed { logic [7:0] pc; logic tag; } payload;
endinterface

// Expanded interface members must be exported as references, including nets
// and entire packed structs. Two interfaces with identical member names must
// resolve independently. Local interface reads and ordinary hierarchical
// outputs continue to work alongside the generated interface outputs.
// CHECK-LABEL: moore.module private @core(
// CHECK-SAME: out command.valid : !moore.ref<l1>, out other.valid : !moore.ref<l1>, out command.ready : !moore.ref<l1>, out command.payload : !moore.ref<struct<{pc: l8, tag: l1}>>, out ordinary : !moore.ref<l1>
module core(input logic v, output logic local_valid);
  bus command();
  bus other();
  logic ordinary;
  // CHECK: %ordinary = moore.variable : <l1>
  // CHECK: %command_valid = moore.variable : <l1>
  // CHECK: %command_ready = moore.net wire : <l1>
  // CHECK: %command_payload = moore.variable : <struct<{pc: l8, tag: l1}>>
  // CHECK: %other_valid = moore.variable : <l1>
  assign command.valid = v;
  assign command.ready = !v;
  assign command.payload = {8'h42, v};
  assign other.valid = !v;
  assign ordinary = v;
  assign local_valid = command.valid;
  // CHECK: %[[LOCAL:.*]] = moore.read %command_valid : <l1>
  // CHECK: moore.assign %local_valid, %[[LOCAL]] : l1
  // CHECK: moore.output %{{[^,]+}}, %command_valid, %other_valid, %command_ready, %command_payload, %ordinary : !moore.l1, !moore.ref<l1>, !moore.ref<l1>, !moore.ref<l1>, !moore.ref<struct<{pc: l8, tag: l1}>>, !moore.ref<l1>
endmodule

// The minimal scalar case, referenced only through the second sibling. This
// also exercises member-name fallback when Slang canonicalizes the first
// sibling's body but the observed member belongs to the second one.
interface minimal_bus;
  logic valid;
endinterface

// MINIMAL-LABEL: moore.module private @MinimalCore(out command.valid : !moore.ref<l1>)
module MinimalCore;
  minimal_bus command();
  assign command.valid = 1'b1;
  // MINIMAL: %command_valid = moore.variable : <l1>
  // MINIMAL: moore.assign %command_valid, %{{.*}} : l1
  // MINIMAL: moore.output %command_valid : !moore.ref<l1>
endmodule

// MINIMAL-LABEL: moore.module @MinimalTop(out q : !moore.l1)
module MinimalTop(output logic q);
  MinimalCore first();
  MinimalCore second();
  assign q = second.command.valid;
  // MINIMAL: %first.command.valid = moore.instance "first" @MinimalCore() -> (command.valid: !moore.ref<l1>)
  // MINIMAL: %second.command.valid = moore.instance "second" @MinimalCore() -> (command.valid: !moore.ref<l1>)
  // MINIMAL: %[[READ:.*]] = moore.read %second.command.valid : <l1>
  // MINIMAL: moore.assign %q, %[[READ]] : l1
endmodule

// CHECK-LABEL: moore.module private @middle(
// CHECK-SAME: out inst.command.valid : !moore.ref<l1>
module middle(input logic v);
  core inst(v, );
  // CHECK: %inst.local_valid, %inst.command.valid, {{.*}} = moore.instance "inst" @core(
  // CHECK: moore.output %inst.command.valid : !moore.ref<l1>
endmodule

// Sibling instances share a module definition but have different values. Check
// the actual instance results used by each read, not just successful import.
// CHECK-LABEL: moore.module @top(
module top(output logic a, b, c, r, output logic [7:0] pc,
           output logic d, e, f);
  core first(1'b0, );
  core second(1'b1, );
  middle mid(1'b1);
  // CHECK: %[[ZERO:.*]] = moore.constant 0 : l1
  // CHECK: %first.local_valid, %first.command.valid, %first.other.valid, {{.*}} = moore.instance "first" @core(v: %[[ZERO]]: !moore.l1)
  // CHECK: %[[ONE:.*]] = moore.constant 1 : l1
  // CHECK: %second.local_valid, %second.command.valid, %second.other.valid, %second.command.ready, %second.command.payload, %second.ordinary = moore.instance "second" @core(v: %[[ONE]]: !moore.l1)
  // CHECK: %mid.inst.command.valid = moore.instance "mid" @middle(
  assign a = first.command.valid;
  // CHECK: %[[A:.*]] = moore.read %first.command.valid : <l1>
  // CHECK: moore.assign %a, %[[A]] : l1
  assign b = second.command.valid;
  // CHECK: %[[B:.*]] = moore.read %second.command.valid : <l1>
  // CHECK: moore.assign %b, %[[B]] : l1
  assign c = second.other.valid;
  // CHECK: %[[C:.*]] = moore.read %second.other.valid : <l1>
  // CHECK: moore.assign %c, %[[C]] : l1
  assign r = second.command.ready;
  // CHECK: %[[R:.*]] = moore.read %second.command.ready : <l1>
  // CHECK: moore.assign %r, %[[R]] : l1
  assign pc = second.command.payload.pc;
  // CHECK: %[[PAYLOAD:.*]] = moore.read %second.command.payload : <struct<{pc: l8, tag: l1}>>
  // CHECK: %[[PC:.*]] = moore.struct_extract %[[PAYLOAD]], "pc" : struct<{pc: l8, tag: l1}> -> l8
  // CHECK: moore.assign %pc, %[[PC]] : l8
  assign d = mid.inst.command.valid;
  // CHECK: %[[D:.*]] = moore.read %mid.inst.command.valid : <l1>
  // CHECK: moore.assign %d, %[[D]] : l1
  assign e = second.ordinary;
  // CHECK: %[[E:.*]] = moore.read %second.ordinary : <l1>
  // CHECK: moore.assign %e, %[[E]] : l1
  assign f = first.other.valid;
  // CHECK: %[[F:.*]] = moore.read %first.other.valid : <l1>
  // CHECK: moore.assign %f, %[[F]] : l1
endmodule
