// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

typedef enum logic [2:0] { X = 3'd1, Y = 3'd4, Z = 3'd6 } Sparse;

// CHECK-LABEL: func.func private @enum.next.Sparse(
// CHECK-SAME:      [[VALUE:%.+]]: !moore.l3, [[COUNT:%.+]]: !moore.i32) -> !moore.l3 {
// CHECK:         [[VAL0:%.+]] = moore.constant 1 : l3
// CHECK:         [[VAL1:%.+]] = moore.constant -4 : l3
// CHECK:         [[VAL2:%.+]] = moore.constant -2 : l3
// CHECK:         [[TABLE:%.+]] = moore.array_create [[VAL2]], [[VAL1]], [[VAL0]] : {{.+}} -> array<3 x l3>

// The value is compared against each enumerand in turn, which yields its
// position among them.
// CHECK:         [[EQ:%.+]] = moore.case_eq [[VALUE]], [[VAL0]] : l3
// CHECK:         [[COND:%.+]] = moore.to_builtin_int [[EQ]] : i1
// CHECK:         [[POS:%.+]] = moore.constant 0 : i32
// CHECK:         cf.cond_br [[COND]], ^[[MATCH:.+]]([[POS]] : !moore.i32), ^[[BB:.+]]
// CHECK:       ^[[BB]]:
// CHECK:         [[EQ:%.+]] = moore.case_eq [[VALUE]], [[VAL1]] : l3
// CHECK:         [[COND:%.+]] = moore.to_builtin_int [[EQ]] : i1
// CHECK:         [[POS:%.+]] = moore.constant 1 : i32
// CHECK:         cf.cond_br [[COND]], ^[[MATCH]]([[POS]] : !moore.i32), ^[[BB:.+]]
// CHECK:       ^[[BB]]:
// CHECK:         [[EQ:%.+]] = moore.case_eq [[VALUE]], [[VAL2]] : l3
// CHECK:         [[COND:%.+]] = moore.to_builtin_int [[EQ]] : i1
// CHECK:         [[POS:%.+]] = moore.constant 2 : i32
// CHECK:         cf.cond_br [[COND]], ^[[MATCH]]([[POS]] : !moore.i32), ^[[BB:.+]]

// Values that are not a member of the enumeration produce the default value,
// which is all-X for a four-valued enum.
// CHECK:       ^[[BB]]:
// CHECK:         [[DEFAULT:%.+]] = moore.constant bXXX : l3
// CHECK:         return [[DEFAULT]] : !moore.l3

// The position is advanced by the step count, wrapping around at both ends of
// the enumerand list, and looked up in the table.
// CHECK:       ^[[MATCH]]([[POS:%.+]]: !moore.i32):
// CHECK:         [[NUM:%.+]] = moore.constant 3 : i32
// CHECK:         [[STEP:%.+]] = moore.modu [[COUNT]], [[NUM]] : i32
// CHECK:         [[SUM:%.+]] = moore.add [[POS]], [[STEP]] : i32
// CHECK:         [[WRAPPED:%.+]] = moore.modu [[SUM]], [[NUM]] : i32
// CHECK:         [[RESULT:%.+]] = moore.dyn_extract [[TABLE]] from [[WRAPPED]] : array<3 x l3>, i32 -> l3
// CHECK:         return [[RESULT]] : !moore.l3

// `prev` walks backwards by subtracting the step count from the number of
// enumerands before offsetting the position.
// CHECK-LABEL: func.func private @enum.prev.Sparse(
// CHECK-SAME:      [[VALUE:%.+]]: !moore.l3, [[COUNT:%.+]]: !moore.i32) -> !moore.l3 {
// CHECK:       ^{{.+}}([[POS:%.+]]: !moore.i32):
// CHECK:         [[NUM:%.+]] = moore.constant 3 : i32
// CHECK:         [[STEP:%.+]] = moore.modu [[COUNT]], [[NUM]] : i32
// CHECK:         [[BACK:%.+]] = moore.sub [[NUM]], [[STEP]] : i32
// CHECK:         [[SUM:%.+]] = moore.add [[POS]], [[BACK]] : i32
// CHECK:         moore.modu [[SUM]], [[NUM]] : i32

// `name` maps the value to the enumerand's name, or to an empty string if it
// is not a member of the enumeration.
// CHECK-LABEL: func.func private @enum.name.Sparse(
// CHECK-SAME:      [[VALUE:%.+]]: !moore.l3) -> !moore.string {
// CHECK:         [[VAL0:%.+]] = moore.constant 1 : l3
// CHECK:         [[VAL1:%.+]] = moore.constant -4 : l3
// CHECK:         [[VAL2:%.+]] = moore.constant -2 : l3
// CHECK:         [[EQ:%.+]] = moore.case_eq [[VALUE]], [[VAL0]] : l3
// CHECK:         [[COND:%.+]] = moore.to_builtin_int [[EQ]] : i1
// CHECK:         [[BYTES:%.+]] = moore.constant_string "X" : i8
// CHECK:         [[STR:%.+]] = moore.int_to_string [[BYTES]] : i8
// CHECK:         cf.cond_br [[COND]], ^[[MATCH:.+]]([[STR]] : !moore.string), ^[[BB:.+]]
// CHECK:       ^[[BB]]:
// CHECK:         moore.case_eq [[VALUE]], [[VAL1]] : l3
// CHECK:         [[BYTES:%.+]] = moore.constant_string "Y" : i8
// CHECK:         [[STR:%.+]] = moore.int_to_string [[BYTES]] : i8
// CHECK:         cf.cond_br {{%.+}}, ^[[MATCH]]([[STR]] : !moore.string), ^[[BB:.+]]
// CHECK:       ^[[BB]]:
// CHECK:         moore.case_eq [[VALUE]], [[VAL2]] : l3
// CHECK:         [[BYTES:%.+]] = moore.constant_string "Z" : i8
// CHECK:         [[STR:%.+]] = moore.int_to_string [[BYTES]] : i8
// CHECK:         cf.cond_br {{%.+}}, ^[[MATCH]]([[STR]] : !moore.string), ^[[BB:.+]]
// CHECK:       ^[[BB]]:
// CHECK:         [[BYTES:%.+]] = moore.constant_string "" : i0
// CHECK:         [[STR:%.+]] = moore.int_to_string [[BYTES]] : i0
// CHECK:         cf.br ^[[MATCH]]([[STR]] : !moore.string)
// CHECK:       ^[[MATCH]]([[RESULT:%.+]]: !moore.string):
// CHECK:         return [[RESULT]] : !moore.string

// CHECK-LABEL: func.func private @Methods(
// CHECK-SAME:      [[A:%.+]]: !moore.l3, [[K:%.+]]: !moore.i32)
function void Methods(Sparse a, int unsigned k);
  Sparse r;
  string n;

  // The step count defaults to one if it is not given explicitly.
  // CHECK: [[ONE:%.+]] = moore.constant 1 : i32
  // CHECK: call @enum.next.Sparse([[A]], [[ONE]]) : (!moore.l3, !moore.i32) -> !moore.l3
  r = a.next();

  // CHECK: call @enum.next.Sparse([[A]], [[K]]) : (!moore.l3, !moore.i32) -> !moore.l3
  r = a.next(k);

  // CHECK: call @enum.prev.Sparse([[A]], %{{.+}}) : (!moore.l3, !moore.i32) -> !moore.l3
  r = a.prev();

  // CHECK: call @enum.name.Sparse([[A]]) : (!moore.l3) -> !moore.string
  n = a.name();

  // The same helper is reused for further calls on the same enum.
  // CHECK: call @enum.name.Sparse([[A]]) : (!moore.l3) -> !moore.string
  n = a.name();

  // Calls on a constant value are folded by Slang.
  // CHECK: [[TMP:%.+]] = moore.constant_string "Y" : i8
  // CHECK: moore.int_to_string [[TMP]] : i8
  n = Y.name();
  // CHECK: [[TMP:%.+]] = moore.constant -2 : l3
  // CHECK: moore.blocking_assign %r, [[TMP]]
  r = Y.next();
  // CHECK: [[TMP:%.+]] = moore.constant 1 : l3
  // CHECK: moore.blocking_assign %r, [[TMP]]
  r = X.first();
  // CHECK: [[TMP:%.+]] = moore.constant -2 : l3
  // CHECK: moore.blocking_assign %r, [[TMP]]
  r = X.last();
endfunction

// CHECK-LABEL: func.func private @Unnamed(
function string Unnamed();
  enum { P, Q } a;
  // CHECK: call @enum.name.anon(
  return a.name();
endfunction

// CHECK-LABEL: func.func private @enum.name.anon(
