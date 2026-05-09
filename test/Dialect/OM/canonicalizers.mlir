// RUN: circt-opt --cse --canonicalize %s | FileCheck %s

om.class @Foo() {
  om.class.fields
}

// CHECK-LABEL: @ObjectsMustNotCSE
func.func @ObjectsMustNotCSE() -> (!om.class.type<@Foo>, !om.class.type<@Foo>) {
  // CHECK-NEXT: [[OBJ1:%.+]] = om.object @Foo
  // CHECK-NEXT: [[OBJ2:%.+]] = om.object @Foo
  // CHECK-NEXT: return [[OBJ1]], [[OBJ2]]
  %obj1 = om.object @Foo() : () -> !om.class.type<@Foo>
  %obj2 = om.object @Foo() : () -> !om.class.type<@Foo>
  return %obj1, %obj2 : !om.class.type<@Foo>, !om.class.type<@Foo>
}

om.class @FooWithAssert() {
  %0 = om.constant false
  om.property_assert %0, "foo" : i1
  om.class.fields
}

// An object instantiation of a class containing a property_assert must not be
// canonicalized away, because the assert has observable side effects.
// CHECK-LABEL: @ObjectWithAssertMustNotDCE
om.class @ObjectWithAssertMustNotDCE() {
  // CHECK: om.object @FooWithAssert
  %0 = om.object @FooWithAssert() : () -> !om.class.type<@FooWithAssert>
  om.class.fields
}

om.class @StringConcatCanonicalization(%str1: !om.string, %str2: !om.string) -> (out1: !om.string, out2: !om.string,
                                                                                  out3: !om.string, out4: !om.string,
                                                                                  out5: !om.string, out6: !om.string,
                                                                                  out7: !om.string, out8: !om.string) {
  %s1 = om.constant "Hello" : !om.string
  %s2 = om.constant "World" : !om.string
  %s3 = om.constant "!" : !om.string
  %empty = om.constant "" : !om.string

  // CHECK-DAG: [[EMPTY:%.+]] = om.constant "" : !om.string
  // CHECK-DAG: [[HELLO:%.+]] = om.constant "Hello" : !om.string
  // CHECK-DAG: [[HELLOWORLD:%.+]] = om.constant "HelloWorld!" : !om.string
  // CHECK-DAG: [[CONST:%.+]] = om.constant "!"

  // Merge all constants
  %0 = om.string.concat %s1, %s2, %s3 : !om.string

  // Drop empty string
  %1 = om.string.concat %s1, %empty : !om.string

  // Single operand replaced with operand
  %2 = om.string.concat %s1 : !om.string

  // Single constant operand folds to the attribute.
  %singleConst = om.string.concat %s3 : !om.string

  // Empty concat
  %3 = om.string.concat %empty, %empty : !om.string

  // Flatten nested concat (single use)
  %4 = om.string.concat %s1, %s2 : !om.string
  %5 = om.string.concat %4, %s3 : !om.string

  // Nested concat with multiple uses should NOT be flattened
  // to avoid fighting with DCE.
  // CHECK-DAG: [[NESTED:%.+]] = om.string.concat %str1, %str2
  // CHECK-DAG: [[CONCAT1:%.+]] = om.string.concat [[NESTED]], [[CONST]]
  %nested = om.string.concat %str1, %str2 : !om.string
  %concat1 = om.string.concat %nested, %s3 : !om.string

  // CHECK: om.class.fields [[HELLOWORLD]], [[HELLO]], [[HELLO]], [[CONST]], [[EMPTY]], [[HELLOWORLD]], [[CONCAT1]], [[NESTED]]
  om.class.fields %0, %1, %2, %singleConst, %3, %5, %concat1, %nested : !om.string, !om.string, !om.string, !om.string, !om.string, !om.string, !om.string, !om.string
}

// CHECK-LABEL: @IntegerBinaryArithmeticFold
om.class @IntegerBinaryArithmeticFold(%x: !om.integer) -> (out1: !om.integer, out2: !om.integer,
                                                           out3: !om.integer, out4: !om.integer,
                                                           out5: !om.integer, out6: !om.integer) {
  %i3 = om.constant #om.integer<3 : si4> : !om.integer
  %i4 = om.constant #om.integer<4 : si4> : !om.integer
  %i2 = om.constant #om.integer<2 : si4> : !om.integer
  %neg1 = om.constant #om.integer<-1 : si4> : !om.integer
  %i1 = om.constant #om.integer<1 : si4> : !om.integer
  %wide = om.constant #om.integer<7 : si6> : !om.integer

  // CHECK-DAG: [[ADD:%.+]] = om.constant #om.integer<7 : si4> : !om.integer
  // Arithmetic uses APSInt semantics at the operands' folded bit width, so
  // 3 * 4 folds to si4 -4 here.
  // CHECK-DAG: [[MUL:%.+]] = om.constant #om.integer<-4 : si4> : !om.integer
  // CHECK-DAG: [[SHR:%.+]] = om.constant #om.integer<1 : si4> : !om.integer
  // CHECK-DAG: [[SHL:%.+]] = om.constant #om.integer<-2 : si4> : !om.integer
  // CHECK-DAG: [[WIDEADD:%.+]] = om.constant #om.integer<9 : si6> : !om.integer
  // CHECK: [[DYN:%.+]] = om.integer.add %x, %{{.+}} : !om.integer
  %0 = om.integer.add %i3, %i4 : !om.integer
  %1 = om.integer.mul %i3, %i4 : !om.integer
  %2 = om.integer.shr %i4, %i2 : !om.integer
  %3 = om.integer.shl %neg1, %i1 : !om.integer

  // Mixed bit widths should still fold after extending operands.
  %4 = om.integer.add %i2, %wide : !om.integer

  // Non-constant operands should remain.
  %5 = om.integer.add %x, %i1 : !om.integer

  // CHECK: om.class.fields [[ADD]], [[MUL]], [[SHR]], [[SHL]], [[WIDEADD]], [[DYN]]
  om.class.fields %0, %1, %2, %3, %4, %5 : !om.integer, !om.integer, !om.integer, !om.integer, !om.integer, !om.integer
}

// CHECK-LABEL: @PropEqFold
om.class @PropEqFold(%str: !om.string, %b: i1, %n: !om.integer) -> (out1: i1, out2: i1,
                                                                     out3: i1, out4: i1,
                                                                     out5: i1, out6: i1,
                                                                     out7: i1, out8: i1,
                                                                     out9: i1) {
  %hello1 = om.constant "hello" : !om.string
  %hello2 = om.constant "hello" : !om.string
  %world  = om.constant "world" : !om.string

  // CHECK-DAG: [[TRUE:%.+]] = om.constant true
  // CHECK-DAG: [[FALSE:%.+]] = om.constant false

  // Equal constant strings fold to true.
  %0 = om.prop.eq %hello1, %hello2 : !om.string

  // Unequal constant strings fold to false.
  %1 = om.prop.eq %hello1, %world : !om.string

  // Non-constant string operands do not fold.
  // CHECK: [[EQ:%.+]] = om.prop.eq %str, %str : !om.string
  %2 = om.prop.eq %str, %str : !om.string

  %true  = om.constant true
  %false = om.constant false

  // Equal constant booleans fold to true.
  %3 = om.prop.eq %true, %true : i1

  // Unequal constant booleans fold to false.
  %4 = om.prop.eq %true, %false : i1

  // Non-constant bool operands do not fold.
  // CHECK: [[BEQ:%.+]] = om.prop.eq %b, %b : i1
  %5 = om.prop.eq %b, %b : i1

  %i42a = om.constant #om.integer<42 : si64> : !om.integer
  %i42b = om.constant #om.integer<42 : si64> : !om.integer
  %i0   = om.constant #om.integer<0 : si64> : !om.integer

  // Equal constant integers fold to true.
  %6 = om.prop.eq %i42a, %i42b : !om.integer

  // Unequal constant integers fold to false.
  %7 = om.prop.eq %i42a, %i0 : !om.integer

  // Non-constant integer operands do not fold.
  // CHECK: [[IEQ:%.+]] = om.prop.eq %n, %n : !om.integer
  %8 = om.prop.eq %n, %n : !om.integer

  // CHECK: om.class.fields [[TRUE]], [[FALSE]], [[EQ]], [[TRUE]], [[FALSE]], [[BEQ]], [[TRUE]], [[FALSE]], [[IEQ]]
  om.class.fields %0, %1, %2, %3, %4, %5, %6, %7, %8 : i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @IntegerBitwiseFold
om.class @IntegerBitwiseFold(%b: i8) -> (out1: i8, out2: i8, out3: i8,
                                          out4: i8, out5: i8, out6: i8,
                                          out7: i8, out8: i8) {
  // CHECK-DAG: [[ZERO:%.+]] = om.constant 0 : i8
  // CHECK-DAG: [[ONES:%.+]] = om.constant -1 : i8
  %zero = om.constant 0 : i8
  %ones = om.constant -1 : i8

  // 0xFF AND 0x00 = 0x00.
  %0 = om.integer.and %ones, %zero : i8

  // 0xFF OR 0x00 = 0xFF.
  %1 = om.integer.or %ones, %zero : i8

  // 0xFF XOR 0x00 = 0xFF.
  %2 = om.integer.xor %ones, %zero : i8

  // 0xFF XOR 0xFF = 0x00.
  %3 = om.integer.xor %ones, %ones : i8

  // Non-constant AND does not fold.
  // CHECK: [[AND:%.+]] = om.integer.and %b, %b : i8
  %4 = om.integer.and %b, %b : i8

  // Non-constant OR does not fold.
  // CHECK: [[OR:%.+]] = om.integer.or %b, %b : i8
  %5 = om.integer.or %b, %b : i8

  // Non-constant XOR does not fold.
  // CHECK: [[XOR:%.+]] = om.integer.xor %b, %b : i8
  %6 = om.integer.xor %b, %b : i8

  // XOR with all-zeros is identity.
  %7 = om.integer.xor %b, %zero : i8

  // CHECK: om.class.fields [[ZERO]], [[ONES]], [[ONES]], [[ZERO]], [[AND]], [[OR]], [[XOR]], %b
  om.class.fields %0, %1, %2, %3, %4, %5, %6, %7 : i8, i8, i8, i8, i8, i8, i8, i8
}
