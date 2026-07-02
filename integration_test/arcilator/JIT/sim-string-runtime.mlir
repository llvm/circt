// RUN: arcilator --run %s --jit-entry=entry | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit

func.func @entry() {
  %hello = sim.string.literal "hello"
  %bang = sim.string.literal "!"
  %empty = sim.string.literal ""
  %h = sim.string.literal "h"
  %e = sim.string.literal "e"
  %o = sim.string.literal "o"
  %he = sim.string.literal "he"
  %ab = sim.string.literal "AB"
  %abc = sim.string.literal "ABC"
  %abcdefghijklmnop = sim.string.literal "ABCDEFGHIJKLMNOP"
  %hello_bang = sim.string.literal "hello!"
  %lo = sim.string.literal "lo"
  %zero8 = arith.constant 0 : i8
  %ab_i16 = arith.constant 16706 : i16
  %ab_i15 = arith.constant 16706 : i15
  %abc_i24 = arith.constant 4276803 : i24
  %abcdefghijklmnop_i128 = arith.constant 0x4142434445464748494a4b4c4d4e4f50 : i128
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %three = arith.constant 3 : i32
  %four = arith.constant 4 : i32
  %neg = arith.constant -1 : i32
  %too_far = arith.constant 99 : i32

  %eq = sim.string.cmp %hello, %hello
  // CHECK: eq = 00000000
  arc.sim.emit "eq", %eq : i32

  %ne_cmp = sim.string.cmp %hello, %hello_bang
  %ne_nonzero = arith.cmpi ne, %ne_cmp, %zero : i32
  %ne_nonzero_i32 = arith.extui %ne_nonzero : i1 to i32
  // CHECK: ne_nonzero = 00000001
  arc.sim.emit "ne_nonzero", %ne_nonzero_i32 : i32

  %empty_cmp = sim.string.cmp %empty, %hello
  %empty_cmp_nonzero = arith.cmpi ne, %empty_cmp, %zero : i32
  %empty_cmp_nonzero_i32 = arith.extui %empty_cmp_nonzero : i1 to i32
  // CHECK: empty_cmp_nonzero = 00000001
  arc.sim.emit "empty_cmp_nonzero", %empty_cmp_nonzero_i32 : i32

  %ch = sim.string.get_char %hello[%one]
  // CHECK: getc = 65
  arc.sim.emit "getc", %ch : i8

  %first_ch = sim.string.get_char %hello[%zero]
  // CHECK: first_getc = 68
  arc.sim.emit "first_getc", %first_ch : i8

  %last_ch = sim.string.get_char %hello[%four]
  // CHECK: last_getc = 6f
  arc.sim.emit "last_getc", %last_ch : i8

  %legacy_ch = sim.string.get %hello[%one]
  // CHECK: legacy_getc = 65
  arc.sim.emit "legacy_getc", %legacy_ch : i8

  %legacy_first_ch = sim.string.get %hello[%zero]
  // CHECK: legacy_first_getc = 68
  arc.sim.emit "legacy_first_getc", %legacy_first_ch : i8

  %legacy_last_ch = sim.string.get %hello[%four]
  // CHECK: legacy_last_getc = 6f
  arc.sim.emit "legacy_last_getc", %legacy_last_ch : i8

  %neg_ch = sim.string.get_char %hello[%neg]
  // CHECK: neg_getc = 00
  arc.sim.emit "neg_getc", %neg_ch : i8

  %legacy_neg_ch = sim.string.get %hello[%neg]
  // CHECK: legacy_neg_getc = 00
  arc.sim.emit "legacy_neg_getc", %legacy_neg_ch : i8

  %far_ch = sim.string.get_char %hello[%too_far]
  // CHECK: far_getc = 00
  arc.sim.emit "far_getc", %far_ch : i8

  %legacy_far_ch = sim.string.get %hello[%too_far]
  // CHECK: legacy_far_getc = 00
  arc.sim.emit "legacy_far_getc", %legacy_far_ch : i8

  %empty_ch = sim.string.get_char %empty[%zero]
  // CHECK: empty_getc = 00
  arc.sim.emit "empty_getc", %empty_ch : i8

  %legacy_empty_ch = sim.string.get %empty[%one]
  // CHECK: legacy_empty_getc = 00
  arc.sim.emit "legacy_empty_getc", %legacy_empty_ch : i8

  %chstr = sim.string.int_to_string %ch : i8
  %chstrcmp = sim.string.cmp %chstr, %e
  // CHECK: chstrcmp = 00000000
  arc.sim.emit "chstrcmp", %chstrcmp : i32

  %zero_str = sim.string.int_to_string %zero8 : i8
  %zero_strlen64 = sim.string.length %zero_str
  %zero_strlen = arith.trunci %zero_strlen64 : i64 to i32
  // CHECK: zero_strlen = 00000000
  arc.sim.emit "zero_strlen", %zero_strlen : i32

  %ab_str = sim.string.int_to_string %ab_i16 : i16
  %abstrcmp = sim.string.cmp %ab_str, %ab
  // CHECK: abstrcmp = 00000000
  arc.sim.emit "abstrcmp", %abstrcmp : i32

  %ab_i15_str = sim.string.int_to_string %ab_i15 : i15
  %ab_i15_strcmp = sim.string.cmp %ab_i15_str, %ab
  // CHECK: ab_i15_strcmp = 00000000
  arc.sim.emit "ab_i15_strcmp", %ab_i15_strcmp : i32

  %abc_str = sim.string.int_to_string %abc_i24 : i24
  %abcstrcmp = sim.string.cmp %abc_str, %abc
  // CHECK: abcstrcmp = 00000000
  arc.sim.emit "abcstrcmp", %abcstrcmp : i32

  %wide_str = sim.string.int_to_string %abcdefghijklmnop_i128 : i128
  %widestrcmp = sim.string.cmp %wide_str, %abcdefghijklmnop
  // CHECK: widestrcmp = 00000000
  arc.sim.emit "widestrcmp", %widestrcmp : i32

  %sub = sim.string.substr %hello[%one : %three]
  %sublen64 = sim.string.length %sub
  %sublen = arith.trunci %sublen64 : i64 to i32
  // CHECK: sublen = 00000003
  arc.sim.emit "sublen", %sublen : i32

  %single_sub = sim.string.substr %hello[%zero : %zero]
  %single_sub_cmp = sim.string.cmp %single_sub, %h
  // CHECK: single_sub_cmp = 00000000
  arc.sim.emit "single_sub_cmp", %single_sub_cmp : i32

  %last_single_sub = sim.string.substr %hello[%four : %four]
  %last_single_sub_cmp = sim.string.cmp %last_single_sub, %o
  // CHECK: last_single_sub_cmp = 00000000
  arc.sim.emit "last_single_sub_cmp", %last_single_sub_cmp : i32

  %prefix_sub = sim.string.substr %hello[%zero : %one]
  %prefix_sub_cmp = sim.string.cmp %prefix_sub, %he
  // CHECK: prefix_sub_cmp = 00000000
  arc.sim.emit "prefix_sub_cmp", %prefix_sub_cmp : i32

  %clamped_start = sim.string.substr %hello[%neg : %one]
  %clamped_start_cmp = sim.string.cmp %clamped_start, %he
  // CHECK: clamped_start_cmp = 00000000
  arc.sim.emit "clamped_start_cmp", %clamped_start_cmp : i32

  %inverted_sub = sim.string.substr %hello[%three : %one]
  %inverted_sublen64 = sim.string.length %inverted_sub
  %inverted_sublen = arith.trunci %inverted_sublen64 : i64 to i32
  // CHECK: inverted_sublen = 00000000
  arc.sim.emit "inverted_sublen", %inverted_sublen : i32

  %negative_end_sub = sim.string.substr %hello[%zero : %neg]
  %negative_end_sublen64 = sim.string.length %negative_end_sub
  %negative_end_sublen = arith.trunci %negative_end_sublen64 : i64 to i32
  // CHECK: negative_end_sublen = 00000000
  arc.sim.emit "negative_end_sublen", %negative_end_sublen : i32

  %far_sub = sim.string.substr %hello[%too_far : %too_far]
  %far_sublen64 = sim.string.length %far_sub
  %far_sublen = arith.trunci %far_sublen64 : i64 to i32
  // CHECK: far_sublen = 00000000
  arc.sim.emit "far_sublen", %far_sublen : i32

  %clamped_end = sim.string.substr %hello[%three : %too_far]
  %clamped_end_cmp = sim.string.cmp %clamped_end, %lo
  // CHECK: clamped_end_cmp = 00000000
  arc.sim.emit "clamped_end_cmp", %clamped_end_cmp : i32

  %clamped_full = sim.string.substr %hello[%neg : %too_far]
  %clamped_full_cmp = sim.string.cmp %clamped_full, %hello
  // CHECK: clamped_full_cmp = 00000000
  arc.sim.emit "clamped_full_cmp", %clamped_full_cmp : i32

  %empty_sub = sim.string.substr %empty[%zero : %one]
  %empty_sublen64 = sim.string.length %empty_sub
  %empty_sublen = arith.trunci %empty_sublen64 : i64 to i32
  // CHECK: empty_sublen = 00000000
  arc.sim.emit "empty_sublen", %empty_sublen : i32

  %empty_cat = sim.string.concat ()
  %empty_cat_len64 = sim.string.length %empty_cat
  %empty_cat_len = arith.trunci %empty_cat_len64 : i64 to i32
  // CHECK: empty_cat_len = 00000000
  arc.sim.emit "empty_cat_len", %empty_cat_len : i32

  %single_cat = sim.string.concat (%hello)
  %single_cat_cmp = sim.string.cmp %single_cat, %hello
  // CHECK: single_cat_cmp = 00000000
  arc.sim.emit "single_cat_cmp", %single_cat_cmp : i32

  %cat_empty = sim.string.concat (%hello, %empty)
  %cat_empty_cmp = sim.string.cmp %cat_empty, %hello
  // CHECK: cat_empty_cmp = 00000000
  arc.sim.emit "cat_empty_cmp", %cat_empty_cmp : i32

  %empty_cat_left = sim.string.concat (%empty, %hello)
  %empty_cat_left_cmp = sim.string.cmp %empty_cat_left, %hello
  // CHECK: empty_cat_left_cmp = 00000000
  arc.sim.emit "empty_cat_left_cmp", %empty_cat_left_cmp : i32

  %empty_cat_middle = sim.string.concat (%hello, %empty, %bang)
  %empty_cat_middle_cmp = sim.string.cmp %empty_cat_middle, %hello_bang
  // CHECK: empty_cat_middle_cmp = 00000000
  arc.sim.emit "empty_cat_middle_cmp", %empty_cat_middle_cmp : i32

  %cat = sim.string.concat (%hello, %bang)
  %catlen64 = sim.string.length %cat
  %catlen = arith.trunci %catlen64 : i64 to i32
  // CHECK: catlen = 00000006
  arc.sim.emit "catlen", %catlen : i32

  %catcmp = sim.string.cmp %cat, %hello_bang
  // CHECK: catcmp = 00000000
  arc.sim.emit "catcmp", %catcmp : i32

  return
}
