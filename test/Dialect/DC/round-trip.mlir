// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s


// dc.func @roundtrip(
//         %sel : !dc.value<i1>,
//         %a: !dc.value<i64>,
//         %b : !dc.value<i64>,
//         %c : !dc.value<i64>) -> (!dc.value<i64>) {

//     %sel_t, %sel_v = dc.unwrap %sel : !dc.value<i1>
//     %a_t, %a_v = dc.unwrap %a : !dc.value<i64>
//     %b_t, %b_v = dc.unwrap %b : !dc.value<i64>
//     %c_t, %c_v = dc.unwrap %c : !dc.value<i64>

//     // Symbols
//     dc.symbol @nMux
//     dc.symbol @nAdd

//     // Control side
//     %0 = dc.merge %sel, %a_t, %b_t : !dc.token
//     %1 = dc.control @nMux %0
//     %2 = dc.join %0, %c_t : !dc.token
//     %3 = dc.control @nAdd %2

//     // Data side
//     %4 = arith.select %sel_v, %a_v, %b_v : i64
//     %5 = dc.data @nMux %4 : i64
//     %3 = arith.addi %2, %c_v : i64
//     %6 = dc.data @nAdd %3 : i64

//     // Wrap up
//     %7 = dc.wrap %3, %6 : !dc.token, i64

//     hw.output %7 : !dc.value<i64>
// }

