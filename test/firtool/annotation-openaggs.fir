; RUN: firtool -parse-only %s | FileCheck %s

circuit OpenAgg : %[[
 { "class": "circt.test",
   "target": "~OpenAgg|OpenAgg>out.a"
 },
 { "class": "circt.test",
   "target": "~OpenAgg|OpenAgg>out.b"
 },
 { "class": "circt.test",
   "target": "~OpenAgg|OpenAgg>out"
 }
]]
  ; CHECK-LABEL: module @OpenAgg
  ; CHECK-SAME: (out %out: !firrtl.bundle<a: uint<1>, b: uint<1>> [{circt.fieldID = 1 : i32, class = "circt.test"}, {circt.fieldID = 2 : i32, class = "circt.test"}, {class = "circt.test"}]
  ; CHECK-SAME: , out %out_p: !firrtl.probe<uint<1>>)
  module OpenAgg:
    output out : {a : UInt<1>,
                  p : Probe<UInt<1>>
                  b : UInt<1>}

    out.a <= UInt<1>(1)
    out.b <= UInt<1>(0)

    node n = out.a
    define out.p = probe(n)

