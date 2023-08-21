; RUN: firtool -parse-only %s --verify-diagnostics

; expected-error @below {{Unable to resolve target of annotation: {class = "circt.test", target = "~OpenAgg|OpenAgg>out.p"}}
circuit OpenAgg : %[[
 { "class": "circt.test",
   "target": "~OpenAgg|OpenAgg>out.p"
 }
]]
  ; expected-error @below {{cannot resolve field 'p'}}
  module OpenAgg:
    output out : {a : UInt<1>,
                  p : Probe<UInt<1>>
                  b : UInt<1>}

    out.a <= UInt<1>(1)
    out.b <= UInt<1>(0)

    node n = out.a
    define out.p = probe(n)


