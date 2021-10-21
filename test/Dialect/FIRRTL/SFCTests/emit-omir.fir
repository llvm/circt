; RUN: firtool --verify-diagnostics --verilog --emit-omir %s | FileCheck %s

circuit Foo : %[[
  {
    "class": "freechips.rocketchip.objectmodel.OMIRFileAnnotation",
    "filename": "omir.json"
  },
  {
    "class": "freechips.rocketchip.objectmodel.OMIRAnnotation",
    "nodes": [
      {
        "info": "",
        "id": "OMID:0",
        "fields": [
          {"info": "", "name": "a", "value": "OMReference:0"},
          {"info": "", "name": "b", "value": "OMBigInt:42"},
          {"info": "", "name": "c", "value": "OMLong:ff"},
          {"info": "", "name": "d", "value": "OMString:hello"},
          {"info": "", "name": "f", "value": "OMBigDecimal:10.5"},
          {"info": "", "name": "g", "value": "OMDeleted"},
          {"info": "", "name": "h", "value": "OMConstant:UInt<2>(\"h1\")"},
          {"info": "", "name": "i", "value": 42},
          {"info": "", "name": "j", "value": true},
          {"info": "", "name": "k", "value": 3.14}
        ]
      },
      {
        "info": "",
        "id": "OMID:1",
        "fields": [
          {"info": "", "name": "a", "value": "OMReferenceTarget:~Foo|Foo"},
          {"info": "", "name": "b", "value": "OMInstanceTarget:~Foo|Foo"},
          {"info": "", "name": "c", "value": "OMMemberReferenceTarget:~Foo|Foo"},
          {"info": "", "name": "d", "value": "OMMemberInstanceTarget:~Foo|Foo"},
          {"info": "", "name": "e", "value": "OMDontTouchedReferenceTarget:~Foo|Foo"},
          {"info": "", "name": "f", "value": "OMReferenceTarget:~Foo|Bar"}
        ]
      },
      {
        "info": "",
        "id": "OMID:2",
        "fields": [
          {"info": "", "name": "omType", "value": ["OMString:OMLazyModule", "OMString:OMSRAM"]},
          {"info": "", "name": "instancePath", "value": "OMMemberReferenceTarget:~Foo|Bar>mem"}
        ]
      }
    ]
  }
]]
  extmodule MySRAM:
    defname = MySRAM
  module Foo :
    inst bar of Bar
  module Bar :
    inst mem of MySRAM

; CHECK-LABEL: FILE "omir.json"

; CHECK-LABEL: "id": "OMID:0"
; CHECK-LABEL: "name": "a"
; CHECK-NEXT:  "value": "OMReference:0"
; CHECK-LABEL: "name": "b"
; CHECK-NEXT:  "value": "OMBigInt:42"
; CHECK-LABEL: "name": "c"
; CHECK-NEXT:  "value": "OMLong:ff"
; CHECK-LABEL: "name": "d"
; CHECK-NEXT:  "value": "OMString:hello"
; CHECK-LABEL: "name": "f"
; CHECK-NEXT:  "value": "OMBigDecimal:10.5"
; CHECK-LABEL: "name": "g"
; CHECK-NEXT:  "value": "OMDeleted"
; CHECK-LABEL: "name": "h"
; CHECK-NEXT:  "value": "OMConstant:UInt<2>(\"h1\")"
; CHECK-LABEL: "name": "i"
; CHECK-NEXT:  "value": 42
; CHECK-LABEL: "name": "j"
; CHECK-NEXT:  "value": true
; CHECK-LABEL: "name": "k"
; CHECK-NEXT:  "value": 3.14

; CHECK-LABEL: "id": "OMID:1"
; CHECK-LABEL: "name": "a"
; CHECK-NEXT:  "value": "OMReferenceTarget:~Foo|Foo"
; CHECK-LABEL: "name": "b"
; CHECK-NEXT:  "value": "OMInstanceTarget:~Foo|Foo"
; CHECK-LABEL: "name": "c"
; CHECK-NEXT:  "value": "OMMemberReferenceTarget:~Foo|Foo"
; CHECK-LABEL: "name": "d"
; CHECK-NEXT:  "value": "OMMemberInstanceTarget:~Foo|Foo"
; CHECK-LABEL: "name": "e"
; CHECK-NEXT:  "value": "OMDontTouchedReferenceTarget:~Foo|Foo"
; CHECK-LABEL: "name": "f"
; CHECK-NEXT:  "value": "OMReferenceTarget:~Foo|Bar"

; CHECK-LABEL: "id": "OMID:2"
; CHECK-LABEL: "name": "instancePath"
; CHECK-NEXT:  "value": "OMMemberReferenceTarget:~Foo|Foo/bar:Bar>mem"
