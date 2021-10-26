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
          {"info": "", "name": "instancePath", "value": "OMMemberInstanceTarget:~Foo|Bar>mem1"}
        ]
      },
      {
        "info": "",
        "id": "OMID:3",
        "fields": [
          {"info": "", "name": "omType", "value": ["OMString:OMLazyModule", "OMString:OMSRAM"]},
          {"info": "", "name": "instancePath", "value": "OMMemberReferenceTarget:~Foo|Bar>mem2"}
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
    inst mem1 of MySRAM
    mem mem2 :
        data-type => UInt<42>
        depth => 8
        read-latency => 0
        write-latency => 1
        reader => port
        read-under-write => undefined
    mem2.port is invalid

; CHECK-LABEL: FILE "omir.json"

; CHECK:       "id": "OMID:0"
; CHECK:       "name": "a"
; CHECK-NEXT:  "value": "OMReference:0"
; CHECK:       "name": "b"
; CHECK-NEXT:  "value": "OMBigInt:42"
; CHECK:       "name": "c"
; CHECK-NEXT:  "value": "OMLong:ff"
; CHECK:       "name": "d"
; CHECK-NEXT:  "value": "OMString:hello"
; CHECK:       "name": "f"
; CHECK-NEXT:  "value": "OMBigDecimal:10.5"
; CHECK:       "name": "g"
; CHECK-NEXT:  "value": "OMDeleted"
; CHECK:       "name": "h"
; CHECK-NEXT:  "value": "OMConstant:UInt<2>(\"h1\")"
; CHECK:       "name": "i"
; CHECK-NEXT:  "value": 42
; CHECK:       "name": "j"
; CHECK-NEXT:  "value": true
; CHECK:       "name": "k"
; CHECK-NEXT:  "value": 3.14

; CHECK:       "id": "OMID:1"
; CHECK:       "name": "a"
; CHECK-NEXT:  "value": "OMReferenceTarget:~Foo|Foo"
; CHECK:       "name": "b"
; CHECK-NEXT:  "value": "OMInstanceTarget:~Foo|Foo"
; CHECK:       "name": "c"
; CHECK-NEXT:  "value": "OMMemberReferenceTarget:~Foo|Foo"
; CHECK:       "name": "d"
; CHECK-NEXT:  "value": "OMMemberInstanceTarget:~Foo|Foo"
; CHECK:       "name": "e"
; CHECK-NEXT:  "value": "OMDontTouchedReferenceTarget:~Foo|Foo"
; CHECK:       "name": "f"
; CHECK-NEXT:  "value": "OMReferenceTarget:~Foo|Bar"

; CHECK:       "id": "OMID:2"
; CHECK:       "name": "instancePath"
; CHECK-NEXT:  "value": "OMMemberInstanceTarget:~Foo|Foo/bar:Bar/mem1:MySRAM"

; CHECK:       "id": "OMID:3"
; CHECK:       "name": "instancePath"
; CHECK-NEXT:  "value": "OMMemberInstanceTarget:~Foo|Foo/bar:Bar/mem2:FIRRTLMem_{{[^"]+}}"
