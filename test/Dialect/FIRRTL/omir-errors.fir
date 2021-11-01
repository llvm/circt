; RUN: circt-translate -import-firrtl -verify-diagnostics --split-input-file %s

; The "nodes" member in OMIRAnnotation is required.
;
; expected-error @+2 {{did not contain required key 'nodes'}}
; expected-note @+1 {{The full Annotation is reproduced here}}
circuit Foo: %[[
  {
    "class":"freechips.rocketchip.objectmodel.OMIRAnnotation"
  }
]]
  module Foo:
    skip

; // -----
; The "info" member in an OMNode is required.
;
; expected-error @+2 {{did not contain required key 'info'}}
; expected-note @+1 {{The full Annotation is reproduced here}}
circuit Foo: %[[
  {
    "class":"freechips.rocketchip.objectmodel.OMIRAnnotation",
    "nodes": [
      {
        "id":"OMID:0"
      }
    ]
  }
]]
  module Foo:
    skip

; // -----
; The "id" member in an OMNode is required.
;
; expected-error @+2 {{did not contain required key 'id'}}
; expected-note @+1 {{The full Annotation is reproduced here}}
circuit Foo: %[[
  {
    "class":"freechips.rocketchip.objectmodel.OMIRAnnotation",
    "nodes": [
      {
        "info":""
      }
    ]
  }
]]
  module Foo:
    skip

; // -----
; The "info" member in an OMField is required.
;
; expected-error @+2 {{did not contain required key 'info'}}
; expected-note @+1 {{The full Annotation is reproduced here}}
circuit Foo: %[[
  {
    "class":"freechips.rocketchip.objectmodel.OMIRAnnotation",
    "nodes": [
      {
        "info":"@[A 0:1]",
        "id":"OMID:0",
        "fields":[
          {
            "name":"x",
            "value":"OMReference:0"
          }
        ]
      }
    ]
  }
]]
  module Foo:
    skip

; // -----
; The "name" member in an OMField is required.
;
; expected-error @+2 {{did not contain required key 'name'}}
; expected-note @+1 {{The full Annotation is reproduced here}}
circuit Foo: %[[
  {
    "class":"freechips.rocketchip.objectmodel.OMIRAnnotation",
    "nodes": [
      {
        "info":"@[A 0:1]",
        "id":"OMID:0",
        "fields":[
          {
            "info":"",
            "value":"OMReference:0"
          }
        ]
      }
    ]
  }
]]
  module Foo:
    skip

; // -----
; The "value" member in an OMField is required.
;
; expected-error @+2 {{did not contain required key 'value'}}
; expected-note @+1 {{The full Annotation is reproduced here}}
circuit Foo: %[[
  {
    "class":"freechips.rocketchip.objectmodel.OMIRAnnotation",
    "nodes": [
      {
        "info":"@[A 0:1]",
        "id":"OMID:0",
        "fields":[
          {
            "info":"",
            "name":"x"
          }
        ]
      }
    ]
  }
]]
  module Foo:
    skip

; // -----
; Test the behavior of an OMIR string-encoded type that is never supposed to be
; seen shows up in the OMIR.  This test is checking the behavior for one such
; type, OMInt, which is should show up as an actual integer.
;
; expected-error @+2 {{found known string-encoded OMIR type "OMInt"}}
; expected-note @+1 {{the problematic OMIR is reproduced here}}
circuit Foo: %[[
  {
    "class":"freechips.rocketchip.objectmodel.OMIRAnnotation",
    "nodes": [
      {
        "info":"@[A 0:1]",
        "id":"OMID:0",
        "fields":[
          {
            "info":"",
            "name":"x",
            "value":"OMInt:0"
          }
        ]
      }
    ]
  }
]]
  module Foo:
    skip

; // -----
; Test that an unknown OMIR string-encoded type produces an error.
;
; expected-error @+2 {{found unknown string-encoded OMIR type "OMFoo"}}
; expected-note @+1 {{the problematic OMIR is reproduced here}}
circuit Foo: %[[
  {
    "class":"freechips.rocketchip.objectmodel.OMIRAnnotation",
    "nodes": [
      {
        "info":"@[A 0:1]",
        "id":"OMID:0",
        "fields":[
          {
            "info":"",
            "name":"x",
            "value":"OMFoo:Bar"
          }
        ]
      }
    ]
  }
]]
  module Foo:
    skip

; // -----
; Test that an unexpected MLIR attribute throws an error.
;
; expected-error @+1 {{found unexpected MLIR attribute "unit"}}
circuit Foo: %[[
  {
    "class":"freechips.rocketchip.objectmodel.OMIRAnnotation",
    "nodes": [
      {
        "info":"@[A 0:1]",
        "id":"OMID:0",
        "fields":[
          {
            "info":"",
            "name":"x",
            "value":null
          }
        ]
      }
    ]
  }
]]
  module Foo:
    skip
