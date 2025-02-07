; RUN: firtool %s | FileCheck %s

FIRRTL version 4.2.0
circuit ViewInLayer:
  layer Views, bind, "views/":
  layer Other, bind, "other/":
    layer A, bind, "other/A/":
    layer B, inline:

  public module ViewInLayer:
    input x : UInt<1>
    input y : UInt<1>
    intrinsic(circt_view<name="view", info="{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\", \"defName\": \"Root\", \"elements\": [{\"name\": \"foo\", \"tpe\":{\"class\":\"sifive.enterprise.grandcentral.AugmentedGroundType\"}}]}">, x)

    layerblock Views:
        intrinsic(circt_view<name="view", info="{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\", \"defName\": \"MyView\", \"elements\": [{\"name\": \"foo\", \"tpe\":{\"class\":\"sifive.enterprise.grandcentral.AugmentedGroundType\"}}]}">, x)

    layerblock Other:
      node n = and(x, y)
      layerblock A:
        intrinsic(circt_view<name="view_a", info="{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\", \"defName\": \"AView\", \"elements\": [{\"name\": \"foo\", \"tpe\":{\"class\":\"sifive.enterprise.grandcentral.AugmentedGroundType\"}}]}">, n)
      layerblock B:
        intrinsic(circt_view<name="view_b", info="{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\", \"defName\": \"BView\", \"elements\": [{\"name\": \"foo\", \"tpe\":{\"class\":\"sifive.enterprise.grandcentral.AugmentedGroundType\"}}]}">, n)

; CHECK-NOT: FILE
; CHECK: module ViewInLayer
; CHECK-NOT: FILE
; CHECK: interface Root

; CHECK-DAG: FILE "views{{[/\]}}MyView.sv"
; CHECK-DAG: FILE "other{{[/\]}}A{{[/\]}}AView.sv"
; CHECK-DAG: FILE "other{{[/\]}}BView.sv"
