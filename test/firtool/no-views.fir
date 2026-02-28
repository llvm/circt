; RUN: firtool %s | FileCheck %s --check-prefix=ENABLE
; RUN: firtool %s -no-views | FileCheck %s --check-prefix=DISABLE

FIRRTL version 4.2.0
circuit ViewInLayer:
  layer Views, bind, "views/":

  public module ViewInLayer:
    input x : UInt<1>
    input y : UInt<1>
    intrinsic(circt_view<name="view", info="{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\", \"defName\": \"Root\", \"elements\": [{\"name\": \"foo\", \"tpe\":{\"class\":\"sifive.enterprise.grandcentral.AugmentedGroundType\"}}]}">, x)

    layerblock Views:
        intrinsic(circt_view<name="view", info="{\"class\":\"sifive.enterprise.grandcentral.AugmentedBundleType\", \"defName\": \"MyView\", \"elements\": [{\"name\": \"foo\", \"tpe\":{\"class\":\"sifive.enterprise.grandcentral.AugmentedGroundType\"}}]}">, x)

; ENABLE: interface
; DISABLE-NOT: interface
