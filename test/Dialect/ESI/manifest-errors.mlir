// RUN: split-file %s %t
// RUN: rm -f %t/esi_system_manifest.json %t/esi_system_manifest.json.zlib
// RUN: cd %t && circt-opt offsets.mlir --esi-build-manifest=top=top --split-input-file --verify-diagnostics
// RUN: not test -e %t/esi_system_manifest.json
// RUN: not test -e %t/esi_system_manifest.json.zlib

//--- offsets.mlir

// expected-error@+1 {{ESI system manifest does not support nonzero union member offsets}}
module {
  hw.module @top() {}
  esi.manifest.hier_root @top {}
  esi.manifest.sym @top {payload = !esi.channel<!hw.union<small: i5 offset 20, wide: i16>>}
}

// -----

// expected-error@+1 {{ESI system manifest does not support nonzero union member offsets}}
module {
  hw.type_scope @types {
    hw.typedecl @OffsetUnion : !hw.union<small: i5 offset 2, wide: i16>
  }
  hw.module @top() {}
  esi.manifest.hier_root @top {}
  esi.manifest.sym @top {payload = !hw.struct<items: !hw.array<2x!hw.typealias<@types::@OffsetUnion, !hw.union<small: i5 offset 2, wide: i16>>>>}
}
