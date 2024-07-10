module {
  hw.module @TopCircuitChiselEnum(in %inputEnum : i2, in %io_a : i2, in %io_b : i2, in %io_c : i1, in %i_e : i2, in %i_b_inner_e : i2, in %i_b_NOENUM : i1, in %i_b_inner_ee : i2, in %i_b_inner_b_inner_inner_e : i2, in %i_b_inner_b_inner_NOENUM : i1, in %i_b_inner_b_inner_ee : i2, in %i_b_v_0 : i2, in %i_b_v_1 : i2, in %i_b_v_2 : i2, in %i_v_0 : i2, in %i_v_1 : i2, in %i_v_2 : i2, in %v_0 : i2, in %v_1 : i2, in %v_2 : i2, in %vv_0_0 : i2, in %vv_0_1 : i2, in %vv_1_0 : i2, in %vv_1_1 : i2) {
    %0 = dbg.enumdef "circtTests.tywavesTests.TywavesAnnotationCircuits$DataTypesCircuits$TopCircuitChiselEnum$MyEnum", {A = 0 : i64, B = 1 : i64, C = 2 : i64}
    %1 = dbg.enumdef "circtTests.tywavesTests.TywavesAnnotationCircuits$DataTypesCircuits$TopCircuitChiselEnum$MyEnum2", {D = 0 : i64, E = 1 : i64, F = 2 : i64}
    dbg.variable "inputEnum", %inputEnum {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %2 = dbg.subfield "io.a", %io_a {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %3 = dbg.subfield "io.b", %io_b {typeName = "IO[MyEnum2]"} enumDef %1 : i2
    %4 = dbg.subfield "io.c", %io_c {typeName = "IO[Bool]"} : i1
    %5 = dbg.struct {"a": %2, "b": %3, "c": %4} : !dbg.subfield, !dbg.subfield, !dbg.subfield
    dbg.variable "io", %5 {typeName = "IO[AnonymousBundle]"} : !dbg.struct
    %6 = dbg.subfield "i.e", %i_e {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %7 = dbg.subfield "i.b.inner_e", %i_b_inner_e {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %8 = dbg.subfield "i.b.NOENUM", %i_b_NOENUM {typeName = "IO[Bool]"} : i1
    %9 = dbg.subfield "i.b.inner_ee", %i_b_inner_ee {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %10 = dbg.subfield "i.b.inner_b.inner_inner_e", %i_b_inner_b_inner_inner_e {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %11 = dbg.subfield "i.b.inner_b.inner_NOENUM", %i_b_inner_b_inner_NOENUM {typeName = "IO[Bool]"} : i1
    %12 = dbg.subfield "i.b.inner_b.inner_ee", %i_b_inner_b_inner_ee {typeName = "IO[MyEnum2]"} enumDef %1 : i2
    %13 = dbg.struct {"inner_inner_e": %10, "inner_NOENUM": %11, "inner_ee": %12} : !dbg.subfield, !dbg.subfield, !dbg.subfield
    %14 = dbg.subfield "i.b.inner_b", %13 {typeName = "IO[AnonymousBundle]"} : !dbg.struct
    %15 = dbg.subfield "i.b.v[0]", %i_b_v_0 {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %16 = dbg.subfield "i.b.v[1]", %i_b_v_1 {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %17 = dbg.subfield "i.b.v[2]", %i_b_v_2 {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %18 = dbg.array [%15, %16, %17] : !dbg.subfield
    %19 = dbg.subfield "i.b.v", %18 {params = [{name = "gen", typeName = "=> T"}, {name = "length", typeName = "Int", value = "3"}], typeName = "IO[MyEnum[3]]"} : !dbg.array
    %20 = dbg.struct {"inner_e": %7, "NOENUM": %8, "inner_ee": %9, "inner_b": %14, "v": %19} : !dbg.subfield, !dbg.subfield, !dbg.subfield, !dbg.subfield, !dbg.subfield
    %21 = dbg.subfield "i.b", %20 {typeName = "IO[AnonymousBundle]"} : !dbg.struct
    %22 = dbg.subfield "i.v[0]", %i_v_0 {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %23 = dbg.subfield "i.v[1]", %i_v_1 {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %24 = dbg.subfield "i.v[2]", %i_v_2 {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %25 = dbg.array [%22, %23, %24] : !dbg.subfield
    %26 = dbg.subfield "i.v", %25 {params = [{name = "gen", typeName = "=> T"}, {name = "length", typeName = "Int", value = "3"}], typeName = "IO[MyEnum[3]]"} : !dbg.array
    %27 = dbg.struct {"e": %6, "b": %21, "v": %26} : !dbg.subfield, !dbg.subfield, !dbg.subfield
    dbg.variable "i", %27 {typeName = "IO[AnonymousBundle]"} : !dbg.struct
    %28 = dbg.subfield "v[0]", %v_0 {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %29 = dbg.subfield "v[1]", %v_1 {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %30 = dbg.subfield "v[2]", %v_2 {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %31 = dbg.array [%28, %29, %30] : !dbg.subfield
    dbg.variable "v", %31 {params = [{name = "gen", typeName = "=> T"}, {name = "length", typeName = "Int", value = "3"}], typeName = "IO[MyEnum[3]]"} : !dbg.array
    %32 = dbg.subfield "vv[0][0]", %vv_0_0 {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %33 = dbg.subfield "vv[0][1]", %vv_0_1 {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %34 = dbg.array [%32, %33] : !dbg.subfield
    %35 = dbg.subfield "vv[0]", %34 {params = [{name = "gen", typeName = "=> T"}, {name = "length", typeName = "Int", value = "2"}], typeName = "IO[MyEnum[2]]"} : !dbg.array
    %36 = dbg.subfield "vv[1][0]", %vv_1_0 {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %37 = dbg.subfield "vv[1][1]", %vv_1_1 {typeName = "IO[MyEnum]"} enumDef %0 : i2
    %38 = dbg.array [%36, %37] : !dbg.subfield
    %39 = dbg.subfield "vv[1]", %38 {params = [{name = "gen", typeName = "=> T"}, {name = "length", typeName = "Int", value = "2"}], typeName = "IO[MyEnum[2]]"} : !dbg.array
    %40 = dbg.array [%35, %39] : !dbg.subfield
    dbg.variable "vv", %40 {params = [{name = "gen", typeName = "=> T"}, {name = "length", typeName = "Int", value = "2"}], typeName = "IO[MyEnum[2][2]]"} : !dbg.array
    %41 = dbg.subfield "vBundle[0].e", %i_e {typeName = "Wire[MyEnum]"} enumDef %0 : i2
    %42 = dbg.subfield "vBundle[0].b.inner_e", %i_b_inner_e {typeName = "Wire[MyEnum]"} enumDef %0 : i2
    %43 = dbg.subfield "vBundle[0].b.NOENUM", %i_b_NOENUM {typeName = "Wire[Bool]"} : i1
    %44 = dbg.subfield "vBundle[0].b.inner_ee", %i_b_inner_ee {typeName = "Wire[MyEnum]"} enumDef %0 : i2
    %45 = dbg.subfield "vBundle[0].b.inner_b.inner_inner_e", %i_b_inner_b_inner_inner_e {typeName = "Wire[MyEnum]"} enumDef %0 : i2
    %46 = dbg.subfield "vBundle[0].b.inner_b.inner_NOENUM", %i_b_inner_b_inner_NOENUM {typeName = "Wire[Bool]"} : i1
    %47 = dbg.subfield "vBundle[0].b.inner_b.inner_ee", %i_b_inner_b_inner_ee {typeName = "Wire[MyEnum2]"} : i2
    %48 = dbg.struct {"inner_inner_e": %45, "inner_NOENUM": %46, "inner_ee": %47} : !dbg.subfield, !dbg.subfield, !dbg.subfield
    %49 = dbg.subfield "vBundle[0].b.inner_b", %48 {typeName = "Wire[AnonymousBundle]"} : !dbg.struct
    %50 = dbg.subfield "vBundle[0].b.v[0]", %i_b_v_0 {typeName = "Wire[MyEnum]"} enumDef %0 : i2
    %51 = dbg.subfield "vBundle[0].b.v[1]", %i_b_v_1 {typeName = "Wire[MyEnum]"} enumDef %0 : i2
    %52 = dbg.subfield "vBundle[0].b.v[2]", %i_b_v_2 {typeName = "Wire[MyEnum]"} enumDef %0 : i2
    %53 = dbg.array [%50, %51, %52] : !dbg.subfield
    %54 = dbg.subfield "vBundle[0].b.v", %53 {params = [{name = "gen", typeName = "=> T"}, {name = "length", typeName = "Int", value = "3"}], typeName = "Wire[MyEnum[3]]"} : !dbg.array
    %55 = dbg.struct {"inner_e": %42, "NOENUM": %43, "inner_ee": %44, "inner_b": %49, "v": %54} : !dbg.subfield, !dbg.subfield, !dbg.subfield, !dbg.subfield, !dbg.subfield
    %56 = dbg.subfield "vBundle[0].b", %55 {typeName = "Wire[AnonymousBundle]"} : !dbg.struct
    %57 = dbg.subfield "vBundle[0].v[0]", %i_v_0 {typeName = "Wire[MyEnum]"} enumDef %0 : i2
    %58 = dbg.subfield "vBundle[0].v[1]", %i_v_1 {typeName = "Wire[MyEnum]"} enumDef %0 : i2
    %59 = dbg.subfield "vBundle[0].v[2]", %i_v_2 {typeName = "Wire[MyEnum]"} enumDef %0 : i2
    %60 = dbg.array [%57, %58, %59] : !dbg.subfield
    %61 = dbg.subfield "vBundle[0].v", %60 {params = [{name = "gen", typeName = "=> T"}, {name = "length", typeName = "Int", value = "3"}], typeName = "Wire[MyEnum[3]]"} : !dbg.array
    %62 = dbg.struct {"e": %41, "b": %56, "v": %61} : !dbg.subfield, !dbg.subfield, !dbg.subfield
    %63 = dbg.subfield "vBundle[0]", %62 {typeName = "Wire[AnonymousBundle]"} : !dbg.struct
    %64 = dbg.array [%63] : !dbg.subfield
    dbg.variable "vBundle", %64 {params = [{name = "gen", typeName = "=> T"}, {name = "length", typeName = "Int", value = "1"}], typeName = "Wire[AnonymousBundle[1]]"} : !dbg.array
    dbg.moduleinfo {typeName = "TopCircuitChiselEnum"}
    hw.output
  }
  om.class @TopCircuitChiselEnum_Class(%basepath: !om.basepath) {
  }
}
