// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s -ssp-roundtrip | circt-opt | FileCheck %s --check-prefix=INFRA

// 1) tests the plain parser/printer roundtrip.
// CHECK: ssp.library @Lib {
// CHECK:   operator_type @Opr [latency<1>]
// CHECK: }
// CHECK: ssp.resource @RsrcLib {
// CHECK:   resource_type @Rsrc [limit<1>]
// CHECK: }
// CHECK: module @SomeModule {
// CHECK:   ssp.library @Lib {
// CHECK:     operator_type @Opr [latency<2>]
// CHECK:   }
// CHECK:   ssp.resource @RsrcLib {
// CHECK:     resource_type @Rsrc [limit<2>]
// CHECK:   }
// CHECK: }
// CHECK: ssp.instance @SomeInstance of "ModuloProblem" {
// CHECK:   library @InternalLib {
// CHECK:     operator_type @Opr [latency<3>]
// CHECK:   }
// CHECK:   resource @InternalRsrc {
// CHECK:     resource_type @Rsrc [limit<3>]
// CHECK:   }
// CHECK:   graph {
// CHECK:     operation<@Opr> uses[@Rsrc]()
// CHECK:     operation<@InternalLib::@Opr> uses[@InternalRsrc::@Rsrc]()
// CHECK:     operation<@SomeInstance::@InternalLib::@Opr> uses[@SomeInstance::@InternalRsrc::@Rsrc]()
// CHECK:     operation<@Lib::@Opr> uses[@RsrcLib::@Rsrc]()
// CHECK:     operation<@SomeModule::@Lib::@Opr> uses[@SomeModule::@RsrcLib::@Rsrc]()
// CHECK:   }
// CHECK: }

// 2) Import/export via the scheduling infra (i.e. populates a `Problem` instance and reconstructs the SSP IR from it.)
//    Operator types from stand-alone libraries are appended to the instance's internal library.
// INFRA: ssp.instance @SomeInstance of "ModuloProblem" {
// INFRA:   library @InternalLib {
// INFRA:     operator_type @Opr [latency<3>]
// INFRA:     operator_type @Opr_1 [latency<1>]
// INFRA:     operator_type @Opr_2 [latency<2>]
// INFRA:   }
// INFRA:   resource @InternalRsrc {
// INFRA:     resource_type @Rsrc [limit<3>]
// INFRA:     resource_type @Rsrc_1 [limit<1>]
// INFRA:     resource_type @Rsrc_2 [limit<2>]
// INFRA:   }
// INFRA:   graph {
// INFRA:     operation<@Opr> uses[@Rsrc]()
// INFRA:     operation<@Opr> uses[@Rsrc]()
// INFRA:     operation<@Opr> uses[@Rsrc]()
// INFRA:     operation<@Opr_1> uses[@Rsrc_1]()
// INFRA:     operation<@Opr_2> uses[@Rsrc_2]()
// INFRA:   }
// INFRA: }

ssp.library @Lib {
  operator_type @Opr [latency<1>]
}
ssp.resource @RsrcLib {
  resource_type @Rsrc [limit<1>]
}
module @SomeModule {
  ssp.library @Lib {
    operator_type @Opr [latency<2>]
  }
  ssp.resource @RsrcLib {
    resource_type @Rsrc [limit<2>]
  }
}
ssp.instance @SomeInstance of "ModuloProblem" {
  library @InternalLib {
    operator_type @Opr [latency<3>]
  }
  resource @InternalRsrc {
    resource_type @Rsrc [limit<3>]
  }
  graph {
    operation<@Opr> uses[@Rsrc]()
    operation<@InternalLib::@Opr> uses[@InternalRsrc::@Rsrc]()
    operation<@SomeInstance::@InternalLib::@Opr> uses[@SomeInstance::@InternalRsrc::@Rsrc]()
    operation<@Lib::@Opr> uses[@RsrcLib::@Rsrc]()
    operation<@SomeModule::@Lib::@Opr> uses[@SomeModule::@RsrcLib::@Rsrc]()
  }
}
