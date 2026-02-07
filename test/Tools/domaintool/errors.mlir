// Test various error conditions in domaintool

om.class @ClockDomain(
  %basepath: !om.frozenbasepath,
  %name_in: !om.string,
  %period_in: !om.integer
)  -> (
  name_out: !om.string,
  period_out: !om.integer
) {
  om.class.fields %name_in, %period_in : !om.string, !om.integer
}

om.class @PowerDomain(
  %basepath: !om.frozenbasepath,
  %name_in: !om.string
)  -> (
  name_out: !om.string
) {
  om.class.fields %name_in : !om.string
}

om.class @ClockDomain_out(
  %basepath: !om.frozenbasepath,
  %domainInfo_in: !om.class.type<@ClockDomain>,
  %associations_in: !om.list<!om.frozenpath>
)  -> (
  domainInfo_out: !om.class.type<@ClockDomain>,
  associations_out: !om.list<!om.frozenpath>
) {
  om.class.fields %domainInfo_in, %associations_in : !om.class.type<@ClockDomain>, !om.list<!om.frozenpath>
}

// -----

// Test 1: Too many assignments (class has 1 ClockDomain parameter, but 2 assignments)
// RUN: not domaintool --module OneClock --domain ClockDomain,A,10 --assign 0 --assign 0 %s 2>&1 | FileCheck %s --check-prefix=TOO_MANY_ASSIGNS

om.class @OneClock_Class(
  %basepath: !om.frozenbasepath,
  %clk: !om.class.type<@ClockDomain>
)  -> (
  clk_out: !om.class.type<@ClockDomain_out>
) {
  %0 = om.object @ClockDomain_out(%basepath, %clk, %2) : (
    !om.frozenbasepath,
    !om.class.type<@ClockDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@ClockDomain_out>
  %1 = om.frozenpath_create reference %basepath "OneClock>a"
  %2 = om.list_create %1 : !om.frozenpath
  om.class.fields %0 : !om.class.type<@ClockDomain_out>
}

// TOO_MANY_ASSIGNS: error: class has 1 parameter(s) of type 'ClockDomain' but only assigned 2

// -----

// Test 2: Not enough assignments (class has 2 ClockDomain parameters, but only 1 assigned)
// RUN: not domaintool --module TwoClocks --domain ClockDomain,A,10 --assign 0 %s 2>&1 | FileCheck %s --check-prefix=NOT_ENOUGH_ASSIGNS

om.class @TwoClocks_Class(
  %basepath: !om.frozenbasepath,
  %clk1: !om.class.type<@ClockDomain>,
  %clk2: !om.class.type<@ClockDomain>
)  -> (
  clk1_out: !om.class.type<@ClockDomain_out>,
  clk2_out: !om.class.type<@ClockDomain_out>
) {
  %0 = om.object @ClockDomain_out(%basepath, %clk1, %3) : (
    !om.frozenbasepath,
    !om.class.type<@ClockDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@ClockDomain_out>
  %1 = om.object @ClockDomain_out(%basepath, %clk2, %4) : (
    !om.frozenbasepath,
    !om.class.type<@ClockDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@ClockDomain_out>
  %2 = om.frozenpath_create reference %basepath "TwoClocks>a"
  %3 = om.list_create %2 : !om.frozenpath
  %4 = om.list_create %2 : !om.frozenpath
  om.class.fields %0, %1 : !om.class.type<@ClockDomain_out>, !om.class.type<@ClockDomain_out>
}

// NOT_ENOUGH_ASSIGNS: error: class has 2 parameter(s) of type 'ClockDomain' but only assigned 1

// -----

// Test 4: No assignments (class has 1 ClockDomain parameter, but no assignments)
// RUN: not domaintool --module OneClock --domain ClockDomain,A,10 %s 2>&1 | FileCheck %s --check-prefix=NO_ASSIGNS

// NO_ASSIGNS: error: class has 1 parameter(s) of type 'ClockDomain' but only assigned 0

// -----

// Test 5: Invalid assignment value (non-numeric)
// RUN: not domaintool --module OneClock --domain ClockDomain,A,10 --assign foo %s 2>&1 | FileCheck %s --check-prefix=INVALID_ASSIGN

// INVALID_ASSIGN: illegal assignment value 'foo', must be a number

// -----

// Test 6: Assignment index out of range
// RUN: not domaintool --module OneClock --domain ClockDomain,A,10 --assign 5 %s 2>&1 | FileCheck %s --check-prefix=OUT_OF_RANGE

// OUT_OF_RANGE: unable to assign domain '5' because it is larger than the number of domains provided, '1'

