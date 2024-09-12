# Deprecate initialization macros

Let's deprecate `FIRRTL_BEFORE_INITIAL`, `FIRRTL_AFTER_INITIAL` and `INIT_RANDOM`

## SiFive use case.
* `FIRRTL_BEFORE_INITIAL` and `FIRRTL_AFTER_INITIAL` are used to exclude initial blocks from coverage. -- use `-cm_report noinitial`.

* `INIT_RANDOM` is replaced with a function call fixed seed for modules. -- can we just emit this always? 