# RUN: %rtgtool% %s --seed=0 --output-format=mlir | FileCheck %s

from pyrtg import test, config, Config, report_success, report_failure, String, embed_comment

# CHECK-LABEL: rtg.target @Singleton : !rtg.dict<>
# CHECK-NEXT: }


@config
class Singleton(Config):
  pass


# CHECK-LABEL: rtg.test @test94_report_test_result
# CHECK-DAG: [[STR2:%.+]] = rtg.constant "failure from String" : !rtg.string
# CHECK-DAG: [[STR:%.+]] = rtg.constant "this is a failure message" : !rtg.string
# CHECK: rtg.test.success
# CHECK-NEXT: rtg.test.failure [[STR]]
# CHECK-NEXT: rtg.test.failure [[STR2]]
# CHECK-NEXT: }


@test(Singleton)
def test94_report_test_result(config):
  report_success()
  report_failure("this is a failure message")
  report_failure(String("failure from String"))


# CHECK-LABEL: rtg.test @test95_comment
# CHECK-DAG: [[STR2:%.+]] = rtg.constant "comment from String" : !rtg.string
# CHECK-DAG: [[STR:%.+]] = rtg.constant "this is a comment" : !rtg.string
# CHECK: rtg.comment [[STR]]
# CHECK-NEXT: rtg.comment [[STR2]]


@test(Singleton)
def test95_comment(config):
  embed_comment("this is a comment")
  embed_comment(String("comment from String"))
