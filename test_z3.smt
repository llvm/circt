(define-fun _0 ((x!0 Int) (x!1 Int) (x!2 Int)) Bool
  true)
(define-fun _2 ((x!0 Int) (x!1 Int) (x!2 Int)) Bool
  false)
(define-fun ERR ((x!0 Int) (x!1 Int) (x!2 Int)) Bool
  true)
(define-fun _1 ((x!0 Int) (x!1 Int) (x!2 Int)) Bool
  (and (<= 1 x!0)
       (<= 4 x!0)
       (<= 1 x!1)
       (<= (- 1) x!2)
       (<= 0 x!2)
       (<= 1 x!2)
       (not (<= 2 x!2))))
(define-fun arg0 ((x!0 Int)) Int
  (ite (and (<= 0 x!0) (<= 1 x!0) (not (<= 2 x!0))) 4 5))
(define-fun _3 ((x!0 Int) (x!1 Int) (x!2 Int)) Bool
  false)
(define-fun _4 ((x!0 Int) (x!1 Int) (x!2 Int)) Bool
  false)
(define-fun _5 ((x!0 Int) (x!1 Int) (x!2 Int)) Bool
  false)