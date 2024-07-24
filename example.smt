(set-option :produce-proofs true)
(declare-fun A (Int) Bool)
(assert
(forall ((x Int) (vp Int))
        (= vp (+ v 1))
))
(check-sat)
(get-proof)