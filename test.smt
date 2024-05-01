; solver scope 0
(assert (let ((tmp (exists ((tmp_0 Int) (tmp_1 Int))
( ! (let ((tmp_2 (= tmp_0 tmp_1)))
                           (let ((tmp_3 (=> tmp_2 true)))
                           tmp_3)) :weight 2
                            :pattern ((let ((tmp_4 (= tmp_0 tmp_1)))
                   tmp_4))))))
        tmp))
(check-sat)
(reset)
