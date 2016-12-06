make ./word2vec/src/makefile

time python prepare_word2vec.py

time ./word2vec/bin/word2vec \
-train ./data/word2vec_input.txt \
-output ./models/vec.txt \
-size 50 \
-negative 5 \
-hs 1 \
-sample 1e-4 \
-threads 8 \
-binary 0 \
-iter 5

time python prepare_train_data.py

time python train.py