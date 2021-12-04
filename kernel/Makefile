run:
	gcc -Wall -O2 main.c -o main.x -std=c99 -fopenmp -lm -mavx -mfma
	cat dataset.txt dataset_T.txt means.txt precisions_cholesky.txt | ./main.x

clean:
	rm main.x