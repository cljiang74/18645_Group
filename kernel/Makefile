run:
	gcc -O2 main.c -o main.x -std=c99 -fopenmp -lm
	cat dataset.txt means.txt precisions_cholesky.txt | ./main.x

clean:
	rm main.x