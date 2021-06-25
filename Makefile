all:
	gcc -Wall -Wextra -shared -fopenmp -O3 -fPIC -o mylib.so lib.c
