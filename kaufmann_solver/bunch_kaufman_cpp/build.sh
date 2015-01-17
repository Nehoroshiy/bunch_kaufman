g++ -c -g -std=c++11 matrix_kaufman.cpp -o matrix_kaufman.o
g++ -c -g -std=c++11 vector_kaufman.cpp -o vector_kaufman.o
g++ -c -g -std=c++11 bunch_kaufman.cpp -o bunch_kaufman.o
g++ -c -g -std=c++11 linear_solvers.cpp -o linear_solvers.o
g++ -c -g -std=c++11 tests.cpp -o tests.o
g++ -c -g -std=c++11 main.cpp -o main.o
g++ -g -o kaufman main.o bunch_kaufman.o matrix_kaufman.o vector_kaufman.o linear_solvers.o tests.o libmapm.a -lm