g++ -c -g -std=c++11 bunch_kaufman.cpp -o bunch_kaufman.o
g++ -c -g -std=c++11 matrix_kaufman.cpp -o matrix_kaufman.o
g++ -c -g -std=c++11 vector_kaufman.cpp -o vector_kaufman.o
g++ -g -o kaufman bunch_kaufman.o matrix_kaufman.o vector_kaufman.o