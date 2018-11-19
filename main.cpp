#include <iostream>
#include "cumatrix.h"

int main() {
  int size = 1000;
  cumatrix a(size, size), b(size, size);
  a.fill_rand();
  b.fill_rand();

  cumatrix c = a * b;

  std::cout << c;

  return 0;
}