
#ifndef CUMATRIX_CUH
#define CUMATRIX_CUH
#include <ostream>

struct cumatrix {
  // types:
  typedef cumatrix self;
  typedef self &selfref;
  typedef float value_type;
  typedef value_type &reference;
  typedef const value_type &const_reference;
  typedef value_type *iterator;
  typedef const value_type *const_iterator;
  typedef int size_type;
  typedef value_type *pointer;
  typedef const value_type *const_pointer;

  float *elemns;
  float *d_elemns;
  bool in_device = false;
  int N, M;

  // Constructors
  cumatrix(int n, int m) {
    N = n; M = m;
    elemns = new float[N * M];
  }
  ~cumatrix() {
    if (in_device)
      release_device_data();
    delete [] elemns;
  }

  // Capacity
  constexpr size_type size() const noexcept { return (N * M); }

  constexpr size_type rows() const noexcept { return N; }

  constexpr size_type cols() const noexcept { return M; }

  // element access:
  reference operator[](size_type n) { return elemns[n]; }

  const_reference operator[](size_type n) const { return elemns[n]; }

  reference operator()(size_type x, size_type y) { return elemns[x * M + y]; }

  const_reference operator()(size_type x, size_type y) const {  return elemns[x * M + y]; }

  value_type *data() noexcept { return elemns; }

  pointer get_device_pointer(bool copy = true);

  void refresh_from_device();

  void refresh_to_device();

  void release_device_data();

  void fill_rand();
};

std::ostream& operator << (std::ostream& out, const cumatrix& mat);

cumatrix operator*(cumatrix& a, cumatrix& b);

#endif // CUMATRIX_CUH