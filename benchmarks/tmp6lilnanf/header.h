
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#if defined(__clang__) || defined(__GNUC__)
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif

union Entry {
  int missing;
  float fvalue;
  int qvalue;
};

struct Node {
  uint8_t default_left;
  unsigned int split_index;
  float threshold;
  int left_child;
  int right_child;
};

extern const unsigned char is_categorical[];

__declspec(dllexport) size_t get_num_output_group(void);
__declspec(dllexport) size_t get_num_feature(void);
__declspec(dllexport) const char* get_pred_transform(void);
__declspec(dllexport) float get_sigmoid_alpha(void);
__declspec(dllexport) float get_global_bias(void);
__declspec(dllexport) float predict(union Entry* data, int pred_margin);
