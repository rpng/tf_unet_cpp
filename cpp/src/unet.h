#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <tensorflow/c/c_api.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"

TF_Buffer* read_tf_buffer(const char* file);                                          
void dealloc(void* data, size_t len, void* arg);
void free_buffer(void* data, size_t len);

class UNet
{
public:
    UNet(int64_t n);
    ~UNet();
    std::vector<cv::Rect> run(const std::vector<cv::Mat>& im, std::vector<cv::Mat>& out);

    TF_Graph* graph;
    TF_Status* status;
    TF_Session* sess;
    TF_Tensor *input, *output[2];
    TF_Output in_op, out_op[2];
    int64_t n, w, h, c, s;
    int64_t dims[4];
};
