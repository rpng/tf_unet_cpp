#include "unet.h"
#include <time.h>

#include <opencv2/highgui.hpp>


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("Must provide image file!\n");
        return -1;
    }
    
    clock_t t0 = clock();
    UNet unet(1);
    printf("Initialized in %f ms\n", 1000 * (double)(clock()-t0) / CLOCKS_PER_SEC);

    std::vector<cv::Mat> mask, im;
    cv::namedWindow("display window", cv::WINDOW_AUTOSIZE);

    for (int i = 1; i < argc; i++)
    {
        im.push_back(cv::imread(argv[i]));
        if (!im[0].data)
        {
            printf("Could not open image file: %s\n", argv[i]);
            return -1;
        }
        /*
        if (im[0].rows > 480 && im[0].cols > 640)
        {
            cv::Rect roi(im[0].cols/2-320, im[0].rows/2-240, 640, 480);
            im[0] = im[0](roi);
        }
        */
        t0 = clock();
        unet.run(im, mask);
        printf("Inference took %f ms\n", 1000 * (double)(clock()-t0) / CLOCKS_PER_SEC);
        
        cv::Mat im2;
        cv::resize(im[0], im[0], cv::Size(320, 240));
        cv::cvtColor(im[0], im[0], cv::COLOR_BGR2GRAY);
        cv::hconcat(im[0], 255 * mask[0], im2);
        cv::imshow("Display window", im2);

        cv::waitKey(1000);
        im.clear();
    }
    return 0;
}





















