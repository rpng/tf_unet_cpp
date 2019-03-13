#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "unet.h"
#include "tracker.h"

int main(int argc, char* argv[])
{

    // Load the bag that we want to process
    std::string bag_fl;
    if (argc < 2)
        bag_fl = "test.bag";
    else
        bag_fl = argv[1];

    // IT WE SHOULD APPEND TO THE BAG FILE
    // TODO: WE SHOULD PASS THIS FROM THE COMMAND LINE
    bool do_append2bag = false;

    // Topics that we want to compute masks for
    std::vector<std::string> topics;
    topics.push_back("/cam0/image_raw");

    if (stereo) topics.push_back("/cam1/image_raw");

    // Debug printing
    std::cout << "Loading ROS Bag File.." << std::endl;
    for(size_t i=0; i<topics.size(); i++) {
        std::cout << "\ttopic_" << i << " = " << topics.at(i) << std::endl;
    }

    // Open the bag, select the image topics
    rosbag::Bag bag;
    bag.open(bag_fl, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    // Loop through and get the list of image messages
    std::vector<std::vector<sensor_msgs::ImageConstPtr>> cams_msgs;
    cams_msgs.resize(topics.size());
    for (rosbag::MessageInstance m : view) {
        // Get the message and the topic it is on
        sensor_msgs::ImageConstPtr msg = m.instantiate<sensor_msgs::Image>();
        std::string topic_msg = m.getTopic();
        // check if it matches one of our topics
        for(size_t i=0; i<topics.size(); i++) {
            if (topic_msg == topics.at(i)) {
                cams_msgs.at(i).push_back(msg);
                break;
            }
        }
    }
    bag.close();

    if (stereo && cams_msgs[0].size() != cams_msgs[1].size())
    {
        fprintf(stderr, "Different number of stereo frames!!!\n\n");
        return -1;
    }


    // T from cam 0 to cam 1
    // Note that this is the inverse of what's in the calibration files
    float __T_c0_c1[] = {9.99937076e-01,   1.11321664e-02, -1.38728189e-03, 6.63809634e-02,
                        -1.11332337e-02,  9.99937738e-01, -7.61352977e-04, -3.98700900e-04,
                        1.37872803e-03,   7.76757110e-04,  9.99998746e-01, 7.36277012e-04,
                        0.0,              0.0,             0.0,            1.0};
    cv::Mat T_c0_c1(4, 4, CV_32F, __T_c0_c1);

    float fx0 = 276.4850207717928, fy0 = 278.0310503180516;
    float cx0 = 314.5836189313042, cy0 = 240.16980920673427;

    float fx1 = 277.960323846132, fy1 = 279.4348778432714;
    float cx1 = 322.404194404853, cy1 = 236.72685252691352;

    cv::Mat K0 = cv::Mat::eye(3, 3, CV_32F), K1 = cv::Mat::eye(3, 3, CV_32F);
    K0.at<float>(0,0) = fx0; K0.at<float>(1,1) = fy0;
    K0.at<float>(0,2) = cx0; K0.at<float>(1,2) = cy0;
    K1.at<float>(0,0) = fx1; K1.at<float>(1,1) = fy1; 
    K1.at<float>(0,2) = cx1; K1.at<float>(1,2) = cy1;

    // cam0 distortion
    float __d0[] ={-0.03149689493503132, 0.07696336480701078,
        -0.06608854732019281, 0.019667561645120218};
    //cam 1 distortion
    float __d1[] ={-0.02998039058251529, 0.07202819722706337,
        -0.06178718820631651, 0.017655045017816777};



    size_t n = topics.size();
    // Debug printing
    std::cout << "Done Loading Bag, Now Process Masks!!" << std::endl;
    for(size_t i=0; i<n; i++) {
        std::cout << "\tnum messages " << i << " = " << cams_msgs.at(i).size() << std::endl;
    }

    // Create the UNET object!!
    // TODO: pass the location of the network here...
    UNet unet(n);

    // Re-open the bag
    bag.open(bag_fl, rosbag::bagmode::Read);

    // Nice debug display of the results
    std::vector<cv::Mat> im(n), mask(n);
    std::string windowname = "Network Input | Network Mask Output";
    cv::namedWindow(windowname, cv::WINDOW_AUTOSIZE);

    // Process the left and right images
    //for(size_t i=0; i<topics.size(); i++) {

    // Init our Kalman filter!
    initialize_kf();

    // Stats for average inference times
    double sum_inf = 0;
    double max_inf = 0;
    cv_bridge::CvImageConstPtr image;
    // Loop through this topic's messages
    for (size_t j=0; j<cams_msgs[0].size(); j++) {
        // Convert the original image into a opencv matrix
        try {
            image = cv_bridge::toCvCopy(cams_msgs.at(0).at(j), sensor_msgs::image_encodings::BGR8);
            im[0] = image->image;
            if (stereo)
            {
                image = cv_bridge::toCvCopy(cams_msgs.at(1).at(j),
                        sensor_msgs::image_encodings::BGR8);
                im[1] = image->image;
            }
        } catch (cv_bridge::Exception& e) {
            fprintf(stderr, "cv_bridge exception: %s", e.what());
            bag.close();
            return -1;
        }

        // Run the actual network
        clock_t t0 = clock();
        std::vector<cv::Rect> bbox = unet.run(im, mask);

        // Now write the mask image back to the bag if we need to
        // TODO: fix me...
        /*
        if(do_append2bag) {
            cv_bridge::CvImage out_msg;
            out_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
            out_msg.header = cam_msgs.at(j)->header; 
            out_msg.image = 255*mask[0];
            ros::Time(msg->header.stamp.sec, msg->header.stamp.nsec),
                    out_msg);
        }
        */

        // Do our kalman filtering
        cv::Rect bbox_kf;
        bool success = update_kf(cams_msgs[0][j]->header.stamp.toSec(), bbox[0], bbox_kf);
        
        // Debug display of the  
        cv::Mat im2, im2r, mask2, mask2r, imout;
        cv::resize(im[0], im2, cv::Size(320, 240));
        if (stereo) cv::resize(im[1], im2r, cv::Size(320, 240));

        // copy the image that the box bounds
        cv::Mat im3(cv::Size(320, 240), im[0].type(), cv::Scalar(0));
        if(0 <= bbox_kf.x && 0 <= bbox_kf.width && bbox_kf.x + bbox_kf.width <= im3.cols
                && 0 <= bbox_kf.y && 0 <= bbox_kf.height && bbox_kf.y + bbox_kf.height <= im3.rows) 
        {
            // select subimage    
            cv::Mat subimg = im2(bbox_kf).clone();
            // extract fast for debuging
            std::vector<cv::KeyPoint> corners;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(15,true);
            detector->detect(subimg,corners,cv::Mat());
            auto it = corners.begin();
            while (it != corners.end()) {
                cv::circle(subimg,(*it).pt,1,cv::Scalar(100,100,255),1);
                ++it;
            }                               
            // copy the the larger image
            subimg.copyTo(im3(cv::Rect(160-bbox_kf.width/2, 120-bbox_kf.height/2, subimg.cols, subimg.rows)));
            cv::putText(im3, std::to_string(bbox_kf.width)+" by "+std::to_string(bbox_kf.height), cv::Point(10,20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar::all(255), 1, 1);
        }
        
        // Get inference time and update stats
        double inference_time =  1000*((double)(clock()-t0) / CLOCKS_PER_SEC);
        sum_inf += inference_time;
        max_inf = std::max(max_inf, inference_time);

        // Print out debug messages 
        // Note: timing inference + filter + features
        std::cout << "Processing message " << j << " / " << cams_msgs[0].size()
                    << " (" << std::setprecision(4) << ((double)j/cams_msgs[0].size()*100) << "%)"
                    << " => took " << std::setprecision(4) 
                    << inference_time << " ms"
                    << " | " << sum_inf/(j+1) << " ms avg | " << max_inf << " ms max" << std::endl;
        // draw bounding boxes
        cv::rectangle(im2, bbox[0], cv::Scalar(255,0,0), 1);
        if(success) cv::rectangle(im2, bbox_kf, cv::Scalar(0,255,0), 1);
        else cv::rectangle(im2, bbox_kf, cv::Scalar(0,0,255), 2);

        // finally fuse all the images together
        cv::cvtColor(255 * mask[0], mask2, cv::COLOR_GRAY2BGR);
        if (stereo) cv::cvtColor(255 * mask[1], mask2r, cv::COLOR_GRAY2BGR);
        cv::hconcat(im2, mask2, imout);
        if (stereo) cv::hconcat(imout, mask2r, imout);
        cv::hconcat(imout, im3, imout);
        int mult = stereo ? 1 : 2;
        cv::resize(imout, imout, cv::Size(6*320, mult*240));
        cv::imshow(windowname, imout);
        cv::waitKey(10);

        //if (im.size()) im.clear();
    }
    std::cout << std::endl;
    bag.close();
}
