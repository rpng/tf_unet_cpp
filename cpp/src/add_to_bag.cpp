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

    // Ensure we have enough arguments
    if (argc < 3) {
        std::cerr << "please specify the bag and if you want to append to it" << std::endl;
        std::cerr << "example: ./add_to_bag <do_append2bag> <filename.bag>" << std::endl;
        return -1;
    }


    // Load the bag that we want to process
    std::string bag_fl = argv[2];

    // If we should append to the bag file
    bool do_append2bag = (std::string(argv[1])=="1");

    // If the system is stereo
    bool do_stereo = false;

    // Topics that we want to compute masks for
    std::vector<std::string> topics_in, topics_out;
    topics_in.push_back("/cam0/image_raw");
    topics_out.push_back("/cam0/image_mask");
    topics_in.push_back("/cam1/image_raw");
    topics_out.push_back("/cam1/image_mask");
    assert(topics_in.size()==topics_out.size());

    // Debug printing
    std::cout << "Loading ROS Bag File.." << std::endl;
    for(size_t i=0; i<topics_in.size(); i++) {
        std::cout << "\ttopic_" << i << " = " << topics_in.at(i) << std::endl;
    }

    // Open the bag, select the image topics
    rosbag::Bag bag;
    bag.open(bag_fl, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery(topics_in));

    // Loop through and get the list of image messages
    std::vector<std::vector<sensor_msgs::ImageConstPtr>> cams_msgs;
    cams_msgs.resize(topics_in.size());
    for (rosbag::MessageInstance m : view) {
        // Get the message and the topic it is on
        sensor_msgs::ImageConstPtr msg = m.instantiate<sensor_msgs::Image>();
        std::string topic_msg = m.getTopic();
        // check if it matches one of our topics
        for(size_t i=0; i<topics_in.size(); i++) {
            if (topic_msg == topics_in.at(i)) {
                cams_msgs.at(i).push_back(msg);
                break;
            }
        }
    }
    bag.close();

    // Debug printing
    std::cout << "Done Loading Bag, Now Process Masks!!" << std::endl;
    for(size_t i=0; i<topics_in.size(); i++) {
        std::cout << "\tnum messages " << i << " = " << cams_msgs.at(i).size() << std::endl;
    }

    // Create the UNET object!!
    // TODO: pass the location of the network here...
    UNet unet(1);

    // Re-open the bag
    if(do_append2bag)
        bag.open(bag_fl, rosbag::bagmode::Append);
    else
        bag.open(bag_fl, rosbag::bagmode::Read);

    // Nice debug display of the results
    std::string windowname = "Network Input | Network Mask Output";
    cv::namedWindow(windowname, cv::WINDOW_AUTOSIZE);

    // Process the left and right images
    for(size_t i=0; i<topics_in.size(); i++) {

        // Init our Kalman filter!
        initialize_kf(do_stereo);

        // Stats for average inference times
        double sum_inf = 0;
        double max_inf = 0;

        // Loop through this topic's messages
        for (size_t j=0; j<cams_msgs.at(i).size(); j++) {

            // Convert the original image into a opencv matrix
            cv_bridge::CvImageConstPtr image;
            std::vector<cv::Mat> imgvec;
            try {
                image = cv_bridge::toCvCopy(cams_msgs.at(i).at(j), sensor_msgs::image_encodings::BGR8);
                imgvec.push_back(image->image.clone());
            } catch (cv_bridge::Exception& e) {
                fprintf(stderr, "cv_bridge exception: %s", e.what());
                bag.close();
                return -1;
            }
            assert(imgvec.size()==1);

            // Run the actual network
            clock_t t0 = clock();
            std::vector<cv::Mat> masks_raw;
            std::vector<cv::Rect> bboxes = unet.run(imgvec, masks_raw);
            assert(masks_raw.size()==1);

            // Get inference time and update stats
            double inference_time =  1000*((double)(clock()-t0) / CLOCKS_PER_SEC);
            sum_inf += inference_time;
            max_inf = std::max(max_inf, inference_time);

            // Print out debug messages 
            // Note: timing inference + filter + features
            std::cout << "Processing message " << j << " / " << cams_msgs.at(i).size()
                        << " (" << std::setprecision(4) << ((double)j/cams_msgs.at(i).size()*100) << "%)"
                        << " => took " << std::setprecision(4) 
                        << inference_time << " ms"
                        << " | " << sum_inf/(j+1) << " ms avg | " << max_inf << " ms max" << std::endl;


            //============================================================
            //============================================================
            

            // Now write the mask image back to the bag if we need to
            if(do_append2bag) {
                cv_bridge::CvImage img_bridge;
                img_bridge = cv_bridge::CvImage(cams_msgs.at(i).at(j)->header, sensor_msgs::image_encodings::TYPE_8UC1, 255*masks_raw.at(0));
                // out_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
                // out_msg.header = cams_msgs.at(i).at(j)->header; 
                // out_msg.image = ;
                // out_msg.image = 255*masks_raw.at(0);
                sensor_msgs::Image img_msg;
                img_bridge.toImageMsg(img_msg);
                bag.write(topics_out.at(i),img_msg.header.stamp,img_msg);
            }

            //============================================================
            //============================================================
            
            // Find the bounding box
            cv::Mat mask = masks_raw.at(0).clone();
            /* // Now done by CNN
            std::vector<cv::Point> approx_curve;
            */
            cv::Rect bbox = bboxes.at(0); // get_bounding_box(mask,approx_curve);
            
            // Do our kalman filtering
            cv::Rect bbox_kf;
            bool success = update_kf(cams_msgs.at(i).at(j)->header.stamp.toSec(), bbox, bbox_kf);
            
            // Debug display of the 
            cv::Mat im = imgvec.at(0);
            cv::Mat im2, im2r, mask2, mask2r, imout;
            cv::resize(im, im2, cv::Size(320, 240));

            // copy the image that the box bounds
            cv::Mat im3(cv::Size(320, 240), im.type(), cv::Scalar(0));
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
            
            // draw bounding boxes
            cv::rectangle(im2, bbox, cv::Scalar(255,0,0), 1);
            if(success) cv::rectangle(im2, bbox_kf, cv::Scalar(0,255,0), 1);
            else cv::rectangle(im2, bbox_kf, cv::Scalar(0,0,255), 2);

            // Draw the approx curve
            // if(approx_curve.size() > 2) {
            //     for(size_t z=0; z<approx_curve.size()-1; z++) {
            //         cv::line(im2, approx_curve.at(z), approx_curve.at(z+1), cv::Scalar(255,255,0), 1);
            //     }
            //     cv::line(im2, approx_curve.at(approx_curve.size()-1), approx_curve.at(0), cv::Scalar(255,255,0), 1);
            // }

            // finally fuse all the images together
            cv::cvtColor(255*mask, mask2, cv::COLOR_GRAY2BGR);
            cv::hconcat(im2, mask2, imout);
            cv::hconcat(imout, im3, imout);
            cv::resize(imout, imout, cv::Size(6*320, 2*240));
            cv::imshow(windowname, imout);
            cv::waitKey(1);

        }


    }

    bag.close();
}
