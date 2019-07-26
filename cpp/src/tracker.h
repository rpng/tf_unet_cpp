#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>





// >>>> Kalman Filter
int stateSize;
int measSize;
int contrSize;


// Main KF object
cv::KalmanFilter kf;

// Our state and measurement object
cv::Mat state;
cv::Mat state_old;
cv::Mat meas;

// last time this was tracks
double lastTimeStep = 0;

// Number of seconds that if passed, means we lost the track
double dt_threshold = 5.0;

// Most of this code has been taken from this blog post here:
// https://www.myzhar.com/blog/tutorials/tutorial-opencv-ball-tracker-using-kalman-filter/
void initialize_kf() {

    // set our state size
    stateSize = 6;
    measSize = 8; // 8 bc we get bbox meas from mask and net
    contrSize = 0;



    // init the kalman filter
    kf = cv::KalmanFilter(stateSize, measSize, contrSize, CV_32F);


    //cv::Mat procNoise(stateSize, 1, type)
    // [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

    // Transition State Matrix A
    // Note: set dT at each processing step!
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]


    // Measure Matrix H (8 x 6)
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]



    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]

    kf = cv::KalmanFilter(stateSize, measSize, 0, CV_32F);

    state = cv::Mat(stateSize, 1, CV_32F);  // [x, y, v_x, v_y, w, h]
    meas = cv::Mat(measSize, 1, CV_32F);
    lastTimeStep = 0;

    cv::setIdentity(kf.transitionMatrix);

    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, CV_32F);
    kf.measurementMatrix.at<float>(0 * 6 + 0) = 1.0f;
    kf.measurementMatrix.at<float>(1 * 6 + 1) = 1.0f;
    kf.measurementMatrix.at<float>(2 * 6 + 4) = 1.0f;
    kf.measurementMatrix.at<float>(3 * 6 + 5) = 1.0f;
    kf.measurementMatrix.at<float>(4 * 6 + 0) = 1.0f;
    kf.measurementMatrix.at<float>(5 * 6 + 1) = 1.0f;
    kf.measurementMatrix.at<float>(6 * 6 + 4) = 1.0f;
    kf.measurementMatrix.at<float>(7 * 6 + 5) = 1.0f;

    //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
    // kf.processNoiseCov.at<float>(0) = 1.0f;
    // kf.processNoiseCov.at<float>(7) = 1.0f;
    // kf.processNoiseCov.at<float>(14) = 5.0f;
    // kf.processNoiseCov.at<float>(21) = 5.0f;
    // kf.processNoiseCov.at<float>(28) = 1.0f;
    // kf.processNoiseCov.at<float>(35) = 1.0f;
    kf.processNoiseCov.at<float>(0) = 100.0f;
    kf.processNoiseCov.at<float>(7) = 100.0f;
    kf.processNoiseCov.at<float>(14) = 1.0f;
    kf.processNoiseCov.at<float>(21) = 1.0f;
    kf.processNoiseCov.at<float>(28) = 100.0f;
    kf.processNoiseCov.at<float>(35) = 100.0f;

    // Measures Noise Covariance Matrix R
    
    // From the net (raw)
    kf.measurementNoiseCov.at<float>(0 * 8 + 0) = 1000.0f;
    kf.measurementNoiseCov.at<float>(1 * 8 + 1) = 1000.0f;
    kf.measurementNoiseCov.at<float>(2 * 8 + 2) = 1000.0f;
    kf.measurementNoiseCov.at<float>(3 * 8 + 3) = 1000.0f;

    // From the mask
    kf.measurementNoiseCov.at<float>(4 * 8 + 4) = 1000.0f;
    kf.measurementNoiseCov.at<float>(5 * 8 + 5) = 1000.0f;
    kf.measurementNoiseCov.at<float>(6 * 8 + 6) = 1000.0f;
    kf.measurementNoiseCov.at<float>(7 * 8 + 7) = 1000.0f;

}



// This will update the kalman filter given a new observation
// If we have lost track of it, then we should re-init the filter
//
// measBB should be of length 2, probably allocated on the stack
bool update_kf(double newTimeStep, const cv::Rect *measBB, cv::Rect& upRect) {


    // Calculate delta time
    double dTd = newTimeStep - lastTimeStep;
    float dT = (float) dTd;

    //std::cout << "dT- " << dT << std::endl << std::endl << std::endl;

    // >>>> Matrix A
    kf.transitionMatrix.at<float>(2) = dT;
    kf.transitionMatrix.at<float>(9) = dT;
    // <<<< Matrix A

    // Predict forward in time
    state_old = state.clone();
    state = kf.predict();

    // <<<<< Noise smoothing
    // <<<<< Detection result
    meas.at<float>(0) = measBB[0].x + measBB[0].width / 2;
    meas.at<float>(1) = measBB[0].y + measBB[0].height / 2;
    meas.at<float>(2) = (float) measBB[0].width;
    meas.at<float>(3) = (float) measBB[0].height;

    meas.at<float>(4) = measBB[1].x + measBB[1].width / 2;
    meas.at<float>(5) = measBB[1].y + measBB[1].height / 2;
    meas.at<float>(6) = (float) measBB[1].width;
    meas.at<float>(7) = (float) measBB[1].height;

    // Track if we did an update or not
    bool did_update = false;

    // std::cout  << "dt = " << dTd << std::endl;

    // If lost or initial track, then init it!
    if(dTd > dt_threshold) {

        std::cout  << std::endl << "initalizing KF - dt = " << dTd << std::endl;

        // >>>> Initialization
        kf.errorCovPre.at<float>(0) = 1; // px
        kf.errorCovPre.at<float>(7) = 1; // px
        kf.errorCovPre.at<float>(14) = 1;
        kf.errorCovPre.at<float>(21) = 1;
        kf.errorCovPre.at<float>(28) = 1; // px
        kf.errorCovPre.at<float>(35) = 1; // px

        state.at<float>(0) = meas.at<float>(0);
        state.at<float>(1) = meas.at<float>(1);
        state.at<float>(2) = 0;
        state.at<float>(3) = 0;
        state.at<float>(4) = meas.at<float>(2);
        state.at<float>(5) = meas.at<float>(3);
        // <<<< Initialization
        kf.statePost = state;
        lastTimeStep = newTimeStep;

    } else {

        // compute the error in this measurement
        cv::Mat err = meas - kf.measurementMatrix*state;
        // std::cout << "kf error = " << err << std::endl;

        // calc state covariance if this measurement is to be included
        cv::Mat S = kf.measurementMatrix * kf.errorCovPre * 
                     kf.measurementMatrix.t() + kf.measurementNoiseCov;
        cv::Mat S_inv = S.inv();

        // calc Mahalanobis distance
        cv::Mat chi = err.t()*S_inv*err;
        // std::cout << "kf error = " << err << std::endl;
        std::cout << "chi2 = " << chi.at<float>(0) << std::endl;

        if(chi.at<float>(0) < 300) {
            kf.correct(meas);
            lastTimeStep = newTimeStep;
            did_update = true;
        }
    }


    // Record this retangle
    upRect.width = state.at<float>(4);
    upRect.height = state.at<float>(5);
    upRect.x = state.at<float>(0) - upRect.width / 2;
    upRect.y = state.at<float>(1) - upRect.height / 2;
    return did_update;

}



cv::Rect get_bounding_box(cv::Mat& mask, std::vector<cv::Point>& approx_curve) {

    
    cv::Rect bestrect;

    //bestrect.reserve(n);
    // if (out.size()) out.clear();
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    // >>>>> Contours detection
    // rm noise
    
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element, cv::Point(-1,-1), 1);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element, cv::Point(-1,-1), 2);
    
    // cv::findContours(out, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    // >>>>> Filtering
    double bestarea = 0;
    for (size_t j = 0; j < contours.size(); j++) {
        // bounding box of this contour
        cv::Rect bBox = cv::boundingRect(contours[j]);
        // see if this is the max one in the image
        if (bBox.area() >= bestarea) {
            bestrect = bBox;
            bestarea = bBox.area();
            //double epsilon = 0.05*cv::arcLength(contours[j],true);
            double epsilon = 0.5;
            cv::approxPolyDP(contours[j],approx_curve,epsilon,true);
        }
    }

    // return cv::boundingRect(out);
    return bestrect;




}




