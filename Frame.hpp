#pragma once
#include <opencv2/opencv.hpp>
#include <chrono>


class Frame
{
	//	chrono::milliseconds timeStamp;
	cv::Mat img, mask;

public:
	//	Frame(chrono::milliseconds, Mat, Mat);
	Frame(cv::Mat &pic, cv::Mat &m) : img(pic), mask(m) {}
  ~Frame() = default;
  cv::Mat getImg() { return img; }
};