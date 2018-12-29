#include "MotionCapture.h"

using namespace std;
using namespace cv;
using namespace chrono;
namespace fs = std::__fs::filesystem;

int main() {

  //const int cameraNumber = 0;
//const string fileName = R"(/Users/kspark/Documents/GitHub/detection/TB.mp4)";
  const string_view fileName("/Users/kspark/Downloads/hanjun/hanjun3.avi");
  fs::path p(fileName);
  if (p.empty()) {
    cout << "Error: File " << fileName << " is not found." << endl;
    return -1;
  }
  cv::VideoCapture capture(fileName.data());
  MotionCapture mc(capture);
  while (capture.isOpened()) {
    mc.find();
    mc.display();
  }

  return 0;
}
