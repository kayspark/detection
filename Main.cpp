#include "MotionCapture.h"
#include <thread>
#include <filesystem>

using namespace std;
using namespace cv;
using namespace chrono;
namespace fs = std::__fs::filesystem;

const int cameraNumber = 0;
//const string fileName = R"(/Users/kspark/Documents/GitHub/detection/TB.mp4)";
string fileName("/Users/kspark/Downloads/hanjun/hanjun3.avi");
//const string fileName = R"(/home/drew/ClionProjects/detection/TB.mp4)";


int main() {

  fs::path p(fileName);
  if (p.empty()) {
    cout << "Error: File " << fileName << " is not found." << endl;
    return -1;
  }
  MotionCapture capture(fileName);

  capture.find();

  return 0;
}
