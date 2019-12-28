/*
	main.cpp - Simon M�ller, 09/12/19

*/
#include <iostream>


#include "opencv2/core/core.hpp"

#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/highgui/highgui.hpp"


// If you want to "simplify" code writing, you might want to use:

using namespace cv;

using namespace std;


// TODO probably wrong
double CARD_MAX_AREA = 120000;
double CARD_MIN_AREA = 25000;

const int MAX_LOW_THRESH = 255;
int thresh = 100;
RNG random(12345);
const char* window_name_cam = "Camera";
const char* window_name_canny = "Canny";
const char* window_name_card = "Card %d";

String no_cards = "Detected cards: X";
int noCards = 0;

Mat frame, out_canny;

void canny_threshold_callback(int, void*);
void filter_cards(Mat& image);

bool process_contour(vector<Point>& contour);

void sort_corners(vector <Point2f> & corners);
Mat flatten(vector <Point2f>& corners);

int main(int argc, char** argv)
{
	namedWindow(window_name_cam, WINDOW_AUTOSIZE); // Create Window
	namedWindow(window_name_canny, WINDOW_AUTOSIZE); // Create Window
	//--- INITIALIZE VIDEOCAPTURE
	VideoCapture cap;
	// open the default camera using default API
	// cap.open(0);
	// OR advance usage: select any API backend
	int deviceID = 1;             // 0 = open default camera
	int apiID = cv::CAP_ANY;      // 0 = autodetect default API
	// open selected camera using selected API
	cap.open(deviceID + apiID);
	// check if we succeeded
	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return -1;
	}




	cap.read(frame);
	out_canny = Mat::zeros(frame.size(), CV_8U);
	//Mat r(frame.size(), CV_8U), g(frame.size(), CV_8U), b(frame.size(), CV_8U);

	createTrackbar("Min Treshold:", window_name_canny, &thresh, MAX_LOW_THRESH, canny_threshold_callback);
	canny_threshold_callback(0, 0);


	//--- GRAB AND WRITE LOOP
	cout << "Start grabbing" << endl
		<< "Press any key to terminate" << endl;
	while(1<2)
	{
		// wait for a new frame from camera and store it into 'frame'
		cap.read(frame);
		// check if we succeeded
		if (frame.empty()) {
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}
		//greyscale
		//cvtColor(frame, src_grey, COLOR_BGR2GRAY);
		// Blur a bit
		blur(frame, frame, Size(3, 3));
		//use canny edge detection to filter edges
		canny_threshold_callback(0, 0);
		filter_cards(out_canny);

		cv::putText(frame,
			no_cards,
			cv::Point(5, 20), // Coordinates
			cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
			1.0, // Scale. 2.0 = 2x bigger
			cv::Scalar(255, 255, 255), // BGR Color
			1, // Line Thickness (Optional)
			LINE_AA); // Anti-alias (Optional)

		//drawContours(frame, contours, -1, color, FILLED, LINE_8, hierarchy);
		imshow(window_name_cam, frame);

		imshow(window_name_canny, out_canny);

		// show live and wait for a key with timeout long enough to show images

		if (waitKey(5) >= 0)
			break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;

}

bool process_contour(vector<Point>& contour)
{
	double size = contourArea(contour);
	double peri = arcLength(contour, true);
	vector<Point2f> poly;
	approxPolyDP(contour, poly, 0.01 * peri, true);

	// if it's actually a card
	if ((size < CARD_MAX_AREA) && (size > CARD_MIN_AREA) && poly.size() == 4) 
	{
		
		sort_corners(poly);
		Mat flatImage = flatten(poly);
		
		imshow(format(window_name_card, noCards), flatImage);

		return true;
	}
	
	return false;
}


// sorts corners into bottomLeft, topLeft, topRight, bottomRight.
void sort_corners(vector <Point2f>& corners)
{
	vector<Point2f> top, bot;
	// Get mass center
	cv::Point2f center(0, 0);
	for (int i = 0; i < corners.size(); i++)
		center += corners[i];
	center *= (1. / corners.size());

	for each (auto corner in corners)
	{
		if (corner.y < center.y)
			top.push_back(corner);
		else
			bot.push_back(corner);
	}
	// das geht besser
	cv::Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
	cv::Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
	cv::Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
	cv::Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];

	corners.clear();
	corners.push_back(bl);
	corners.push_back(tl);
	corners.push_back(tr);
	corners.push_back(br);
}

Mat flatten(vector <Point2f>& corners)
{
	// Create a rectangle from the contour with correct numbering 
	Point2f src[4];
	src[0] = corners[0];
	src[1] = corners[1];
	src[2] = corners[2];
	src[3] = corners[3];

	for (int i = 0; i < 4; i++) {
		line(frame, src[i], src[(i + 1) % 4], Scalar(128, 0, 255));
	}

	int maxWidth = 200, maxHeight = 300;

	Point2f dest[4];
	dest[0] = Point2f(0,maxHeight-1);
	dest[1] = Point2f(0,0);
	dest[2] = Point2f(maxWidth-1,0);
	dest[3] = Point2f(maxWidth-1,maxHeight-1);
	// need 4 pairs of corresponding In/Out points
	Mat M = getPerspectiveTransform(src,dest);
	   
	Mat output;
	warpPerspective(frame, output, M, Size(maxWidth, maxHeight));

	return output;
}

void filter_cards(Mat& image) 
{
	noCards = 0;

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Scalar color(0, 0, 255);

	findContours(image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	if (contours.empty())
		return;
	
	vector<vector<Point> > cardContours;

	//TODO: zip contours & hierachys and sort both

	//sort by contour area size
	sort(contours.begin(), contours.end(), [](const vector<Point>& a, const vector<Point>& b) { return contourArea(a) > contourArea(b); });

	// determine if they're cards by 3 criterias:
	// 1 & 2) inside size bounds 3) no parents 4) four corners
	for each (auto contour in contours)
	{
		if (process_contour(contour)) {
			cardContours.push_back(contour);
			noCards++;
		}
	}
	
	no_cards = format("Detected closed contours: %d",(int) cardContours.size());

	drawContours(frame, cardContours, -1, color, 2, LINE_8);

}

void canny_threshold_callback(int, void*)
{
	// Canny Edge Detection
	Canny(frame, out_canny, thresh, thresh * 2);

	// create approx polygons with closed curves
	//vector<vector<Point> > contours_poly(contours.size());
	//vector<Rect> boundRect(contours.size());
	//vector<Point2f>centers(contours.size());

	//for (size_t i = 0; i < contours.size(); i++)
	//{
	//	approxPolyDP(contours[i], contours_poly[i], 3, true);
	//	boundRect[i] = boundingRect(contours_poly[i]);

	//	drawContours(frame, contours_poly, (int)i, color);
	//	rectangle(frame, boundRect[i].tl(), boundRect[i].br(), color, 2);
	//}
}