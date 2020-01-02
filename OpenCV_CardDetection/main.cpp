/*
	main.cpp - Simon Müller, 09/12/19

*/
#include <iostream>
#include <chrono>

#include "opencv2/core/core.hpp"

#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/highgui/highgui.hpp"



using namespace cv;
using namespace std;


// TODO 
double CARD_MAX_AREA = 120000;
double CARD_MIN_AREA = 25000;

const int IS_COLOR_THRESH = 100;

const int MAX_LOW_THRESH = 255;
int thresh = 100;
RNG random(12345);
const char* window_name_cam = "Camera";
const char* window_name_canny = "Canny";
const char* window_name_card = "Card %d";

String no_cards = "Detected cards: X";

Mat frame, out_canny;

vector<Mat> cards;

void canny_threshold_callback(int, void*);
void filter_cards(Mat& image);

bool process_contour(vector<Point>& contour);

void sort_corners(vector <Point2f> & corners);
Mat flatten(vector <Point2f>& corners);
bool is_color(Mat bgr_image, Scalar color, int threshold);

void capture_image(VideoCapture& video)
{
	if (!video.isOpened())
		return;
	// use this as a callback function when a keyboardbutton (space) is pressed

	// wait for keyboard input and then save the current detected cards as images, with current timestamp
	// do some thresholding before to get clearer results
	
	// grab maybe 10 or so images and combine them into one for better result
	Mat image, temp;
	video >> image;
	for (size_t i = 0; i < 10; i++)
	{
		video >> temp;
		image += temp;
	}

	// build file name
	String baseFilename = "img_";
	time_t seconds;
	time(&seconds);
	stringstream ss;
	ss << baseFilename << seconds;

	// write the image
	imwrite(ss.str(), image);

}

void compare_to_known_images(Mat image) 
{
	// load the reference images - do only on the first time

	// compare to every reference image

	// return index of the image with largest overlap, if above a certain threshold
	// or return the string describing it or sth, no need for OO for this prototype
}

void setup_reference_images()
{
	// load all the references into memory
	// for all files in the folder (how?) - imread
}

int main(int argc, char** argv)
{
	namedWindow(window_name_cam, WINDOW_AUTOSIZE); // Create Window
	namedWindow(window_name_canny, WINDOW_AUTOSIZE); // Create Window
	//--- INITIALIZE VIDEOCAPTURE
	VideoCapture cap;
	// open the default camera using default API
	// cap.open(0);
	// OR advance usage: select any API backend

	/*	---------		CHANGE deviceID to 0 for default front camera! my laptop has a second one on the back i use!	--------- */

	int deviceID = 1;             // 0 = open default camera
	int apiID = CAP_ANY;      // 0 = autodetect default API
	// open selected camera using selected API
	cap.open(deviceID + apiID);
	// check if we succeeded
	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return -1;
	}



	cap.read(frame);
	out_canny = Mat::zeros(frame.size(), CV_8U);

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


		putText(frame,
			no_cards,
			Point(5, 20), // Coordinates
			FONT_HERSHEY_COMPLEX_SMALL, // Font
			1.0, // Scale. 2.0 = 2x bigger
			Scalar(255, 255, 255), // BGR Color
			1, // Line Thickness (Optional)
			LINE_AA); // Anti-alias (Optional)


		imshow(window_name_cam, frame);

		imshow(window_name_canny, out_canny);

		// show and wait for a key with timeout long enough to show images

		if (waitKey(5) >= 0)
			break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;

}


// Tries to find card contours in the given image
void filter_cards(Mat& image) 
{

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Scalar color(0, 0, 255);

	// only retrieves the external contour for now
	findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	if (contours.empty())
		return;
	
	vector<vector<Point> > cardContours;


	//sort by contour area size
	sort(contours.begin(), contours.end(), [](const vector<Point>& a, const vector<Point>& b) { return contourArea(a) > contourArea(b); });

	// determine if they're cards by 3 criterias:
	// 1 & 2) inside size bounds 3) no parents 4) four corners
	for each (auto contour in contours)
	{
		if (process_contour(contour)) {
			cardContours.push_back(contour);
		}
	}
	
	no_cards = format("Detected closed contours: %d",(int) cardContours.size());

	drawContours(frame, cardContours, -1, color, 2, LINE_8);

	for (int i = 0; i < cards.size(); i++)
	{
		imshow(format(window_name_card, i), cards[i]);
	}

}

bool is_color(Mat bgr_image, Scalar low, Scalar high, Scalar low2 = Scalar(), Scalar high2 = Scalar())
{
	// convert to Hue-Saturation-Value color space
	Mat hsv;
	cvtColor(bgr_image, hsv, COLOR_BGR2HSV);

	Mat mask, mask2;
	inRange(hsv, low, high, mask);
	// if additional parameters are given, do it again
	if ((low != Scalar()) && (high != Scalar()))
	{
		inRange(hsv, low2, high2, mask2);
		mask += mask2;
	}
	imshow("mask", mask);
	
	// random value > 0 used rn.. - if hue value exists aka mask is not empty
	if (sum(mask)[0] > IS_COLOR_THRESH)
		return true;
	else
		return false;

}

// process the contour, returns true and creates a flattened image if it's probably a card
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
		
		// Get mass center
		Point2f center(0, 0);
		for (int i = 0; i < poly.size(); i++)
			center += poly[i];
		center *= (1. / poly.size());
		String textcolor = "black";
		// test: check if red color, using two upper/lower limits
		if (is_color(flatImage, Scalar(0, 120, 70), Scalar(10, 255, 255), Scalar(170, 120, 70), Scalar(180, 255, 255)))
			textcolor = "red";

		putText(frame,
			textcolor,
			center, // Coordinates
			FONT_HERSHEY_COMPLEX_SMALL, // Font
			1.0, // Scale. 2.0 = 2x bigger
			Scalar(0,0,0), // BGR Color
			1, // Line Thickness (Optional)
			LINE_AA); // Anti-alias (Optional)


		cards.push_back(flatImage);

		return true;
	}
	
	return false;
}

// sorts corners into bottomLeft, topLeft, topRight, bottomRight.
void sort_corners(vector <Point2f>& corners)
{
	// there has to be a beter way than this


	vector<Point2f> top, bot;
	// Get mass center
	Point2f center(0, 0);
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
	Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
	Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
	Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
	Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];

	Point2f top_edge = tr - tl;
	double top_dist =  sqrt(top_edge.x * top_edge.x + top_edge.y * top_edge.y);
	Point2f left_edge = bl - tl;
	double left_dist =  sqrt(left_edge.x * left_edge.x + left_edge.y * left_edge.y);

	// if the left side edge is shorter than the top, rotate the point assignments to make the shorter edge the top one
	if (left_dist < top_dist) {
		Point2f temp_tl = tl;
		tl = bl;
		bl = br;
		br = tr;
		tr = temp_tl;
	}
		

	corners.clear();
	corners.push_back(bl);
	corners.push_back(tl);
	corners.push_back(tr);
	corners.push_back(br);
}

// use the list of 4 cornerpoints of a card inside the global frame to return a transformed image of the card only
Mat flatten(vector <Point2f>& corners)
{
	// Create a rectangle from the contour with correct numbering 
	Point2f* src = corners.data();

	//draw the outline on the global frame
	for (int i = 0; i < 4; i++) {
		line(frame, src[i], src[(i + 1) % 4], Scalar(128, 0, 255));
	}

	// pixel size of the resulting card image
	int maxWidth = 200, maxHeight = 300;

	Point2f dest[4];
	dest[0] = Point2f(0,maxHeight-1);
	dest[1] = Point2f(0,0);
	dest[2] = Point2f(maxWidth-1,0);
	dest[3] = Point2f(maxWidth-1,maxHeight-1);
	// needs 4 pairs of corresponding In/Out points
	Mat M = getPerspectiveTransform(src,dest);
	   
	Mat output;
	warpPerspective(frame, output, M, Size(maxWidth, maxHeight));

	return output;
}

// Trackbar callback method
void canny_threshold_callback(int, void*)
{
	// Canny Edge Detection
	Canny(frame, out_canny, thresh, thresh * 2);



	// not used anymore:

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