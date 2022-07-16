
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
constexpr auto PI = 3.141592;;

using namespace cv;
using namespace std;

float* GetCubemapCoordinate(int x, int y, int face, int edge, float* point)
{
    float a = 2 * (x / (float)edge);
    float b = 2 * (y / (float)edge);

    if (face == 0) { point[0] = -1; point[1] = 1 - a; point[2] = 3 - b; }         // Back
    else if (face == 1) { point[0] = a - 3, point[1] = -1;     point[2] = 3 - b; }   // Left
    else if (face == 2) { point[0] = 1;      point[1] = a - 5; point[2] = 3 - b; }   // Front
    else if (face == 3) { point[0] = 7 - a;   point[1] = 1;     point[2] = 3 - b; }   // Right
    else if (face == 4) { point[0] = a - 3;   point[1] = 1 - b; point[2] = 1; }   // Top
    else if (face == 5) { point[0] = a - 3;   point[1] = b - 5; point[2] = -1; }   // Bottom

    return point;
}


Mat CvtSph2Cub(Mat* pano) {

    /*
       구면 파노라마 이미지 너비, 높이 구하기
    */
    int cubeWidth = pano->size().width;
    float cubeHeight = (0.75f) * pano->size().width;

    int edge = pano->size().width / 4;
    int face, startIndex, range;

    Mat cubemap;
    cubemap = Mat::zeros(cubeHeight, cubeWidth, pano->type());

    for (int x = 0; x < cubeWidth; x++)
    {
        face = x / edge;
        if (face == 1) // Left
        {
            startIndex = 0;
            range = 3 * edge;
        }
        else
        {
            startIndex = edge;
            range = 2 * edge;
        }
        int prev_face = face;

        for (int y = startIndex; y < range; y++)
        {
            if (y < edge) // Top
                face = 4;
            else if (y >= 2 * edge) // Bottom
                face = 5;


            float* point = new float[3];
            point = GetCubemapCoordinate(x, y, face, edge, point);

            float latitude;
            float longitude;
            int polarX;
            int polarY;

            latitude = atan2(point[1], point[0]);
            longitude = atan2(point[2], sqrt(pow(point[0], 2) + pow(point[1], 2)));
            polarX = 2 * edge * ((latitude + PI) / PI);
            polarY = 2 * edge * (((PI / 2) - longitude) / PI);
            // cout << polarX << " " <<  polarY << " " << face << endl;

            cubemap.at<Vec3b>(y, x) = pano->at<Vec3b>(polarY, polarX);

            face = prev_face;
        }
    }
    return cubemap;
}

int main(int ac, char** av) {

    /* 파노라마 이미지 불러오기 */
    Mat img = imread("Panorama.png"); //자신이 저장시킨 이미지 이름이 입력되어야 함, 확장자까지

    imshow("Spherical Panorama Image", img);
    waitKey(0);

     /* 큐브맵 저장 받을 Mat 변수 */
    int cubeWidth = img.size().width;
    float cubeHeight = (0.75f) * img.size().width;
    Mat Result;
    Result = Mat::zeros(cubeHeight, cubeWidth, img.type());

    /* 변환함수 */
    Result = CvtSph2Cub(&img);
    imshow("Cubemap Image", Result);
    waitKey(0);

    return 0;
}