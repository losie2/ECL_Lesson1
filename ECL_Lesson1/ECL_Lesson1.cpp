
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <math.h>
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

Mat CvtCub2Sph(Mat* cube, Mat *original) {
    /*
        구면 파노라마 이미지의 길이는 큐브 맵의 길이를 따른다.
        구면 파노라마 이미지의 높이는 큐브 맵 길이의 1/2. 
    */
    int Width = cube->size().width;
    float Height = (0.5f) * cube->size().width;

    Mat spherical;
    spherical = Mat::zeros(original->size().height, original->size().width, cube->type());

    /*
        좌표계를 0부터 1로 정규화 한다. (0, 0)
        경도를 나타내기 위한 변수 phi
        위도를 나타내기 위한 변수 theta
    */
    float u, v;
    float phi, theta;
    int cubeFaceWidth, cubeFaceHeight;

    cubeFaceWidth = cube->size().width / 4;
    cubeFaceHeight = cube->size().height / 3; //3 vertical faces


    for (int j = 0; j < Height; j++)
    {
        /*
            (i = 0, j = 0) 부터 j를 높이까지 증가.
            즉, 구면의 위도 생성.
            왼쪽 아래부터 시작.
        */
        v = 1 - ((float)j / Height);
        theta = v * PI;

        for (int i = 0; i < Width; i++)
        {
            // 위도 상의 한 점(0부터)에서 경도 끝까지 증가
            u = ((float)i / Width);
            phi = u * 2 * PI;

            float x, y, z; //Unit vector
            x = cos(phi) * sin(theta) * -1;
            y = sin(phi) * sin(theta) * -1;
            z = cos(theta);

            float xa, ya, za;
            float a;

            a = max(abs(x), max(abs(y), abs(z)));

            //Vector Parallel to the unit vector that lies on one of the cube faces
            xa = x / a;
            ya = y / a;
            za = z / a;

            int xPixel, yPixel;
            int xOffset, yOffset;

            if (ya == 1)
            {
                //Right
                xPixel = (int)((((xa + 1.f) / 2.f) - 1.f) * cubeFaceWidth);
                xOffset = 3 * cubeFaceWidth; //Offset
                yPixel = (int)((((za + 1.f) / 2.f)) * cubeFaceHeight);
                yOffset = cubeFaceHeight; //Offset

            }
            else if (ya == -1)
            {
                //Left
                xPixel = (int)((((xa + 1.f) / 2.f) + 1.f) * cubeFaceWidth);
                xOffset = 0;
                yPixel = (int)((((za + 1.f) / 2.f)) * cubeFaceHeight);
                yOffset = cubeFaceHeight;
            }
            else if (za == 1)
            {
                //Bottom
                xPixel = (int)((((xa + 1.f) / 2.f)) * cubeFaceWidth);
                xOffset = cubeFaceWidth;
                yPixel = (int)((((ya + 1.f) / 2.f)) * cubeFaceHeight);
                yOffset = 2 * cubeFaceHeight;
            }
            else if (za == -1)
            {
                //Top
                xPixel = (int)((((xa + 1.f) / 2.f)) * cubeFaceWidth);
                xOffset = cubeFaceWidth;
                yPixel = (int)((((ya + 1.f) / 2.f) - 1.f) * cubeFaceHeight);
                yOffset = 0;
            }
            else if (xa == -1)
            {
                //Back
                xPixel = (int)((((ya + 1.f) / 2.f) -1.f) * cubeFaceWidth);
                xOffset = 0;
                yPixel = (int)((((za + 1.f) / 2.f)) * cubeFaceHeight);
                yOffset = cubeFaceHeight;
            }
            else if (xa == 1)
            {
                //Front
                xPixel = (int)((((ya + 1.f) / 2.f)) * cubeFaceWidth);
                xOffset = 2 * cubeFaceWidth;
                yPixel = (int)((((za + 1.f) / 2.f)) * cubeFaceHeight);
                yOffset = cubeFaceHeight;
            }
            else
            {
                xPixel = 0;
                yPixel = 0;
                xOffset = 0;
                yOffset = 0;
            }

            xPixel = abs(xPixel);
            yPixel = abs(yPixel);

            xPixel += xOffset;
            yPixel += yOffset;


            spherical.at<Vec3b>(j, i) = cube->at<Vec3b>(yPixel, xPixel);
        }
    }
    return spherical;
}


Mat CvtSph2Cub(Mat* pano) {

    /*
       구면 파노라마 이미지 너비, 높이 구하기
       이 때 cubeWidth는 6칸(Back, Left, Front, Right, Top, Bottom)으로 나누어진 맵의 가로값. 파노라마 이미지의 가로값과 같음
       cubeHeight는 맵의 세로값. 즉 (Top, Left, Bottom) 세 칸을 차지함.
       본 이미지는 1:2 비율을 갖고 4칸:3칸이므로 0.75f를 곱해줌.
    */
    int cubeWidth = pano->size().width;
    float cubeHeight = (0.75f) * pano->size().width;

    /*
        edge(한 큐브맵의 선)는 정사각형의 한 선이므로 width / 4
    */
    int edge = pano->size().width / 4;
    int face, startIndex, range;


    /*
        Mat은 col, row를 사용.
    */
    Mat cubemap;
    cubemap = Mat::zeros(cubeHeight, cubeWidth, pano->type());

    /*
        가로(cubeWidth)는 동일한 상태에서
        세로3칸 가로4칸을 수행하기 위해 face가 1(가로) 혹은 4,5(세로) 일 때의 경우 범위를 다르게 접근.
    */
    for (int x = 0; x < cubeWidth; x++)
    {
        face = x / edge;
        /*
            Left(face = 1)일 경우. 가로 범위이기 때문에 top, bottom은 고려할 필요가 없음 
            face = 0, (1, 4, 5,) 2, 3 으로 진행.
        */
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

        /* face가 1일 때(Left) 아래 조건에 걸려 face가 4, 5로 변경되는 것을 막아주기 위한 변수. */
        int prev_face = face;

        for (int y = startIndex; y < range; y++)
        {
            if (y < edge) // Top
                face = 4;
            else if (y >= 2 * edge) // Bottom
                face = 5;


            /* 
            
                1. 구면 파노라마 이미지 좌표를 큐브맵에 배치 
                2. edge 값에 따라 face가 결정. face에 따른 3차원 좌표값(구면 파노라마 이미지 좌표계)을 큐브맵 좌표계(2차원)으로 투영.
                3. 투영된 2차원 좌표값을 구면 좌표계로 변환
                4. 변환된 구면 좌표계를 구면 파노라마 이미지의 좌표값으로 배치.

            */
            float* point = new float[3];
            point = GetCubemapCoordinate(x, y, face, edge, point);

            // 경도값
            float latitude; 

            // 위도값
            float longitude;

            // 큐브맵 좌표계
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


     /* 큐브맵 저장 받을 Mat 변수 */
    int cubeWidth = img.size().width;
    float cubeHeight = (0.75f) * img.size().width;
    Mat Result;
    Result = Mat::zeros(cubeHeight, cubeWidth, img.type());

    /* 변환함수 */
    Result = CvtSph2Cub(&img);

    Mat Result2;
    Result2 = Mat::zeros(img.size().height, img.size().width, img.type());
    Result2 = CvtCub2Sph(&Result, &img);

    imshow("Spherical Panorama Image", img);
    imshow("Cubemap Image", Result);
    imshow("Spherical Image", Result2);
    waitKey(0);

    return 0;
}