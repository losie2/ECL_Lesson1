
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
    /*
        -1 ~ 1 사이 정규화된 좌표값 내에서 point를 정해주어야 한다.
        즉, 2의 길이가 필요하기 때문에 a, b에 2를 곱해준다.
        edge < y < 2*edge 사이 b = 2 * 1.??????f이고, 2~4의 범위를 갖는다. 
    */
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

        1. 구면 파노라마 이미지 좌표(j, i)를 정규화, 구면 좌표계로 변환한다.
        2. 구면 좌표계에 대응하는 큐브맵의 좌표를 찾는다.
        3. 찾은 큐브맵의 좌표가 어느 face에 있는지 찾는다.
        4. 큐브맵의 좌표를 구면 파노라마 이미지의 좌표(j, i)에 대입한다.
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
    cubeFaceHeight = cube->size().height / 3;


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

            /*
                큐브 면 중 하나에 있는 단위 벡터와 평행한 벡터.
                즉, ya와 평행하다면 xz 평면과 평행 하다는 의미.
                이 때, ya가 -1인지 1인지(Left, Right) 값을 보고 평면을 결정.
            */
            xa = x / a;
            ya = y / a;
            za = z / a;

            int xPixel, yPixel;
            int xOffset, yOffset;

            
            /*
                1. 정규화를 거친 좌표계이기 때문에 2.f로 나누어준다.
                2. -1 ~ 1로 정규화 되어있는 좌표계에 edge 길이(cubeFaceWidth, cubeFaceHeight)를 곱해준다.
                3. WorldOffset에 적용한다. 이때 Offset은 큐브맵 좌표계에 따른다. 
            */
            if (ya == -1)
            {
                //Left
                xPixel = (int)((((xa + 1.f) / 2.f)) * cubeFaceWidth);
                xOffset = cubeFaceWidth;
                yPixel = (int)((((za + 1.f) / 2.f)) * cubeFaceHeight);
                yOffset = cubeFaceHeight;
            }
            else if (ya == 1)
            {
                //Right
                xPixel = (int)((((xa - 1.f) / 2.f)) * cubeFaceWidth);
                xOffset = 3 * cubeFaceWidth;
                yPixel = (int)((((za + 1.f) / 2.f)) * cubeFaceHeight);
                yOffset = cubeFaceHeight; 
            }            
            else if (za == -1)
            {
                //Top
                xPixel = (int)((((xa + 1.f) / 2.f)) * cubeFaceWidth);
                xOffset = cubeFaceWidth;
                yPixel = (int)((((ya - 1.f) / 2.f)) * cubeFaceHeight);
                yOffset = 0;
            }
            else if (za == 1)
            {
                //Bottom
                xPixel = (int)((((xa + 1.f) / 2.f)) * cubeFaceWidth);
                xOffset = cubeFaceWidth;
                yPixel = (int)((((ya + 1.f) / 2.f)) * cubeFaceHeight);
                yOffset = 2 * cubeFaceHeight;
            }
            else if (xa == 1)
            {
                //Front
                xPixel = (int)((((ya + 1.f) / 2.f)) * cubeFaceWidth);
                xOffset = 2 * cubeFaceWidth;
                yPixel = (int)((((za + 1.f) / 2.f)) * cubeFaceHeight);
                yOffset = cubeFaceHeight;
            }
            else if (xa == -1)
            {
                //Back
                xPixel = (int)((((ya - 1.f) / 2.f)) * cubeFaceWidth);
                xOffset = 0;
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
        if (face == 1) // Left(Top, Bottom)
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
            float phi; 

            // 위도값
            float theta;

            // 큐브맵 좌표계
            int polarX;
            int polarY;


            /*
                atan2(a, b)는 원점에서 (a, b) 까지 상대적인 각도(위치)이다.
                phi는 경도이고, 원점에서 (point[1] = y, point[0] = x)까지의 상대적인 각도를 구한다.
                이 때 (point[1] = y, point[0] = x)는 큐브맵 위의 한 face 위 점이므로 구면 파노라마의 한 좌표로 표현하기 위해 atan2를 사용한다.

                theta는 위도이고, 90 ~ -90 degree의 범위를 가진다.
                범위를 벗어나게 하지 않기 위해 x, y 사이 거리를 2번째 인자로 준다.
                원점이 기준이므로 원점과 (point[2] = z, 거리) 사이의 각도이다.

            */
            phi = atan2(point[1], point[0]);
            theta = atan2(point[2], sqrt(pow(point[0], 2) + pow(point[1], 2)));


            /*
                (phi + PI) / PI는 phi 만큼 회전한 구면 위의 이미지 상 (x, y)좌표이다.
                ((PI / 2) - theta) / PI는 90 ~ -90 범위내에서 theta만큼 회전한 구면 위의 이미지 상 (x, y, z)좌표이다.
                이렇게 구한 구면 파노라마 이미지 상의 좌표를 큐브맵 이미지 좌표계에 대입한다.
            */
            polarX = 2 * edge * ((phi + PI) / PI);
            polarY = 2 * edge * (((PI / 2) - theta) / PI);

            cubemap.at<Vec3b>(y, x) = pano->at<Vec3b>(polarY, polarX);

            face = prev_face;
        }       
    }
    return cubemap;
}

int main(int ac, char** av) {

    /* 파노라마 이미지 불러오기 */
    Mat img = imread("Panorama.png"); //자신이 저장시킨 이미지 이름이 입력되어야 함, 확장자까지


     /* Mat 선언 */
    Mat SphericalToCubemap;
    SphericalToCubemap = Mat::zeros((0.75f) * img.size().width, img.size().width, img.type());
    Mat CubemapToSpherical;
    CubemapToSpherical = Mat::zeros(img.size().height, img.size().width, img.type());

    /* 변환함수 */
    SphericalToCubemap = CvtSph2Cub(&img);
    CubemapToSpherical = CvtCub2Sph(&SphericalToCubemap, &img);

    /* 이미지 출력 */
    imshow("Spherical Panorama Original Image", img);
    imshow("Cubemap Image", SphericalToCubemap);
    imshow("Spherical Image", CubemapToSpherical);
    waitKey(0);

    return 0;
}