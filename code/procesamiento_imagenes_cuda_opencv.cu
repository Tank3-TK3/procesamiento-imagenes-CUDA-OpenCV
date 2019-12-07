//////////////////////////////////////////////////
//      LIBRERIAS
#include <iostream>
#include <cstdio>
#include <cmath>
#include <string>
#include <ctime>
//      CUDA
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//      OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
//////////////////////////////////////////////////
#define numFHpB 4 //Numero de Filas de Hilos por Bloque
#define numCHpB 4 //Numero de Columnas de Hilos por Bloque

using namespace cv;
using namespace std;
//////////////////////////////////////////////////
//      CLASE
//template <typename DataType>
class ImagenCuda{
public:
    //HOST
    uchar4 *img1; //Imagen 1
    uchar4 *img2; //Imagen 1
    uchar4 *imgSalCPU; //Imagen salida de CPU
    uchar4 *imgSalGPU; //Imagen salida de GPU
    string nombreImg1; //Nombre de la imagen 1
    string nombreImg2; //Nombre de la imagen 2
    Mat imgM1; //Imagen en MAT 1
    Mat imgM2; //Imagen en MAT 2
    Mat imgRGBA1; //Imagen Red, Green, Blue, Alpha 1
    Mat imgRGBA2; //Imagen Red, Green, Blue, Alpha 2
    Mat sal1; //Salida 1
    Mat sal2; //Salida 2
    int filas;
    int columnas;
    int indice;
    size_t numPixeles;
    //DEVICE
    uchar4 *dimg1;
    uchar4 *dimg2;
    uchar4 *dimgSalida;
    //CONSTRUCTOR
    ImagenCuda(Mat imgM1, Mat imgM2)
    {
        //HOST
        this->imgM1 = imgM1;
        this->imgM2 = imgM2;
        cvtColor(imgM1, imgRGBA1, CV_BGR2RGBA);
        cvtColor(imgM2, imgRGBA2, CV_BGR2RGBA);
        this->filas = imgM1.rows;
        this->columnas = imgM2.cols;
        this->numPixeles = filas * columnas;
        this->img1 = (uchar4 *) imgRGBA1.ptr<unsigned char>(0);
        this->img2 = (uchar4 *) imgRGBA2.ptr<unsigned char>(0);
        this->imgSalCPU = (uchar4 *) malloc(sizeof(uchar4) * numPixeles);
        this->imgSalGPU = (uchar4 *) malloc(sizeof(uchar4) * numPixeles);
        memset(imgSalCPU, 0, sizeof(uchar4) * numPixeles);
        //DEVICE
        cudaMalloc(&dimg1, sizeof(uchar4) * numPixeles);
        cudaMalloc(&dimg2, sizeof(uchar4) * numPixeles);
        cudaMalloc(&dimgSalida, sizeof(uchar4) * numPixeles);
        cudaMemset(dimgSalida, 0, sizeof(uchar4) * numPixeles);
        cudaMemcpy(dimg1, img1, sizeof(uchar4) * numPixeles, cudaMemcpyHostToDevice);
        cudaMemcpy(dimg2, img2, sizeof(uchar4) * numPixeles, cudaMemcpyHostToDevice);
    };
    //DESTRUCTOR
   /* ~ImagenCuda()
    {
        //free HOST
        if (img1 != NULL) free(img1);
        if (img2 != NULL) free(img2);
        if (imgSalCPU != NULL) free(imgSalCPU);
        if (imgSalGPU != NULL) free(imgSalGPU);
        //free DEVICE
        if (dimg1 != NULL) cudaFree(dimg1);
        if (dimg2 != NULL) cudaFree(dimg2);
        if (dimgSalida != NULL) cudaFree(dimgSalida);
    };*/
    //METODOS
    void restaCPU()
    {
        int val1 = 0;
        int val2 = 0;
        int val3 = 0;
        for (int i = 0; i < (filas*columnas); i++)
        {
            val1 = img1[i].x - img2[i].x;
            val2 = img1[i].y - img2[i].y;
            val3 = img1[i].z - img2[i].z;
            if (val1<0)    val1 = 0;
            if (val1>255)  val1 = 255;
            if (val2<0)    val2 = 0;
            if (val2>255)  val2 = 255;
            if (val3<0)    val3 = 0;
            if (val3>255)  val3 = 255;
            imgSalCPU[i].x = (uchar)val1;
            imgSalCPU[i].y = (uchar)val2;
            imgSalCPU[i].z = (uchar)val3;
            imgSalCPU[i].w = img1[i].w; //Alfa
        }
    };

    void regresaDatos(Mat temp1, Mat temp2)
    {
        cudaDeviceSynchronize();
        cudaMemcpy(imgSalGPU, dimgSalida, sizeof(uchar4) * numPixeles, cudaMemcpyDeviceToHost);
        cvtColor(temp1, sal1, CV_RGBA2BGR);
        cvtColor(temp2, sal2, CV_RGBA2BGR);
    };
};
//////////////////////////////////////////////////
//      FUNCIONES
__global__ void restaGPU(ImagenCuda a)
{
    int fila = (blockIdx.x * blockDim.x) + threadIdx.x;
    int columna = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (columna >= a.columnas || fila >= a.filas)
    {
        return;
    }
    int indice = fila * a.columnas + columna;
    int val1 = 0;
    int val2 = 0;
    int val3 = 0;
    val1 = a.dimg1[indice].x - a.dimg2[indice].x;
    val2 = a.dimg1[indice].y - a.dimg2[indice].y;
    val3 = a.dimg1[indice].z - a.dimg2[indice].z;
    if (val1<0)    val1 = 0;
    if (val1>255)  val1 = 255;
    if (val2<0)    val2 = 0;
    if (val2>255)  val2 = 255;
    if (val3<0)    val3 = 0;
    if (val3>255)  val3 = 255;
    a.dimgSalida[indice].x = (uchar)val1;
    a.dimgSalida[indice].y = (uchar)val2;
    a.dimgSalida[indice].z = (uchar)val3;
    a.dimgSalida[indice].w = a.dimg1[indice].w; //Alfa
}
//////////////////////////////////////////////////
//      MAIN
int main(int argc, char **argv){
    if (argv[1] == NULL)
    {
        cout << "No argumento 1...\n";
        return 0;
    }
    else if(argv[2] == NULL)
    {
        cout << "No argumento 2...\n";
        return 0;

    }

    cout << "Argumento 1: " << argv[1] << "\n";
    cout << "Argumento 2: " << argv[2] << "\n";

    const string nombreImagen1 = argv[1];
    const string nombreImagen2 = argv[2];

    Mat imgA = imread(nombreImagen1.c_str(), CV_LOAD_IMAGE_COLOR);
    Mat imgB = imread(nombreImagen2.c_str(), CV_LOAD_IMAGE_COLOR);

    if (imgA.empty())
    {
        cout << "No imagen 1...\n";
        return 0;
    }
    else if(imgB.empty())
    {
        cout << "No imagen 2...\n";
        return 0;

    }else{

        cout << "Canales imgA -> " << imgA.channels() << "\n";
        cout << "Canales imgB -> " << imgB.channels() << "\n";
    }

    ImagenCuda imagenA(imgA, imgB);

    clock_t timer1 = clock();
    imagenA.restaCPU();
    timer1 = clock() - timer1;
    cout << "Tiempo en CPU: " << ((float) timer1) << "\n";

    const dim3 gridSize((imagenA.filas / numFHpB) + 1, (imagenA.columnas / numCHpB) + 1, 1);
    const dim3 blockSize(numFHpB, numCHpB, 1);
    timer1 = clock();
    restaGPU << <gridSize, blockSize >> > (imagenA);
    timer1 = clock() - timer1;
    cout << "Tiempo en GPU: " << ((float) timer1) << "\n";

    cout << "Configuracion de ejecucion: \n";
    cout << "Grid [" << gridSize.x << ", " << gridSize.y << ", " << gridSize.z << "]\n";
    cout << "Bloque [" << blockSize.x <<", " << blockSize.y << ", " << blockSize.z << "]\n";

    ////////////////////////
    //comparar cpu con gpu//
    ////////////////////////

    Mat temp1(imagenA.filas, imagenA.columnas, CV_8UC4, imagenA.imgSalCPU);
        Mat temp2(imagenA.filas, imagenA.columnas, CV_8UC4, imagenA.imgSalGPU);

    imagenA.regresaDatos(temp1, temp2);

    imshow("Imagen 1", imgA);
    imshow("Imagen 2", imgB);
    imshow("Salida CPU", imagenA.sal1);
    imshow("Salida GPU", imagenA.sal2);

    imwrite("salidaCPU.png", imagenA.sal1);
    imwrite("salidaGPU.png", imagenA.sal2);

    waitKey(0);

    //FREE HOST
    free(imagenA.imgSalCPU);
    free(imagenA.imgSalGPU);
    //FREE DEVICE
    cudaFree(imagenA.dimg1);
    cudaFree(imagenA.dimg2);
    cudaFree(imagenA.dimgSalida);

    return 0;
}
