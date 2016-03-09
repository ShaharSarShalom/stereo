/*
 *  stereo_match.cpp
 *  calibration
 *
 *  Created by Victor  Eruhimov on 1/18/10.
 *  Copyright 2010 Argus Corp. All rights reserved.
 *
 */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <stdio.h>
#include "imageLib.h"

using namespace cv;

static void print_help()
{
    printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
    printf("\nUsage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i <intrinsic_filename>] [-e <extrinsic_filename>]\n"
           "[--no-display] [-o <disparity_image>] [-p <point_cloud_file>]\n");
}

static void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

CFloatImage ConvertMatToCImage(Mat mat) {
    CFloatImage img(mat.cols, mat.rows, 1);
    for (int x = 0; x < mat.cols; x++) {
        for (int y = 0; y < mat.rows; y++) {
            float* ptr = (float *) img.PixelAddress(x, y, 0);
            *ptr = mat.at<float>(y, x)/16;
            if (*ptr < 0)
                *ptr = INFINITY;
        }
    }
    return img;
}

// // write pfm image (added by DS 10/24/2013)
// // 1-band PFM image, see http://netpbm.sourceforge.net/doc/pfm.html
// void WriteFilePFM(float *data, int width, int height, const char* filename, float scalefactor=1/255.0)
// {
//     // Open the file
//     FILE *stream = fopen(filename, "wb");
//     if (stream == 0) {
//         fprintf(stderr, "WriteFilePFM: could not open %s\n", filename);
// 	exit(1);
//     }
//
//     // sign of scalefact indicates endianness, see pfms specs
//     if (littleendian())
// 	scalefactor = -scalefactor;
//
//     // write the header: 3 lines: Pf, dimensions, scale factor (negative val == little endian)
//     fprintf(stream, "Pf\n%d %d\n%f\n", width, height, scalefactor);
//
//     int n = width;
//     // write rows -- pfm stores rows in inverse order!
//     for (int y = height-1; y >= 0; y--) {
// 	float* ptr = data + y * width;
// 	// change invalid pixels (which seem to be represented as -10) to INF
// 	for (int x = 0; x < width; x++) {
// 	    if (ptr[x] < 0)
// 		ptr[x] = INFINITY;
//         if (ptr[x] > 1/abs(scalefactor)) {
//             ptr[x] = 1/abs(scalefactor);
//         }
// 	}
// 	if ((int)fwrite(ptr, sizeof(float), n, stream) != n) {
// 	    fprintf(stderr, "WriteFilePFM: problem writing data\n");
// 	    exit(1);
// 	}
//     }
//     // close file
//     fclose(stream);
//
//     CFloatImage img;
//     int verbose = 0;
//     ReadImageVerb(img, filename, verbose);
// }

int main(int argc, char** argv)
{
    const char* algorithm_opt = "--algorithm=";
    const char* maxdisp_opt = "--max-disparity=";
    const char* blocksize_opt = "--blocksize=";
    const char* scale_opt = "--scale=";

    if(argc < 3)
    {
        print_help();
        return 0;
    }
    const char* img1_filename = 0;
    const char* img2_filename = 0;
    const char* intrinsic_filename = 0;
    const char* extrinsic_filename = 0;
    const char* disparity_filename = 0;
    const char* point_cloud_filename = 0;

    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3 };
    int alg = STEREO_SGBM;
    int SADWindowSize = 0, numberOfDisparities = 0;
    float scale = 1.f;

    Ptr<StereoBM> bm = StereoBM::create(16,9);
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);

    for( int i = 1; i < argc; i++ )
    {
        if( argv[i][0] != '-' )
        {
            if( !img1_filename )
                img1_filename = argv[i];
            else
                img2_filename = argv[i];
        }
        else if( strncmp(argv[i], algorithm_opt, strlen(algorithm_opt)) == 0 )
        {
            char* _alg = argv[i] + strlen(algorithm_opt);
            alg = strcmp(_alg, "bm") == 0 ? STEREO_BM :
                  strcmp(_alg, "sgbm") == 0 ? STEREO_SGBM :
                  strcmp(_alg, "hh") == 0 ? STEREO_HH :
                  strcmp(_alg, "var") == 0 ? STEREO_VAR : -1;
            if( alg < 0 )
            {
                printf("Command-line parameter error: Unknown stereo algorithm\n\n");
                print_help();
                return -1;
            }
        }
        else if( strncmp(argv[i], maxdisp_opt, strlen(maxdisp_opt)) == 0 )
        {
            if( sscanf( argv[i] + strlen(maxdisp_opt), "%d", &numberOfDisparities ) != 1 ||
                numberOfDisparities < 1 )
            {
                printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer\n");
                print_help();
                return -1;
            }
        }
        else if( strncmp(argv[i], blocksize_opt, strlen(blocksize_opt)) == 0 )
        {
            if( sscanf( argv[i] + strlen(blocksize_opt), "%d", &SADWindowSize ) != 1 ||
                SADWindowSize < 1 || SADWindowSize % 2 != 1 )
            {
                printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
                return -1;
            }
        }
        else if( strncmp(argv[i], scale_opt, strlen(scale_opt)) == 0 )
        {
            if( sscanf( argv[i] + strlen(scale_opt), "%f", &scale ) != 1 || scale < 0 )
            {
                printf("Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number\n");
                return -1;
            }
        }
        else if( strcmp(argv[i], "-i" ) == 0 )
            intrinsic_filename = argv[++i];
        else if( strcmp(argv[i], "-e" ) == 0 )
            extrinsic_filename = argv[++i];
        else if( strcmp(argv[i], "-o" ) == 0 )
            disparity_filename = argv[++i];
        else if( strcmp(argv[i], "-p" ) == 0 )
            point_cloud_filename = argv[++i];
        else
        {
            printf("Command-line parameter error: unknown option %s\n", argv[i]);
            return -1;
        }
    }

    if( !img1_filename || !img2_filename || !disparity_filename)
    {
        printf("Command-line parameter error: both left and right images and the output image must be specified\n");
        return -1;
    }

    if( (intrinsic_filename != 0) ^ (extrinsic_filename != 0) )
    {
        printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
        return -1;
    }

    if( extrinsic_filename == 0 && point_cloud_filename )
    {
        printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
        return -1;
    }

    int color_mode = alg == STEREO_BM ? 0 : -1;
    Mat img1 = imread(img1_filename, color_mode);
    Mat img2 = imread(img2_filename, color_mode);

    if (img1.empty())
    {
        printf("Command-line parameter error: could not load the first input image file\n");
        return -1;
    }
    if (img2.empty())
    {
        printf("Command-line parameter error: could not load the second input image file\n");
        return -1;
    }

    if (scale != 1.f)
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }

    Size img_size = img1.size();

    Rect roi1, roi2;
    Mat Q;

    if( intrinsic_filename )
    {
        // reading intrinsic parameters
        FileStorage fs(intrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", intrinsic_filename);
            return -1;
        }

        Mat M1, D1, M2, D2;
        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;

        M1 *= scale;
        M2 *= scale;

        fs.open(extrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", extrinsic_filename);
            return -1;
        }

        Mat R, T, R1, P1, R2, P2;
        fs["R"] >> R;
        fs["T"] >> T;

        stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

        Mat img1r, img2r;
        remap(img1, img1r, map11, map12, INTER_LINEAR);
        remap(img2, img2r, map21, map22, INTER_LINEAR);

        img1 = img1r;
        img2 = img2r;
    }

    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;

    bm->setROI1(roi1);
    bm->setROI2(roi2);
    bm->setPreFilterCap(31);
    bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
    bm->setMinDisparity(0);
    bm->setNumDisparities((numberOfDisparities-1)/16*16);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(15);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);

    sgbm->setPreFilterCap(63);
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);

    int cn = img1.channels();

    sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities((numberOfDisparities-1)/16*16);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(alg == STEREO_HH ? StereoSGBM::MODE_HH : StereoSGBM::MODE_SGBM);

    Mat disp, disp8;
    //Mat img1p, img2p, dispp;
    //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
    //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

    int64 t = getTickCount();
    if( alg == STEREO_BM )
        bm->compute(img1, img2, disp);
    else if( alg == STEREO_SGBM || alg == STEREO_HH )
        sgbm->compute(img1, img2, disp);
    t = getTickCount() - t;
    printf("Time elapsed: %fms\n", t*1000/getTickFrequency());

    disp.convertTo(disp, DataType<float>::type);
    WriteFilePFM(ConvertMatToCImage(disp), disparity_filename, 1.0f/(numberOfDisparities-1));
    // if( alg != STEREO_VAR )
    //     disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
    // else
    //     disp.convertTo(disp8, CV_8U);
    // imwrite("temp.png", disp8);

    // CFloatImage gtdisp;
    // int verbose;
    // ReadImageVerb(gtdisp, "trainingQ/Vintage/disp0GT.pfm", verbose);
    // CByteImage mask;
    // ReadImageVerb(mask, "trainingQ/Vintage/mask0nocc.png", verbose);
    //
    // int width = gtdisp.Shape().width, height = gtdisp.Shape().height;
    // int n = 0;
    // int bad = 0;
    // int invalid = 0;
    // float serr = 0;
    // int badthresh = 10;
    // for (int y = 0; y < height; y++) {
    // for (int x = 0; x < width; x++) {
    //     float gt = gtdisp.Pixel(x, y, 0);
    //     if (gt == INFINITY) // unknown
    //     continue;
    //     float d = disp.at<float>(y, x);
    //     int valid = (d >=0 );
    //     d = round(d/16);
    //     float err = fabs(d - gt);
    //     if (mask.Pixel(x, y, 0) != 255) { // don't evaluate pixel
    //     } else {
    //     n++;
    //     if (valid) {
    //         serr += err;
    //         if (err > badthresh) {
    //         bad++;
    //         }
    //     } else {// invalid (i.e. hole in sparse disp map)
    //         invalid++;
    //     }
    //     }
    // }
    // }
    // float badpercent =  100.0*bad/n;
    // float invalidpercent =  100.0*invalid/n;
    // float totalbadpercent =  100.0*(bad+invalid)/n;
    // float avgErr = serr / (n - invalid); // CHANGED 10/14/2014 -- was: serr / n
    // //printf("mask  bad%.1f  invalid  totbad   avgErr\n", badthresh);
    // printf("%4.1f  %6.2f  %6.2f   %6.2f  %6.2f\n",   100.0*n/(width * height),
	//    badpercent, invalidpercent, totalbadpercent, avgErr);

    if(point_cloud_filename)
    {
        printf("storing the point cloud...");
        fflush(stdout);
        Mat xyz;
        reprojectImageTo3D(disp, xyz, Q, true);
        saveXYZ(point_cloud_filename, xyz);
        printf("\n");
    }

    return 0;
}
