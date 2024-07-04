#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define _BASETSD_H

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rga.h"
#include "rknn_api.h"
#include <dirent.h>

#define PERF_WITH_POST 1

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++)
    {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

int PostProcess(int8_t **PtrBlob, std::vector<int> &QntZp, std::vector<float> &QntScale, int (*SegMask)[640], int *LineType)
{
    int8_t *PtrSeg = (int8_t *)PtrBlob[0];
    int8_t *PtrCls = (int8_t *)PtrBlob[1];

    int InputH = 480;
    int InputW = 640;
    int SegOutputC = 9;

    int SegVal = 0;
    int SegMaxVal = 0;
    int MaxValIndex = 0;

    for (int h = 0; h < InputH; h++)
    {
        for (int w = 0; w < InputW; w++)
        {

            for (int c = 0; c < SegOutputC; c++)
            {
                SegVal = PtrSeg[InputH * InputW * c + InputW * h + w];
                if (0 == c)
                {
                    SegMaxVal = SegVal;
                    MaxValIndex = c;
                }
                else
                {
                    if (SegVal > SegMaxVal)
                    {
                        SegMaxVal = SegVal;
                        MaxValIndex = c;
                    }
                }
            }
            SegMask[h][w] = MaxValIndex;
        }
    }

    int ClsOutputC = 8;
    int ClsOutputCls = 11;
    int ClsVal = 0;
    int ClsMaxVal = 0;
    int ClsValIndex = 0;
    for (int i = 0; i < ClsOutputC; i++)
    {
        for (int j = 0; j < ClsOutputCls; j++)
        {
            ClsVal = PtrCls[ClsOutputCls * i + j];
            if (0 == j)
            {
                ClsMaxVal = ClsVal;
                ClsValIndex = j;
            }
            else
            {
                if (ClsVal > ClsMaxVal)
                {
                    ClsMaxVal = ClsVal;
                    ClsValIndex = j;
                }
            }
        }
        LineType[i] = ClsValIndex;
    }
    return 1;
}

int detect(char *model_path, char *image_path, char *save_image_path)
{
    rknn_context ctx;
    int img_width = 0;
    int img_height = 0;
    struct timeval start_time, stop_time;
    int ret;

    // init rga context
    rga_buffer_t src;
    rga_buffer_t dst;
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

    printf("Read %s ...\n", image_path);
    cv::Mat orig_img = cv::imread(image_path, 1);
    if (!orig_img.data)
    {
        printf("cv::imread %s fail!\n", image_path);
        return -1;
    }
    cv::Mat img;
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);

    img_width = img.cols;
    img_height = img.rows;

    printf("img width = %d, img height = %d\n", img_width, img_height);

    /* Create the neural network */
    int model_data_size = 0;
    unsigned char *model_data = load_model(model_path, &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    int channel = 3;
    int width = 0;
    int height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    // You may not need resize when src resulotion equals to dst resulotion
    void *resize_buf = nullptr;

    if (img_width != width || img_height != height)
    {
        printf("resize with RGA!\n");
        resize_buf = malloc(height * width * channel);
        memset(resize_buf, 0x00, height * width * channel);

        src = wrapbuffer_virtualaddr((void *)img.data, img_width, img_height, RK_FORMAT_RGB_888);
        dst = wrapbuffer_virtualaddr((void *)resize_buf, width, height, RK_FORMAT_RGB_888);
        ret = imcheck(src, dst, src_rect, dst_rect);
        if (IM_STATUS_NOERROR != ret)
        {
            printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
            return -1;
        }
        IM_STATUS STATUS = imresize(src, dst);
        // cv::Mat resize_img(cv::Size(width, height), CV_8UC3, resize_buf);
        // cv::imwrite("resize_input.jpg", resize_img);

        inputs[0].buf = resize_buf;
    }
    else
    {
        inputs[0].buf = (void *)img.data;
    }

    gettimeofday(&start_time, NULL);
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0;
    }

    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    gettimeofday(&stop_time, NULL);

    printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    std::vector<float> OutScales;
    std::vector<int32_t> OutZps;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        OutScales.push_back(output_attrs[i].scale);
        OutZps.push_back(output_attrs[i].zp);
    }

    int8_t *pblob[2];
    for (int i = 0; i < io_num.n_output; ++i)
    {
        pblob[i] = (int8_t *)outputs[i].buf;
    }

    int SegMask[480][640];
    int LineType[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    ret = PostProcess(pblob, OutZps, OutScales, SegMask, LineType);

    cv::Mat SegMaskMat = cv::Mat(cv::Size(640, 480), CV_8UC3, cv::Scalar(0, 0, 0));
    int ColorList[10][3] = {{100, 149, 237}, {0, 0, 255}, {173, 255, 47}, {240, 255, 255}, {0, 100, 0}, {47, 79, 79}, {255, 228, 196}, {138, 43, 226}, {165, 42, 42}, {222, 184, 135}};

    int temp = 0;
    for (int i = 0; i < 480; i++)
    {
        for (int j = 0; j < 640; j++)
        {
            temp = SegMask[i][j];
            if (0 != temp)
            {
                SegMaskMat.at<cv::Vec3b>(i, j)[0] = ColorList[temp][0];
                SegMaskMat.at<cv::Vec3b>(i, j)[1] = ColorList[temp][1];
                SegMaskMat.at<cv::Vec3b>(i, j)[2] = ColorList[temp][2];
            }
        }
    }

    cv::Mat SegResult;
    cv::resize(SegMaskMat, SegResult, cv::Size(img_width, img_height), 0, 0, cv::INTER_LINEAR);

    std::vector<std::string> LineTypeEmu = {"No lane markings",
                                            "Single white solid line",
                                            "Single white dashed line",
                                            "Single solid yellow line",
                                            "Single yellow dashed line",
                                            "Double solid white lines",
                                            "Double solid yellow lines",
                                            "Double yellow dashed lines",
                                            "Double white yellow solid lines",
                                            "Double white dashed lines",
                                            "Double white solid dashed lines"};

    char text[256];
    for (int i = 0; i < 8; i++)
    {
        if (0 == LineType[i])
        {
            continue;
        }
        sprintf(text, "%d:%d:%s", i, LineType[i], LineTypeEmu[LineType[i]].c_str());
        putText(orig_img, text, cv::Point(25, i * 25 + 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }

    cv::Mat ImgAddMask;
    cv::addWeighted(orig_img, 0.8, SegResult, 0.2, 0, ImgAddMask, -1);
    cv::imwrite(save_image_path, ImgAddMask);

    // release
    ret = rknn_destroy(ctx);

    if (model_data)
    {
        free(model_data);
    }

    if (resize_buf)
    {
        free(resize_buf);
    }

    return 0;
}

int main(int argc, char **argv)
{
    printf("============= This is main ... ==============\n");

    char model_path[256] = "/home/firefly/zhangqian/rknn/rknpu2_1.4.0_20220909/examples/rknn_UnetMUtilLaneSeg_demo/model/RK3588/UNet_mutilLane_202305011.rknn";
    char image_path[256] = "/home/firefly/zhangqian/rknn/rknpu2_1.4.0_20220909/examples/rknn_UnetMUtilLaneSeg_demo/test.jpg";
    char save_image_path[256] = "/home/firefly/zhangqian/rknn/rknpu2_1.4.0_20220909/examples/rknn_UnetMUtilLaneSeg_demo/test_result.jpg";

    int ret = detect(model_path, image_path, save_image_path);
    printf("============= inference finished ! ==============\n");

    return 0;
}
