# UNetMultiLane_rknn_Cplusplus

UNetMultiLane 瑞芯微 rknn 板端 C++部署，使用平台 rk3588。

模型转换和仿真测试参考 [onnx转rknn](https://blog.csdn.net/zhangqian_1/article/details/139591990)。

## 编译和运行

1）编译

```
cd examples/rknn_UNetMultiLaneSeg_demo

bash build-linux_RK3588.sh

```

2）运行

```
cd install/rknn_UNetMultiLaneSeg_demo_Linux

./rknn_UNetMultiLaneSeg_demo

```

注意：修改模型、测试图像、保存图像的路径，修改文件为src下的main.cc

```

int main(int argc, char **argv)
{
    char model_path[256] = "/home/firefly/zhangqian/UNetMultiLane_rknn_Cplusplus/examples/rknn_UNetMultiLaneSeg_demo/model/RK3588/UNetMultiLane_seg.rknn";
    char image_path[256] = "/home/firefly/zhangqian/UNetMultiLane_rknn_Cplusplus//examples/rknn_UNetMultiLaneSeg_demo/test.jpg";
    char save_image_path[256] = "/home/firefly/zhangqian/UNetMultiLane_rknn_Cplusplus/rknn/examples/rknn_UNetMultiLaneSeg_demo/test_result.jpg";

    detect(model_path, image_path, save_image_path);
    return 0;
}
```

# 板端测试效果
![image](https://github.com/cqu20160901/UNetMultiLane_rknn_Cplusplus/blob/main/examples/rknn_UNetMultiLaneSeg_demo/test_result.jpg)
