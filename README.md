# OpenCV_Aruco
使用C++ OpenCV实现椭圆区域检测与Aruco码的生成与检测并估计位姿


博客：[https://blog.csdn.net/sxf1061700625/article/details/125658466](https://blog.csdn.net/sxf1061700625/article/details/125658466)

ED_Lib：[椭圆检测](https://github.com/CihanTopal/ED_Lib)    
```bash
git clone git@github.com:CihanTopal/ED_Lib.git
```
> 如果下载完ED_Lib后make报错，就在ED_Lib下添加`ArucoDetect.cpp`和`ArucoDetect.h`（或者直接删除这个的调用，因为目前什么都没做）：     
>** ArucoDetect.cpp**：
> ```c++
> //
> // Created by sxf on 22-4-23.
> //
> 
> #include "ArucoDetect.h"
> 
> namespace sxf {
> } // sxf
> ```
> **ArucoDetect.h**：
>  ```c++
> //
> // Created by sxf on 22-4-23.
> //
> 
> #ifndef DEMO_ARUCODETECT_H
> #define DEMO_ARUCODETECT_H
> 
> namespace sxf {
> 
>     class ArucoDetect {
> 
>     };
> 
> } // sxf
> 
> #endif //DEMO_ARUCODETECT_H
> ```


## 使用
```bash
cmake .
make -j 4
./demo
```
内容修改：`main.cpp`  (当时随手写的，比较乱)    
