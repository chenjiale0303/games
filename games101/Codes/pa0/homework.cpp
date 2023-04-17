#include <cmath>
#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

int main()
{
    //齐次坐标：用n+1个数来表示n维坐标，笛卡尔坐标(x, y) => 齐次坐标(x, y, w), 其中笛卡尔坐标中的x = x / w, y = y / w
    Eigen::Vector3f p(2.0, 1.0, 1.0);
    
    //绕原点逆时针旋转45°
    Eigen::Matrix3f rotation, translation;
    double angle = 45.0 / 180.0 * acos(-1);

    rotation << cos(angle), -sin(angle), 0,
        sin(angle), cos(angle), 0,
        0, 0, 1;
    translation << 1, 0, 1,
        0, 1, 2, 
        0, 0, 1;
    //p绕原点逆时针旋转45°后
    std::cout << "p绕原点逆时针旋转45°后:" << std::endl;
    std::cout << rotation * p << std::endl;
    //p绕原点逆时针旋转45°后 再平移(1, 2)
    std::cout << "p绕原点逆时针旋转45°后 再平移(1, 2):" << std::endl;
    std::cout << translation * rotation * p << std::endl;

    //affine = translation * rotation
    Eigen::Matrix3f affine;
    affine << cos(angle), -sin(angle), 1,
        sin(angle), cos(angle), 2,
        0, 0, 1;
    //p经过仿射变换直接实现旋转45°且平移(1, 2)
    std::cout << "p经过仿射变换直接实现旋转45°且平移(1, 2):" << std::endl;
    std::cout << affine * p << std::endl;

    return 0;
}