//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_INTERSECTION_H
#define RAYTRACING_INTERSECTION_H
#include "Vector.hpp"
#include "Material.hpp"
class Object;
class Sphere;

struct Intersection
{
    Intersection(){
        happened=false;
        coords=Vector3f();
        normal=Vector3f();
        distance= std::numeric_limits<double>::max();
        obj =nullptr;
        m=nullptr;
    }
    //是否相交
    bool happened;
    //交点坐标
    Vector3f coords;
    //交点法线
    Vector3f normal;
    //光线起点到交点的距离
    double distance;
    //与光线相交的obj
    Object* obj;
    //交点处的材质
    Material* m;
};
#endif //RAYTRACING_INTERSECTION_H
