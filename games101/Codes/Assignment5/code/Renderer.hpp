#pragma once
#include "Scene.hpp"

struct hit_payload
{
    //光线和物体相交的最近距离
    float tNear;
    //光线和物体相交的三角形的索引
    uint32_t index;
    //光线和物体相交点在三角形内部的重心坐标
    Vector2f uv;
    //光线和物体相交的物体指针
    Object* hit_obj;
};

class Renderer
{
public:
    void Render(const Scene& scene);

private:
};