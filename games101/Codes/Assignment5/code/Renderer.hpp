#pragma once
#include "Scene.hpp"

struct hit_payload
{
    //���ߺ������ཻ���������
    float tNear;
    //���ߺ������ཻ�������ε�����
    uint32_t index;
    //���ߺ������ཻ�����������ڲ�����������
    Vector2f uv;
    //���ߺ������ཻ������ָ��
    Object* hit_obj;
};

class Renderer
{
public:
    void Render(const Scene& scene);

private:
};