//
// Created by Göksu Güvendiren on 2019-05-14.
//

#include "Scene.hpp"


void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const
{
    return this->bvh->Intersect(ray);
}

void Scene::sampleLight(Intersection &pos, float &pdf) const
{
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum){
                objects[k]->Sample(pos, pdf);
                break;
            }
        }
    }
}

bool Scene::trace(
        const Ray &ray,
        const std::vector<Object*> &objects,
        float &tNear, uint32_t &index, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vector2f uvK;
        if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear) {
            *hitObject = objects[k];
            tNear = tNearK;
            index = indexK;
        }
    }


    return (*hitObject != nullptr);
}

// Implementation of Path Tracing
Vector3f Scene::castRay(const Ray &ray, int depth) const
{
    // TO DO Implement Path Tracing Algorithm here
    Vector3f hitColor = this->backgroundColor;
    Intersection inter_p = Scene::intersect(ray);
    
    //若光线与场景中的物体有交点
    if (inter_p.happened)
    {
        Vector3f L_dir(0.0f), L_indir(0.0f);

        // 取出交点信息
        Vector3f p = inter_p.coords;
        Vector3f N = inter_p.normal;
        // 与课程中的入射方向相反
        Vector3f wo = ray.direction;

        
        // 从光源出发出一条光线
        Intersection inter_light;
        float pdf_light;
        sampleLight(inter_light, pdf_light);
        // Get x, ws, NN, emit from inter_light
        Vector3f x = inter_light.coords;
        Vector3f ws = normalize(x - p);
        Vector3f NN = inter_light.normal;
        Vector3f emit = inter_light.emit;
        //Shoot a ray from p to x
        float distance_x_to_p = (x - p).norm();
        p = (dotProduct(ray.direction, N) < 0) ?
            p + N * EPSILON :
            p - N * EPSILON;
        Ray r(p, ws);
        //计算p到x之间是否有block
        Intersection inter_block = intersect(r);
        //若光源和点之间没有物体遮挡
        if (fabs(inter_block.distance - distance_x_to_p) < 0.01)
            L_dir = emit * inter_p.m->eval(wo, ws, N) * dotProduct(ws, N) * dotProduct(-ws, NN) / (distance_x_to_p * distance_x_to_p * pdf_light);

        //Test Russian Roulette with probability RussianRoulette
        float ksi = get_random_float();
        //需要计算间接光照
        if (ksi < RussianRoulette)
        {   
            Vector3f wi = normalize(inter_p.m->sample(wo, N));
            Ray r(p, wi);
            //计算p到其他间接光源的交点
            Intersection inter_indir_light = intersect(r);
            if (inter_indir_light.happened && !inter_indir_light.m->hasEmission())
            {
                float pdf = inter_p.m->pdf(wo, wi, N);
                if (pdf > EPSILON)
                    L_indir = castRay(r, depth + 1) * inter_p.m->eval(wo, wi, N) * dotProduct(wi, N) / (pdf * RussianRoulette);
            }
        }
        hitColor = inter_p.m->getEmission() + L_dir + L_indir;
    }
    return hitColor;
}