#include <algorithm>
#include <cassert>
#include "BVH.hpp"

BVHAccel::BVHAccel(std::vector<Object*> p, int maxPrimsInNode,
                   SplitMethod splitMethod)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)), splitMethod(splitMethod),
      primitives(std::move(p))
{
    time_t start, stop;
    time(&start);
    if (primitives.empty())
        return;

    if (splitMethod == SplitMethod::NAIVE)
        root = recursiveBuild(primitives);
    else if (splitMethod == SplitMethod::SAH)
        root = recursiveBuildBySAH(primitives);

    time(&stop);
    double diff = difftime(stop, start);
    int hrs = (int)diff / 3600;
    int mins = ((int)diff / 60) - (hrs * 60);
    int secs = (int)diff - (hrs * 3600) - (mins * 60);

    if (splitMethod == SplitMethod::NAIVE)
        printf(
            "\rBVH Generation complete: \nTime Taken: %i hrs, %i mins, %i secs\n\n",
            hrs, mins, secs);
    else if (splitMethod == SplitMethod::SAH)
        printf(
            "\rBVH Generation By SAH complete: \nTime Taken: %i hrs, %i mins, %i secs\n\n",
            hrs, mins, secs);
}


BVHBuildNode* BVHAccel::recursiveBuildBySAH(std::vector<Object*> objects)
{
    BVHBuildNode* node = new BVHBuildNode();

    Bounds3 bounds;
    for (int i = 0; i < objects.size(); ++i)
        bounds = Union(bounds, objects[i]->getBounds());

    if (objects.size() == 1) {
        // Create leaf _BVHBuildNode_
        node->bounds = objects[0]->getBounds();
        node->object = objects[0];
        node->left = nullptr;
        node->right = nullptr;
        return node;
    }
    else if (objects.size() == 2) {
        node->left = recursiveBuildBySAH(std::vector{ objects[0] });
        node->right = recursiveBuildBySAH(std::vector{ objects[1] });

        node->bounds = Union(node->left->bounds, node->right->bounds);
        return node;
    }
    else {
        //����Ͱ������
        int B = 32;
        double min_cost = std::numeric_limits<double>::infinity();
        int min_cost_object_split = 0;
        int min_cost_dim = 0;
        
        //��������Χ�ɵİ�Χ�еı����
        Bounds3 centroidBounds;
        for (auto& obj : objects)
            centroidBounds = Union(centroidBounds, obj->getBounds().Centroid());
        double Sn = centroidBounds.SurfaceArea();

        //ö�ٰ���ÿ���Ữ�֣�������cost
        for (int dim = 0; dim < 3; dim++) {
            switch (dim) {
            case 0:
                std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                    return f1->getBounds().Centroid().x <
                        f2->getBounds().Centroid().x;
                    });
                break;
            case 1:
                std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                    return f1->getBounds().Centroid().y <
                        f2->getBounds().Centroid().y;
                    });
                break;
            case 2:
                std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                    return f1->getBounds().Centroid().z <
                        f2->getBounds().Centroid().z;
                    });
                break;
            }

            
            //�����Ұ�Χ��
            for (int index = 1; index < B; index++) {
                auto beginning = objects.begin();
                auto middling = objects.begin() + (index * objects.size() / B);
                auto ending = objects.end();
                auto leftshapes = std::vector<Object*>(beginning, middling);
                auto rightshapes = std::vector<Object*>(middling, ending);

                Bounds3 leftBounds, rightBounds;
                for (auto& obj : leftshapes)
                    leftBounds = Union(leftBounds, obj->getBounds().Centroid());
                for (auto& obj : rightshapes)
                    rightBounds = Union(rightBounds, obj->getBounds().Centroid());
                double Sa = leftBounds.SurfaceArea(), Sb = rightBounds.SurfaceArea();
                int countA = leftshapes.size(), countB = rightshapes.size();
                double cost = countA * Sa / Sn + countB * Sb / Sn;
                //���˴λ��ִ��۸�С������
                if (cost < min_cost) {
                    min_cost = cost;
                    min_cost_object_split = index * objects.size() / B;
                    min_cost_dim = dim;
                }
            }
        }
        //������С�Ĳ������д˴λ���
        switch (min_cost_dim) {
        case 0:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().x <
                    f2->getBounds().Centroid().x;
                });
            break;
        case 1:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().y <
                    f2->getBounds().Centroid().y;
                });
            break;
        case 2:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().z <
                    f2->getBounds().Centroid().z;
                });
            break;
        }

        auto beginning = objects.begin();
        auto middling = objects.begin() + min_cost_object_split;
        auto ending = objects.end();

        auto leftshapes = std::vector<Object*>(beginning, middling);
        auto rightshapes = std::vector<Object*>(middling, ending);

        assert(objects.size() == (leftshapes.size() + rightshapes.size()));

        node->left = recursiveBuildBySAH(leftshapes);
        node->right = recursiveBuildBySAH(rightshapes);

        node->bounds = Union(node->left->bounds, node->right->bounds);
    }

    return node;
}


BVHBuildNode* BVHAccel::recursiveBuild(std::vector<Object*> objects)
{
    BVHBuildNode* node = new BVHBuildNode();

    // Compute bounds of all primitives in BVH node
    Bounds3 bounds;
    for (int i = 0; i < objects.size(); ++i)
        bounds = Union(bounds, objects[i]->getBounds());
    if (objects.size() == 1) {
        // Create leaf _BVHBuildNode_
        node->bounds = objects[0]->getBounds();
        node->object = objects[0];
        node->left = nullptr;
        node->right = nullptr;
        return node;
    }
    else if (objects.size() == 2) {
        node->left = recursiveBuild(std::vector{ objects[0] });
        node->right = recursiveBuild(std::vector{ objects[1] });

        node->bounds = Union(node->left->bounds, node->right->bounds);
        return node;
    }
    else {
        Bounds3 centroidBounds;
        //�����а�Χ�е����Ĺ��ɵİ�Χ�м�������������жϵ�ǰӦ�ð����ĸ������ƽ��
        for (int i = 0; i < objects.size(); ++i)
            centroidBounds =
            Union(centroidBounds, objects[i]->getBounds().Centroid());
        //�õ���ǰ����ᣬ012�ֱ��Ӧxyz��
        int dim = centroidBounds.maxExtent();
        switch (dim) {
        case 0:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().x <
                    f2->getBounds().Centroid().x;
                });
            break;
        case 1:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().y <
                    f2->getBounds().Centroid().y;
                });
            break;
        case 2:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().z <
                    f2->getBounds().Centroid().z;
                });
            break;
        }

        auto beginning = objects.begin();
        auto middling = objects.begin() + (objects.size() / 2);
        auto ending = objects.end();

        auto leftshapes = std::vector<Object*>(beginning, middling);
        auto rightshapes = std::vector<Object*>(middling, ending);

        assert(objects.size() == (leftshapes.size() + rightshapes.size()));

        node->left = recursiveBuild(leftshapes);
        node->right = recursiveBuild(rightshapes);

        node->bounds = Union(node->left->bounds, node->right->bounds);
    }

    return node;  
}

Intersection BVHAccel::Intersect(const Ray& ray) const
{
    Intersection isect;
    if (!root)
        return isect;
    isect = BVHAccel::getIntersection(root, ray);
    return isect;
}

Intersection BVHAccel::getIntersection(BVHBuildNode* node, const Ray& ray) const
{
    // TODO Traverse the BVH to find intersection
    Intersection isect;
    auto dirIsNeg = std::array<int, 3>{ ray.direction.x < 0.0f, ray.direction.y < 0.0f, ray.direction.z < 0.0f };

    isect.happened = node->bounds.IntersectP(ray, ray.direction_inv, dirIsNeg);
    if (!isect.happened)
        return isect;

    //����Ҷ�ӽڵ�ֱ���ж�object��ray�Ľ���
    if (!node->left && !node->right)
        return node->object->getIntersection(ray);

    auto hit1 = getIntersection(node->left, ray);
    auto hit2 = getIntersection(node->right, ray);

    //ѡ����������һ������
    if (hit1.distance < hit2.distance) isect = hit1;
    else isect = hit2;

    return isect;
}