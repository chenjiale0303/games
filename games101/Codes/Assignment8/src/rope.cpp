#include <iostream>
#include <vector>

#include "CGL/vector2D.h"

#include "mass.h"
#include "rope.h"
#include "spring.h"

namespace CGL {

    Rope::Rope(Vector2D start, Vector2D end, int num_nodes, float node_mass, float k, vector<int> pinned_nodes)
    {
        // TODO (Part 1): Create a rope starting at `start`, ending at `end`, and containing `num_nodes` nodes.

        // Comment-in this part when you implement the constructor

        // 创建num_nodes个mass
        for (int i = 0; i < num_nodes; i++) 
        {
            Vector2D point = start + i * (end - start) / (num_nodes - 1);
            masses.push_back(new Mass(point, node_mass, false));
        }
        // 将固定的mass标记为true
        for (auto &i : pinned_nodes) {
            masses[i]->pinned = true;
        }
        
        //创建num_springs个spring
        int num_springs = num_nodes - 1;
        for (int i = 0; i < num_springs; i++)
        {
            springs.push_back(new Spring(masses[i], masses[i + 1], k));
        }
    }

    void Rope::simulateEuler(float delta_t, Vector2D gravity)
    {
        for (auto &s : springs)
        {
            // TODO (Part 2): Use Hooke's law to calculate the force on a node
            // 弹簧两端的质点a和b
            auto a = s->m1, b = s->m2;
            auto f_a_to_b = s->k * (b->position - a->position).unit() * ((b->position - a->position).norm() - s->rest_length);
            a->forces += f_a_to_b;
            b->forces -= f_a_to_b;
        }

        // 定义阻尼系数kd
        float kd = 0.01f;
        for (auto &m : masses)
        {
            if (!m->pinned)
            {
                // TODO (Part 2): Add the force due to gravity, then compute the new velocity and position
                m->forces += gravity * m->mass;
                // TODO (Part 2): Add global damping
                m->forces -= kd * m->velocity;
                // 根据F=ma计算加速度a
                auto a = m->forces / m->mass;

                // // 显示欧拉方法：
                // // 根据x(t+1)=x(t)+dt*v(t)计算出下一时刻的位置
                // m->position += delta_t * m->velocity;
                // // 根据v(t+1)=v(t)+dt*a(t)计算出下一时刻的速度
                // m->velocity += delta_t * a;

                // 半隐式欧拉方法：
                // 根据v(t+1)=v(t)+dt*a(t)计算出下一时刻的速度
                m->velocity += delta_t * a;
                // 根据x(t+1)=x(t)+dt*v(t+1)计算出下一时刻的位置
                m->position += delta_t * m->velocity;
            }

            // Reset all forces on each mass
            m->forces = Vector2D(0, 0);
        }
    }

    void Rope::simulateVerlet(float delta_t, Vector2D gravity)
    {
        for (auto &s : springs)
        {
            // TODO (Part 3): Simulate one timestep of the rope using explicit Verlet （solving constraints)
            auto a = s->m1, b = s->m2;
            auto f_a_to_b = s->k * (b->position - a->position).unit() * ((b->position - a->position).norm() - s->rest_length);
            a->forces += f_a_to_b;
            b->forces -= f_a_to_b;
        }

        for (auto &m : masses)
        {
            if (!m->pinned)
            {
                Vector2D temp_position = m->position;
                m->forces += gravity * m->mass;
                auto a = m->forces / m->mass;
                // // 显式verlet：
                // // TODO (Part 3.1): Set the new position of the rope mass
                // // 根据x(t+1)=x(t)+[x(t)-x(t-1)]+a(t)*dt*dt求出下一时刻的位置
                // m->position = m->position + (m->position - m->last_position) + a * delta_t * delta_t;

                // 加入阻尼的显式verlet：
                // TODO (Part 4): Add global Verlet damping
                // 加入damping_factor
                float damping_factor = 0.00005f;
                m->position = m->position + (1 - damping_factor) * (m->position - m->last_position) + a * delta_t * delta_t;

                // 用last_position记录x(t-1)
                m->last_position = temp_position;
            }
            m->forces = Vector2D(0, 0);
        }
    }
}
