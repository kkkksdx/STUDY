#ifndef SCENE_H
#define SCENE_H

#include "object.h"
#include <glm/glm.hpp>
#include <vector>

class Scene : public Object
{
protected:
    glm::vec2 camera_position_ = glm::vec2(0); // 相机位置
    std::vector<Object *> objects_;            // 场景中的物体

public:
    Scene() = default;
    virtual ~Scene() = default;

    virtual void init() override {}
    virtual void handleEvents(SDL_Event &event) override {}
    virtual void update(float dt) override {}
    virtual void render() override {}
    virtual void clean() override {}
};

#endif // SCENE_H