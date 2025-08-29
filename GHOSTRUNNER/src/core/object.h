#ifndef OBJECT_H
#define OBJECT_H

#include "game.h"

class Object
{
protected:
    Game &game_ = Game::GetInstance();

public:
    Object() = default;
    virtual ~Object() = default; // 所有的类，不在构造函数和析构函数里面做任何事。

    virtual void init() {} // 需要初始化的事物，在init()函数里面做。
    virtual void handleEvents(SDL_Event &event) {}
    virtual void update(float dt) {}
    virtual void render() {}
    virtual void clean() {} // 需要清理的资源，在clean()函数里面做。
};

#endif // OBJECT_H