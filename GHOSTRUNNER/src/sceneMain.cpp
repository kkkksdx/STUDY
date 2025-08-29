#include "sceneMain.h"

void SceneMain::init()
{
    world_size_ = game_.getScreenSize() * 3.0f;
    camera_position_ = glm::vec2(-100.0f);
}

void SceneMain::handleEvents(SDL_Event &event)
{
}

void SceneMain::update(float dt)
{
    camera_position_ += glm::vec2(10.0f, 10.0f) * dt;
}

void SceneMain::render()
{
    renderBackground();
}

void SceneMain::clean()
{
}

void SceneMain::renderBackground()
{
    auto start = -camera_position_;
    auto end = world_size_ - camera_position_;
    game_.drawGrid(start, end, 80.0f, {0.5, 0.5, 0.5, 1.0});
    game_.drawBoundary(start, end, 5.0f, {1.0, 1.0, 1.0, 1.0});
}
