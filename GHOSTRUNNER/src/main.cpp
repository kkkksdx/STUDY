#include "core/game.h"
int main(int, char **)
{
    // 初始化游戏
    Game &game = Game::GetInstance();
    game.init("Ghost Runner", 1280, 720);
    // 运行游戏主循环
    game.run();
    return 0;
}