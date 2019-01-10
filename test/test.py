import retro
for game in retro.data.list_games():
    print(game, retro.data.list_states(game))