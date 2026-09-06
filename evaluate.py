import torch
from TileDance import MahjongEnv, MahjongNet

def evaluate():
    num_games = 1000
    wins = 0
    total_steps = 0
    
    env = MahjongEnv()
    
    network = MahjongNet()
    
    network.load_state_dict(torch.load("best_model.pt", weights_only=True))
    
    network.eval
    
    for game in range(num_games):
        env.reset()
        env.draw_tile()
        env.sort_hand()
        
        done = 0
        
        episode_steps = 0
        
        while done==0:
            state = env.get_state()
            
            state_tensor = torch.tensor(state, dtype=torch.float32)
            
            with torch.no_grad():
                q_values = network(state_tensor)
                
                action=torch.argmax(q_values).item()
                
            next_hand, reward, done = env.step(action)
            
            episode_steps += 1
            
            if done == 1 and reward == 10:
                wins+= 1
                
        total_steps += episode_steps
        
    win_rate = wins / num_games

    average_length = (
        total_steps / num_games
    )


    print("Evaluation games:", num_games)

    print(
        "Wins:",
        wins
    )

    print(
        "Win rate:",
        round(win_rate, 4)
    )

    print(
        "Average episode length:",
        round(average_length, 2)
    )


if __name__ == "__main__":

    evaluate()