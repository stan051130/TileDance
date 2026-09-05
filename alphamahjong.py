import random
import torch
import torch.nn as nn
from collections import deque
import matplotlib.pyplot as plt
import csv
import json
import os


class Tile:
    def __init__(self, suit, rank):
        self.suit = suit #"bamboos" "characters" "Dot"
        self.rank = rank

    def __repr__(self):
        return f"{self.suit}:{self.rank}"
    
    
    
class MahjongEnv:
    def __init__(self):
        self.tile_types = []
            
        #numbered suits
        for suit in ["bamboo", "character", "dot"]:
            for rank in range(1,10):
                self.tile_types.append(Tile(suit, rank))
                    
        #winds
        for winds in ["east", "south","west", "north"]:
            self.tile_types.append(Tile("wind", winds))
                
        #dragon
        for dragon in ["red", "green", "white"]:
            self.tile_types.append(Tile("dragon", dragon))
    
        self.deck=[]
        self.hand=[]
        self.discards=[]


    #Reset game environment
    def reset(self):
        self.deck=[]
        self.discards=[]
        
        for tile in self.tile_types:
            for _ in range(4):
                self.deck.append(Tile(tile.suit, tile.rank))
        
        random.shuffle(self.deck)
        
        self.hand = []
        
        for _ in range(13):
            self.hand.append(self.deck.pop())
            
        self.sort_hand()
        
        return self.hand
    
    
    #get one tile from the deck
    def draw_tile(self):
        
        if len(self.deck) == 0:
            return 0
        
        self.hand.append(self.deck.pop())
        return 1


    #discard one tile from hand 
    def discard_tile(self, index):
        tile = self.hand.pop(index)
        self.discards.append(tile)
    
    
    #sort hand
    def sort_hand(self):
        
        suit_order = {
            'bamboo' : 0,
            'character': 1,
            'dot' : 2,
            'wind' : 3,
            'dragon' : 4
        }
        
        #give numerical order to tiles
        def get_tile_order(tile):
            
            if tile.suit =='bamboo':            
                return suit_order['bamboo'] * 10 + tile.rank
            elif tile.suit =='character':            
                return suit_order['character'] * 10 + tile.rank
            elif tile.suit =='dot':            
                return suit_order['dot'] * 10 + tile.rank
            elif tile.suit =='wind':
                if tile.rank == 'east':
                    return 30
                elif tile.rank == 'south':
                    return 31
                elif tile.rank == 'west':
                    return 32
                elif tile.rank == 'north':
                    return 33
            elif tile.suit =='dragon':
                if tile.rank == 'red':
                    return 40
                if tile.rank == 'green':
                    return 41
                if tile.rank == 'white':
                    return 42
        self.hand.sort(key=get_tile_order)
            

    def hand_quality(self):
        counts = self.get_state()
        
        score = 0.0
        
        #bamboo, character, dot
        for suit_start in [0, 9, 18]:
            
            #pair, triplet score
            for i in range(suit_start, suit_start + 9):
                
                if counts[i] >= 2:
                    score += 1
                    
                if counts[i] >= 3:
                    score += 2
                    
            #adjacent tiles, for example, bamboo 2 and bamboo 3
            for i in range(suit_start, suit_start + 8):
                if counts[i] > 0 and counts[i+1] > 0:
                    score += 0.5
                    
            #gap tiles, for example, bamboo 1 and bamboo 3
            for i in range(suit_start, suit_start + 7):
                if counts[i] > 0 and counts[i+2] >0:
                    score += 0.25
                    
        #honor tiles, winds and dragons
        for i in range(27, 34):
            if counts[i] >=2:
                score += 1.0
            if counts[i] >=3:
                score += 2.0
            if counts[i] == 1:
                score -= 1
        
        return score
    
    
    #check if won
    def check_hu(self):
        
        if len(self.hand) != 14:
            return 0
        
        original_hand = self.hand.copy()
        
        def check_sequence(hand):
            
            if len(hand) < 3:
                return None
            
            tile = hand[0]
            if tile.suit == "wind" or tile.suit == "dragon":
                return None
            
            index_1 = -1
            index_2 = -1
            
            for i in range(1, len(hand)):
                
                if hand[i].suit == tile.suit:
                    if hand[i].rank == tile.rank + 1 and index_1 == -1:
                        index_1 = i
                    elif hand[i].rank == tile.rank + 2 and index_2 == -1:
                        index_2 = i
                        
            if index_1 != -1 and index_2 != -1:
                return [0, index_1, index_2]
            
            return None
                    
        
        def check_triplet(hand):
            
            if len(hand) < 3:
                return None
            
            tile = hand[0]
            if tile.suit == hand[1].suit and hand[2].suit == tile.suit:
                if hand[1].rank == tile.rank and hand[2].rank == tile.rank:
                    return 1

            return None
            
        def check_pair(hand, i):
            
            if i + 1 >= len(hand):
                return 0
            
            if hand[i+1].suit == hand[i].suit:
                if hand[i+1].rank == hand[i].rank:
                    return 1
            return 0
        
        def check_recursion(hand):
            
            if len(hand) == 0:
                return 1
            
            if check_triplet(hand):
                new_hand = hand.copy()
                new_hand.pop(0)
                new_hand.pop(0)
                new_hand.pop(0)
                
                if check_recursion(new_hand):
                    return 1
                
            sequence = check_sequence(hand)
            
            if sequence != None:
                new_hand = hand.copy()
                
                index1 = sequence[1]
                index2 = sequence[2]
                
                new_hand.pop(index2)
                new_hand.pop(index1)
                new_hand.pop(0)
                            
                if check_recursion(new_hand):
                    return 1
                
            return 0
                
        for i in range(len(original_hand)):
            if check_pair(original_hand, i):
                hand = original_hand.copy()
                hand.pop(i)
                hand.pop(i)
        
                if check_recursion(hand):
                    return 1
            
            
        return 0
                
                
    # RL
    def step(self, action):
        
        current_quality = self.hand_quality()
        
        self.discard_tile(action)

        if len(self.deck) == 0:
            return self.hand, -2, 1
        
        
        self.draw_tile()
        self.sort_hand()
        
        if self.check_hu():
            return self.hand, 10, 1
        
        next_quality = self.hand_quality()
        
        beta = 0.02
        
        reward = beta * (next_quality - current_quality)
        
        return self.hand, reward, 0
    
    
    def tile_to_index(self, tile):
        
        for i in range(len(self.tile_types)):
            
            current_tile = self.tile_types[i]
            
            if current_tile.suit == tile.suit and current_tile.rank == tile.rank:
                return i
            
    def get_state(self):
        state = [0] * 34
        
        for tile in self.hand:
            index = self.tile_to_index(tile)
            
            state[index] += 1
            
        return state
        
        
class MahjongNet(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.fc1 = nn.Linear(34, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 14)
    
    def forward(self, x):
        
        x = self.fc1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        
        x = self.fc3(x)
        
        return x
        
def main():
    # =========================
    # Logging
    # =========================

    os.makedirs("experiments", exist_ok=True)

    with open(
        f"experiments/{experiment_name}_config.json",
        "w"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )
    
    experiment_name = "dqn_vec_target100_shape002"

    config = {
        "algorithm": "DQN",
        "episodes": 10000,
        "replay_buffer_size": 50000,
        "batch_size": 32,
        "vectorized_replay": True,
        "target_update": 100,
        "gamma": 0.99,
        "epsilon": 0.1,
        "learning_rate": 0.001,
        "reward_shaping_beta": 0.02,
        "win_reward": 10,
        "lose_reward": -2,
        "seed": 42
    }

    seed = config["seed"]
    
    episode_history = []
    win_rate_history = []
    loss_history = []
    reward_history = []
    q_history = []
    length_history = []

    log_every = 100

    window_wins = 0
    window_loss = 0.0
    window_reward = 0.0
    window_q = 0.0
    window_length = 0
    
    target_update = config["target_update"]
    gamma = config["gamma"]
    epsilon = config["epsilon"]
    batch_size = config["batch_size"]
    
    env = MahjongEnv()
    target_net = MahjongNet()
    network = MahjongNet()
    
    target_net.load_state_dict(network.state_dict())

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=config["learning_rate"]
    )
    replay_buffer = deque(maxlen=config["replay_buffer_size"])
    loss_function = nn.MSELoss()
    
    best_win_rate = 0.0
    
    for episode in range(1,config["episodes"] + 1):
        
        env.reset()
        env.draw_tile()
        env.sort_hand()

        episode_reward = 0.0
        episode_loss = 0.0
        episode_updates = 0

        episode_q_sum = 0.0
        episode_steps = 0

        episode_won = False
        
        done = 0
        step_count = 0

        while done == 0:

            state = env.get_state()
            
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
            
            )

            with torch.no_grad():

                q_values = network(state_tensor)

                episode_q_sum += torch.max(q_values).item()

                if random.random() < epsilon:

                    action = random.randint(
                        0,
                        len(env.hand) - 1
                    )

                else:

                    action = torch.argmax(q_values).item()

            episode_steps += 1

            next_hand, reward, done = env.step(action)
            episode_reward += reward
            
            if done==1 and reward==10:
                episode_won=True
            
            next_state = env.get_state()
            
            replay_buffer.append((state, action, reward, next_state, done))
            
            
            if len(replay_buffer)>= batch_size:
                
                batch = random.sample(replay_buffer, batch_size)
                
                batch_states = []
                batch_actions = []
                batch_rewards = []
                batch_next_states = []
                batch_dones= []
                
                for experience in batch:
                    
                    batch_states.append(experience[0])
                    batch_actions.append(experience[1])
                    batch_rewards.append(experience[2])
                    batch_next_states.append(experience[3])
                    batch_dones.append(experience[4])
                
                #[32,34]
                batch_states_tensor = torch.tensor(
                                            batch_states, 
                                            dtype=torch.float32
                                    )
                    
                #[32]
                batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long) 
                
                #[32]
                batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
                
                #[32,34]
                batch_next_state_tensor = torch.tensor(batch_next_states, dtype=torch.float32)
                
                #[32]
                batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float32)         

                
                #Current Q
                #[32,34] ->network -> [32,14]
                
                batch_q_values = network(batch_states_tensor)
                
                row_indices = torch.arange(batch_size)
                
                batch_current_q = batch_q_values[row_indices, batch_actions_tensor]
                
                #Target Q
                with torch.no_grad():
                    batch_next_q_values = target_net(batch_next_state_tensor)
                    
                    batch_max_next_q = torch.max(batch_next_q_values, dim=1).values
                    
                    batch_target_q = (batch_rewards_tensor + gamma * batch_max_next_q * (1-batch_dones_tensor))
                    
                #Training
                loss = loss_function(batch_current_q, batch_target_q)
                
                optimizer.zero_grad()
                
                loss.backward()
                
                optimizer.step()
                
                episode_loss += loss.item()
                
                episode_updates += 1
                
        # =========================
        # End of episode logging
        # =========================

        if episode_won:
            window_wins += 1


        # average loss in this episode
        if episode_updates > 0:
            avg_episode_loss = episode_loss / episode_updates
        else:
            avg_episode_loss = 0.0


        # average max Q in this episode
        if episode_steps > 0:
            avg_episode_q = episode_q_sum / episode_steps
        else:
            avg_episode_q = 0.0


        window_loss += avg_episode_loss
        window_reward += episode_reward
        window_q += avg_episode_q
        window_length += episode_steps
        
        if episode % target_update == 0:
            target_net.load_state_dict(
                network.state_dict()
            )
        
        if episode % log_every == 0:
            win_rate = window_wins / log_every

            avg_loss = window_loss / log_every
            avg_reward = window_reward / log_every
            avg_q = window_q / log_every
            avg_length = window_length / log_every


            episode_history.append(episode)

            win_rate_history.append(win_rate)
            loss_history.append(avg_loss)
            reward_history.append(avg_reward)
            q_history.append(avg_q)
            length_history.append(avg_length)


            print(
                "Episode:", episode,
                "| Win rate:", round(win_rate, 3),
                "| Loss:", round(avg_loss, 5),
                "| Reward:", round(avg_reward, 3),
                "| Q:", round(avg_q, 3),
                "| Length:", round(avg_length, 1)
            )
            
            if win_rate > best_win_rate:

                best_win_rate = win_rate

                torch.save(
                    network.state_dict(),
                    f"experiments/{experiment_name}_best_model.pt"
                )

            # reset window
            window_wins = 0
            window_loss = 0.0
            window_reward = 0.0
            window_q = 0.0
            window_length = 0
            
    # =========================
    # Plot results
    # =========================

    plt.figure()
    plt.plot(episode_history, win_rate_history)
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Training Win Rate")
    plt.grid()
    plt.savefig("training_win_rate.png")
    plt.show()


    plt.figure()
    plt.plot(episode_history, loss_history)
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.savefig("training_loss.png")
    plt.show()


    plt.figure()
    plt.plot(episode_history, reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Average Episode Reward")
    plt.title("Episode Reward")
    plt.grid()
    plt.savefig("training_reward.png")
    plt.show()


    plt.figure()
    plt.plot(episode_history, q_history)
    plt.xlabel("Episode")
    plt.ylabel("Average Max Q")
    plt.title("Q Value")
    plt.grid()
    plt.savefig("training_Qvalue.png")
    plt.show()


    plt.figure()
    plt.plot(episode_history, length_history)
    plt.xlabel("Episode")
    plt.ylabel("Average Episode Length")
    plt.title("Episode Length")
    plt.grid()
    plt.savefig("training_episode_length.png")
    plt.show()
    
    with open(
        f"experiments/{experiment_name}_results.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "episode",
            "win_rate",
            "avg_loss",
            "avg_reward",
            "avg_q",
            "avg_episode_length"
        ])

        for i in range(len(episode_history)):

            writer.writerow([
                episode_history[i],
                win_rate_history[i],
                loss_history[i],
                reward_history[i],
                q_history[i],
                length_history[i]
            ])
    
if __name__ == "__main__":
    main()
  