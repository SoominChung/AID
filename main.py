import d3rlpy
import numpy as np
import os 
import torch
 
def main():
    torch.set_num_threads(16) 
    ############ 1. Hyperparameter 
    n_epochs=2000
    batch_size = 4096 
    use_gpu=1
    discount_factor = 0.99

    ############ 2. Load data
    path = '/RL/AID/'
    data_path = os.path.join(path,f'data')
    train_data = np.load(os.path.join(data_path, 'train.npz'))
    valid_data = np.load(os.path.join(data_path, 'valid.npz'))
    
    ############ 3. Get terminals, environment, actions, and rewards from data
    train_terminals_np, train_enviornment_np, train_actions_np, train_rewards_np = train_data['terminals'], train_data['env'], train_data['action'], train_data['reward']
    valid_terminals_np, valid_enviornment_np, valid_actions_np, valid_rewards_np = valid_data['terminals'], valid_data['env'], valid_data['action'], valid_data['reward']

    ############ 4. Build MDP Dataset
    train_dataset = d3rlpy.dataset.MDPDataset(
        observations=train_enviornment_np,
        actions=train_actions_np,
        rewards=train_rewards_np,
        terminals=train_terminals_np
    )
    valid_dataset = d3rlpy.dataset.MDPDataset(
        observations=valid_enviornment_np,
        actions=valid_actions_np,
        rewards=valid_rewards_np,
        terminals=valid_terminals_np
    )
    
    ############ 5. Call episodes
    train_episodes = train_dataset.episodes
    valid_episodes = valid_dataset.episodes

    ############ 6. Train model 
    CQL = d3rlpy.algos.DiscreteCQL(gamma=discount_factor,scaler='standard',reward_scaler='standard',use_gpu=use_gpu,batch_size=batch_size)
    CQL.fit(train_episodes,save_interval=100,n_epochs=n_epochs,eval_episodes=valid_episodes)        
 

if __name__ == "__main__":
    main()