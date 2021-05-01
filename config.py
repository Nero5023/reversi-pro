from util import dotdict
#########################################################
# self play
#########################################################
self_play_epoch_max = 2
# simulations num per search
simu_num = 400
self_play_batch_size = 32
self_play_parallel_num = 8

# play the game of the num of simulations
play_simu_num = 4000


#########################################################
# Train config
#########################################################
train_model_type = 1


#########################################################
# MCTS config
#########################################################
# less than which epoch to add noise
self_play_batch_noise_move_epoch = 20


#########################################################
# board config
#########################################################
board_len = 8
# the extra 1 is pass mvoe
total_possible_move = board_len * board_len + 1


game_config = dotdict({
    "board_size": (board_len, board_len),
    "action_size":  total_possible_move,
    "feature_channels": 7,
    "feature_channels_v2": 15
})



#
train_status_path = 'train_status.json'
train_status_default = {
    'version': None
}
