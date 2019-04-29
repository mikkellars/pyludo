import numpy as np

player_colors = ['green', 'blue', 'red', 'yellow']


def star_jump(pos):
    if pos == -1 or pos > 51:
        return 0
    if pos % 13 == 6:
        return 6
    if pos % 13 == 12:
        return 7
    return 0


def is_globe_pos(pos):
    if pos == -1 or pos > 51:
        return False
    if pos % 13 == 1:
        return True
    if pos % 13 == 9:
        return True
    return False


def valid_dice_roll(n):
    return 1 <= n <= 6


def will_send_self_home(state, next_state):
    return np.sum(state[0] == -1) < np.sum(next_state[0] == -1)


def will_send_opponent_home(state, next_state):
    return np.sum(state[1:] == -1) < np.sum(next_state[1:] == -1)


def token_vulnerability(state, token_id):
    """ returns an approximation of the amount, n, of opponent dice rolls that can send the token home """
    player = state[0]
    token = player[token_id]

    if token == -1 or token == 1 or token > 51:  # in home, start or end positions
        return 0
    if token % 13 == 1 and np.sum(state[token // 13] == -1) == 0:  # on globe outside empty home
        return 0
    if token % 13 != 1 and np.sum(player == token) > 1 or token % 13 == 9:  # blockade or globe
        return 0

    n = 0

    if token % 13 == 1:  # on opponent start pos
        n += 1

    star = star_jump(token)
    if star > 0:
        star = 6 if star == 7 else 7

    for opponent_id in range(1, 4):
        opponent = state[opponent_id]
        for opp_token in set(opponent):
            if opp_token == -1 or opp_token > 51:
                continue
            req_dice_roll = (token - opp_token) % 52
            rel_opp_token = (opp_token - opponent_id * 13) % 52
            would_enter_end_zone = rel_opp_token + req_dice_roll > 51
            if not would_enter_end_zone and 1 <= req_dice_roll <= 6:
                n += 1
            if star > 0:
                req_dice_roll = (token - opp_token - star) % 52
                would_enter_end_zone = rel_opp_token + req_dice_roll + star > 51
                if not would_enter_end_zone and 1 <= req_dice_roll <= 6:
                    n += 1
    return n
