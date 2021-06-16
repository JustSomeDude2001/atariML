# This is a breakout game emulator from the book.
# Can play using A, D and SPACE keys.

# NOTE: Run as root.
#       Enable access for assistive devices
import gym

import queue, threading, time
from pynput.keyboard import Key, Listener

# keyboard controller

def keyboard(queue):
    def on_press(key):
        if key == Key.esc:
            queue.put(-1)
        elif key == Key.space:
            queue.put(ord(' '))
        else:
            key = str(key).replace("'", '')
            if key in ['w', 'a', 's', 'd']:
                queue.put(ord(key))
    
    def on_release(key):
        if key == Key.esc:
            return False
    
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

# game starting function

def start_game(queue):

    atari = gym.make('Breakout-v0')
    key_to_act = atari.env.get_keys_to_action()
    key_to_act = {k[0]: a for k, a in key_to_act.items() if len(k) > 0}
    atari.reset()

    # Input and game 

    while True:
        atari.render()
        action = 0 if queue.empty() else queue.get(block=False)
        if action == -1:
            break
        action = key_to_act.get(action, 0)
        observation, reward, done, _ = atari.step(action)
        if action != 0:
            print("Action {}, reward {}".format(action, reward))
        if done:
            print("Game finished")
            break
        time.sleep(0.05)

# Start the game.

if __name__ == "__main__":
    queue = queue.Queue(maxsize = 10)
    game = threading.Thread(target=start_game, args = (queue,))
    game.start()    
    keyboard(queue)

# Keyboard method will be replaced with agent in future.