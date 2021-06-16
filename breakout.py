# This is a breakout game emulator from the book.
# Can play using A, D and SPACE keys.

# NOTE: Run as root.
#       Enable access for assistive devices
import gym

import queue, threading, time
from pynput.keyboard import Key, Listener

import numpy

import cv2

# Image cropping

def cv2_resize_image(image, resized_shape=(84, 84),
                     method='crop', crop_offset=8):
    height, width = image.shape
    resized_height, resized_width = resized_shape

    if method == 'crop':
        h = int(round(float(height) * resized_width / width))
        resized = cv2.resize(image,
                             (resized_width, h),
                             interpolation=cv2.INTER_LINEAR)
        crop_y_cutoff = h - crop_offset - resized_height
        cropped = resized[crop_y_cutoff:crop_y_cutoff+resized_height, :]
        return numpy.asarray(cropped, dtype=numpy.uint8)
    elif method == 'scale':
        return numpy.asarray(cv2.resize(image,
                                        (resized_width, resized_height),
                                        interpolation=cv2.INTER_LINEAR),
                                        dtype=numpy.uint8)
    else:
        raise ValueError('Unrecognized image resize method.')

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
    observation = atari.reset()

    # This section will important for preparing data for our model.

    import numpy
    from PIL import Image
    img = numpy.dot(observation, [0.2126, 0.7152, 0.0722])
    img = cv2_resize_image(img)
    img = Image.fromarray(img)
    img.save('save/{}.jpg'.format(0))

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