import threading
import keyboard
import time

def last_pressed_key(flags):
    while True:
        event = keyboard.read_event()
        if event.name == "shift":
            flags["shift"] = event.event_type == keyboard.KEY_DOWN
            continue
        if event.event_type == keyboard.KEY_DOWN:
            flags["key_change"] = True
            flags["last_key"] = event.name
            flags["is_pressed"] = True
        if event.event_type == keyboard.KEY_UP:
            flags["is_pressed"] = False

flags = {"shift": False, "key_change": False, "last_key": "", "is_pressed": False}
t = threading.Thread(target=last_pressed_key, args=(flags,))
t.start()
while True:
    time.sleep(1)
    if flags["key_change"]:
        print(flags)
        flags["key_change"] = False

