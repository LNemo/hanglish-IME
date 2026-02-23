from pynput import keyboard
import time
import Quartz

def darwin_intercept(event_type, event):
    if event_type == Quartz.kCGEventKeyDown:
        keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
        if keycode == 48: # Tab
            print("Intercepted Tab!")
            return None
    return event

def on_press(key):
    print("Pressed", key)
    if key == keyboard.Key.esc:
        return False

with keyboard.Listener(on_press=on_press, darwin_intercept=darwin_intercept) as listener:
    print("Press Tab. It should print 'Intercepted Tab!' and 'Pressed Key.tab', but not switch apps. Press Esc to quit.")
    listener.join()
