import Quartz
import time

def simulate_backspace(count):
    # kVK_Delete is 51 in Carbon
    time.sleep(1) # wait for focus
    print("Injecting")
    for _ in range(count):
        # Key down
        event_down = Quartz.CGEventCreateKeyboardEvent(None, 51, True)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)
        # Key up
        event_up = Quartz.CGEventCreateKeyboardEvent(None, 51, False)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)
        # time.sleep(0.01)

simulate_backspace(5)
print("Done")
