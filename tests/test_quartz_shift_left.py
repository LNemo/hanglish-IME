import Quartz
import time

def simulate_shift_left(count):
    time.sleep(2) # wait for focus
    print("Injecting Shift+Left")
    for _ in range(count):
        # 123 is Left Arrow
        event_down = Quartz.CGEventCreateKeyboardEvent(None, 123, True)
        Quartz.CGEventSetFlags(event_down, Quartz.kCGEventFlagMaskShift)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_down)
        
        event_up = Quartz.CGEventCreateKeyboardEvent(None, 123, False)
        # Quartz.CGEventSetFlags(event_up, Quartz.kCGEventFlagMaskShift) # Up event usually doesn't need flags
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event_up)
        # time.sleep(0.005)

simulate_shift_left(8)
print("Done")
