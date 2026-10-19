"""Manual macOS selection probe. Use only in a disposable test document."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Send eight Shift+Left keystrokes after a five-second delay.')
    args = parser.parse_args()
    if not args.run:
        parser.error('This manual keyboard diagnostic requires --run.')
    import time
    import Quartz
    print('In five seconds: sending eight Shift+Left keys to the focused app. Ctrl+C cancels.', flush=True)
    time.sleep(5)
    for _ in range(8):
        for down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(None, 123, down)
            Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskShift)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


if __name__ == '__main__':
    main()
