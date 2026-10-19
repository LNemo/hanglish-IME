"""Manual macOS Backspace probe. Use only in a disposable, empty test document."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Send five Backspace keystrokes after a five-second delay.')
    args = parser.parse_args()
    if not args.run:
        parser.error('This manual keyboard diagnostic requires --run.')
    import time
    import Quartz
    print('In five seconds: sending five Backspace keys to the focused app. Ctrl+C cancels.', flush=True)
    time.sleep(5)
    for _ in range(5):
        for down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(None, 51, down)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


if __name__ == '__main__':
    main()
