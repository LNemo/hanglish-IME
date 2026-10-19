"""Manual macOS keyboard interception probe; never imported by the test suite."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Start the global keyboard listener; Esc stops it.')
    args = parser.parse_args()
    if not args.run:
        parser.error('This manual keyboard diagnostic requires --run.')
    from pynput import keyboard
    import Quartz

    def intercept(event_type, event):
        if event_type == Quartz.kCGEventKeyDown:
            if Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode) == 48:
                print('Intercepted Tab!', flush=True)
                return None
        return event

    def on_press(key):
        print('Pressed', key, flush=True)
        return False if key == keyboard.Key.esc else None

    print('Listening to global keyboard events. Press Tab to test; Esc to stop.', flush=True)
    with keyboard.Listener(on_press=on_press, darwin_intercept=intercept) as listener:
        listener.join()


if __name__ == '__main__':
    main()
