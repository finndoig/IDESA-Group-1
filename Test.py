import sys
import os

try:
    import tty
    import termios
    _IS_UNIX = True
except ImportError:
    import msvcrt
    _IS_UNIX = False

def get_key():
    """Get a single key press from user input (cross-platform).

    On Unix this uses `tty`/`termios`. On Windows it uses `msvcrt.getch()`.
    For consistency with the original code, Windows arrow keys are mapped to
    the same final characters used by ANSI escape sequences: 'A','B','C','D'.
    """
    if _IS_UNIX:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    else:
        ch = msvcrt.getch()
        # Arrow and function keys on Windows return a prefix (b'\x00' or b'\xe0')
        if ch in (b"\x00", b"\xe0"):
            seq = msvcrt.getch()
            mapping = {b'H': 'A', b'P': 'B', b'M': 'C', b'K': 'D'}
            return mapping.get(seq, '')
        else:
            try:
                return ch.decode('utf-8')
            except Exception:
                return ''

def main():
    """Main loop to handle arrow key input."""
    print("Press arrow keys (Up/Down/Left/Right) or 'q' to quit:")
    
    while True:
        try:
            ch = get_key()

            if ch == 'q':
                print("Exiting...")
                break

            # Unix: arrow keys come as escape sequences '\x1b', '[', 'A' etc.
            if ch == '\x1b':
                next1 = get_key()
                next2 = get_key()
                code = next2
            else:
                # On Windows we return 'A','B','C','D' directly for arrows
                code = ch

            if code == 'A':
                print("Forward")
            elif code == 'B':
                print("Backwards")
            elif code == 'C':
                print("Right")
            elif code == 'D':
                print("Left")
            else:
                print("no input")
                
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()