import tkinter as tk

# Simple tkinter GUI: shows current arrow while pressed, 'No Input' when released.

KEY_MAP = {
    'Up': 'Forward',
    'Down': 'Backwards',
    'Left': 'Left',
    'Right': 'Right',
}


def main():
    root = tk.Tk()
    root.title('Arrow Input')

    label = tk.Label(root, text='No Input', font=('Helvetica', 48), width=18)
    label.pack(padx=20, pady=20)

    # Keep track of which arrow keys are currently held down
    pressed = set()

    def show_for_key(key_sym):
        label.config(text=KEY_MAP.get(key_sym, 'No Input'))

    def on_press(event):
        key = event.keysym
        if key in KEY_MAP:
            pressed.add(key)
            show_for_key(key)
        elif key.lower() == 'q':
            root.destroy()

    def on_release(event):
        key = event.keysym
        if key in pressed:
            pressed.discard(key)

        # If another arrow is still held, show that one; otherwise 'No Input'
        for k in ('Up', 'Down', 'Left', 'Right'):
            if k in pressed:
                show_for_key(k)
                break
        else:
            label.config(text='No Input')

    # Bind to all key events so the window responds even when focus is elsewhere
    root.bind_all('<KeyPress>', on_press)
    root.bind_all('<KeyRelease>', on_release)

    # Ensure the window has focus to receive key events in many environments
    root.after(100, lambda: root.focus_force())

    root.mainloop()
if __name__ == '__main__':
    main()