import time
import subprocess

from look_at.wmctrl import WmCtrl


class ShowImageMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.viewers = []
        self.active_window = WmCtrl().get_active_window()
        # This is because this lib is not python3 ready ...
        self.active_window.id = self.active_window.id.decode('utf-8')

    def show_image(self, image_path):
        if image_path:
            self.viewers.append(subprocess.Popen(['eog', '--single-window', image_path]))
            time.sleep(0.25)
            self.active_window.activate()
        else:
            self.close_viewers()

    def close_viewers(self):
        for p in self.viewers:
            p.terminate()
            p.kill()
        self.viewers = []
