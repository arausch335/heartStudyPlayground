import os


class Environment(object):
    def __init__(self):
        self.scene = None

        self.BASE_PATH = os.getcwd()
        self.REGISTRATION_PATH = os.path.join(self.BASE_PATH, "registration")

        self.DATA_PATH = os.path.join(self.BASE_PATH, "data")
        self.IO_DATA_PATH = os.path.join(self.DATA_PATH, "io")
        self.PO_DATA_PATH = os.path.join(self.DATA_PATH, "po")

        self.DICOM_DATA_PATH = os.path.join(self.PO_DATA_PATH, "CMR", "DICOM")

    def set_scene(self, scene):
        self.scene = scene
