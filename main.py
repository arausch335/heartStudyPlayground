from registration.scene import Scene
from environment import Environment

env = Environment()
scene = Scene(env)
env.set_scene(scene)

scene.load_io_data()
scene.segment_io_frames()
scene.define_or_from_target()
scene.register_io_frames()
scene.io_frames[0].visualize(retractor=False)
scene.io_frames[1].visualize(retractor=True)

# scene.load_po_data(env.DICOM_DATA_PATH, load_view="LAX", replace=True)
# scene.po_series[0].summary()
# scene.po_series[0].visualize()
# scene.po_frames[11].visualize()
