from mujocoGen import MujocoMotionGenerator

if __name__ == "__main__":
   gen = MujocoMotionGenerator(
    xml_path="humanoid.xml",
    clip_seconds=10,
    camera_name=None,   # or named camera in XML
)

   gen.render_to_mp4("motion_preview.mp4")