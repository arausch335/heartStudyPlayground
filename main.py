from spatial_registration.main import main

main(target_path="/Users/Alexander/PycharmProjects/heartStudyPlayground/data/target.ply",
     source_path="/Users/Alexander/PycharmProjects/heartStudyPlayground/data/source.ply",
     distance_threshold=1e-2,
     icp_iterations=60,
     correspondence_threshold=0.05,
     normal_neighbors=30,
     output_dir="data/outputs"
     )
