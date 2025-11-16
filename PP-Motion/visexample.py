from lib.model.load_critic import load_critic
from render.render import render_multi
import torch
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
render_output_path = "render_output"
motion_pth = "data/visexample.pth" # torch tensor: [bs, frame, 25, 3]), axis-angle with 24 SMPL joints and 1 XYZ root location
motion_seq = torch.load(motion_pth, map_location=device)
critic_model = load_critic("output/pp-motion_pretrained/checkpoint_latest.pth", device)
critic_scores = critic_model.module.batch_critic(motion_seq).tolist()
comments = []
output_paths = []
print(f"critic scores are {critic_scores}")
for idx, score in enumerate(critic_scores):
    score = round(score[0], 2)
    comments.append(f"PP-Motion score is: {score}")
    output_paths.append(os.path.join(render_output_path, f"visexample_{idx}.mp4"))
# rendering
print("Rendering...")
motion_seq = motion_seq.permute(0, 2, 3, 1) # [batch_size, 25, 3, num_frames=60]
render_multi(motion_seq, device, comments, output_paths, pose_format="rotvec")