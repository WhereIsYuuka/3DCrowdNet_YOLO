import torch
from module import PositionNet, RotationNet, Vposer

# Function to print details of each layer
def print_model_parameters(model):
    params_info = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_info.append(f"Layer: {name} | Size: {param.size()} | Params: {param.numel()}")
    return params_info

# Initialize PositionNet model
position_net = PositionNet()
rotation_net = RotationNet()
vposer = Vposer()
# pose2feat = Pose2Feat()

# Get parameters for PositionNet
position_net_params = print_model_parameters(position_net)
# print(f"PositionNet Parameters: {position_net_params}")
print(f"这里是PositionNet的param的输出: {sum(p.numel() for p in position_net.parameters() if p.requires_grad)}")

# Get parameters for RotationNet
rotation_net_params = print_model_parameters(rotation_net)
# print(f"RotationNet Parameters: {rotation_net_params}")
print(f"这里是RotationNet的param的输出: {sum(p.numel() for p in rotation_net.parameters() if p.requires_grad)}")

# Get parameters for Vposer
vposer_params = print_model_parameters(vposer)
# print(f"Vposer Parameters: {vposer_params}")
print(f"这里是Vposer的param的输出: {sum(p.numel() for p in vposer.parameters() if p.requires_grad)}")

# Get parameters for Pose2Feat
# pose2feat_params = print_model_parameters(pose2feat)
# print(f"Pose2Feat Parameters: {pose2feat_params}")

# You can save these details to a file if needed
with open("model_params.txt", "w") as f:
    f.write(f"PositionNet Parameters:\n{position_net_params}\n")
    f.write(f"RotationNet Parameters:\n{rotation_net_params}\n")
    f.write(f"Vposer Parameters:\n{vposer_params}\n")
    # f.write(f"Pose2Feat Parameters:\n{pose2feat_params}\n")
