import sys, os
import glob

os.chdir("../")
sys.path.insert(0, "./dataset_generation//utilities")
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

from utilities.common import read_frame_history, file_under_folder, decode_pos

OUTPUT_FOLDER = "./dataset_generation/output/**/frame_history"
# OUTPUT_FOLDER = './nbv_explore_net/output'


def select_menu():
    file_list = file_under_folder(f"./{OUTPUT_FOLDER}/*.bin")
    # pattern = "dataset_generation/output/**/frame_history/*.bin"
    # file_list = glob.glob(pattern, recursive=True)
    for i, v in enumerate(file_list):
        file_path = Path(v)
        stem_name = file_path.stem
        last_folder = file_path.parents[0].name
        print(f"{i}: {last_folder} -> {stem_name}")

    if len(file_list) == 0:
        return -1
    

    user_select = input(f"Please select the *.bin you want to check:")
    return file_list[int(user_select)]


if __name__ == "__main__":
    # filename_user = select_menu()
    filename_user = r"E:\BENBV\dataset_generation\output\Stanford3D\frame_history\bunny_frame_history_Ours.bin"
    if filename_user == -1:
        print(f"No file found! Please check {OUTPUT_FOLDER}")
        exit()
    print(f"you selected {filename_user}")

    frame_history = read_frame_history(filename_user)
    print(f"YOU have {len(frame_history)} scan(s) in total.")

    print("\n=== Checking Frame Structure ===")
    for f_i in range(len(frame_history)):
        print(f"Frame {f_i}: has {len(frame_history[f_i])} elements")
        for idx in range(len(frame_history[f_i])):
            element = frame_history[f_i][idx]
            print(f"  Element {idx}: {type(element)}", end="")
            if hasattr(element, 'shape'):
                print(f" - Shape: {element.shape}")
            elif hasattr(element, '__len__'):
                print(f" - Length: {len(element)}")
            else:
                print(f" - Value: {element}")
    print("=== End Check ===\n")

    fig = go.Figure()
    for f_i in range(len(frame_history)):
        points = np.array(frame_history[f_i][0])
        # print(type(points))
        
        curr_pos = points[:, 0:3]
        # print(len(curr_pos))
        curr_view = frame_history[f_i][1]
        next_view = frame_history[f_i][2]
        
        boundary_points = np.array(frame_history[f_i][5])
        curr_boundary = boundary_points[:, 0:3]
        # print(type(boundary_points))
        # print(len(curr_boundary))
        # exit()

        # nbv_list = frame_history[f_i][3]
        # score_list = frame_history[f_i][4]

        cur_data = go.Scatter3d(
            x=curr_pos[:, 0], y=curr_pos[:, 1], z=curr_pos[:, 2], 
            mode="markers", 
            marker=dict(size=1, color="green"),
            name=f"Point Cloud {f_i+1} - th"
        )



        # boundary_data = None
        # if len(frame_history[f_i]) == 6:
        #     boundary_points = frame_history[f_i][5]
        #     if boundary_points is not None and len(boundary_points) > 0:
        #         boundary_data = go.Scatter3d(
        #             x=boundary_points[:, 0], 
        #             y=boundary_points[:, 1], 
        #             z=boundary_points[:, 2], 
        #             mode="markers", 
        #             marker=dict(size=3, color="red", symbol="diamond"),
        #             name=f"Boundary Points {f_i+1}"
        #         )       

        boundary_data = go.Scatter3d(
            x=curr_boundary[:, 0], y=curr_boundary[:, 1], z=curr_boundary[:, 2], 
            mode="markers", 
            marker=dict(size=2, color="black"),
            name=f"Boundary Points {f_i+1}" 
        )

        if f_i != len(frame_history) - 1:
            target_pos, camera_pos = decode_pos(data=curr_view)
            curr_vector = go.Scatter3d(
                x=[target_pos[0], camera_pos[0]],
                y=[target_pos[1], camera_pos[1]],
                z=[target_pos[2], camera_pos[2]],
                marker=dict(size=10, color="rgb(0,255,0)"),
                line=dict(color="rgb(0,255,0)", width=1),
                name=f"current view {f_i+1}",
            )

            target_pos, camera_pos = decode_pos(data=next_view)
            next_vector = go.Scatter3d(
                x=[target_pos[0], camera_pos[0]],
                y=[target_pos[1], camera_pos[1]],
                z=[target_pos[2], camera_pos[2]],
                marker=dict(size=5, color="rgb(255,0,0)"),
                line=dict(color="rgb(0,0,255)", width=1),
                name=f"next-best-view {f_i+1}",
            )

            # all_views = []
            # for nbv_i, nbv in enumerate(nbv_list):
            #     target_pos, camera_pos = decode_pos(data=nbv)
            #     if np.count_nonzero(target_pos) == 0 and np.count_nonzero(camera_pos) == 0:
            #         continue
            #     tmp = go.Scatter3d(
            #         x=[target_pos[0], camera_pos[0]],
            #         y=[target_pos[1], camera_pos[1]],
            #         z=[target_pos[2], camera_pos[2]],
            #         marker=dict(size=5, color="rgb(255,0,0)"),
            #         line=dict(color="rgb(0,0,255)", width=1),
            #         name=f"nbv-{100*overlap_list[nbv_i]:.2f}%",
            #     )
            #     all_views.append(tmp)
            # data = [cur_data, curr_vector, next_vector, *all_views]

            data = [cur_data, boundary_data, curr_vector, next_vector]


        else:
            cur_data.name = f"Final Point Cloud"
            data = [cur_data]  # Final points do not have the next and current view
        # end if
        fig.add_traces(data)
    # end for

    layout = go.Layout(
        template="none",
        margin=dict(t=40, r=2, b=2, l=2),
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode="data"),
    )
    fig.update_layout(layout)
    fig.show()
