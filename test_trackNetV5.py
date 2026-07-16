# import infer_on_video
import os
import argparse
from dataclasses import dataclass
import cv2

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--dataset_path",default = r"D:\FOLDER_OF_FINAL_BACHELOR_THESIS\test_single_ball_frames",help="Please, write path to hand anotated last frames")
argument_parser.add_argument("--videos_path",default = r"D:\FOLDER_OF_FINAL_BACHELOR_THESIS\test_trackNet_videos",help="Please, write path were are videos of size three")


@dataclass
class Metrics:
    
    TP: int = 0
    FP: int = 0
    FN: int = 0
    TN: int = 0

def test_videos(trackNetV1_label,hand_label,metrics):
    
    if trackNetV1_label is None and len(hand_label) == 0:
        metrics.TN += 1
    
    elif trackNetV1_label is None and len(hand_label) != 0:
        metrics.FN += 1
    
    elif  trackNetV1_label is not None and len(hand_label) == 0:
        metrics.FP += 1
    
    elif trackNetV1_label[0] >= (2*hand_label[0] - hand_label[2]) and trackNetV1_label[0] <= (2*hand_label[2] - hand_label[0]) and trackNetV1_label[1] >= (2*hand_label[1] - hand_label[3]) and trackNetV1_label[1] <= (2*hand_label[3] - hand_label[1]):
        metrics.TP += 1
    
    else:
        metrics.FP += 1
    
def get_frame_name(video_name):
    
    video_index_and_his_index = video_name.split("_")
    last_frame_index = int(video_index_and_his_index[1].split(".")[0])*3
    frame_name = video_index_and_his_index[0] + "_" + str(last_frame_index)
    
    return frame_name

def get_label_corners(dataset_frame_path,frame_name,video_height,video_width):

    label_frame_path = os.path.join(dataset_frame_path,"labels","test",f"{frame_name}.txt")
    
    with open(label_frame_path,"r") as label_file:
        label = [float(bounding_box_label_info) for bounding_box_label_info in label_file.readline().split(" ")[1:]]
    
    
    if label == []:
        return ()
    
    x_center,y_center,width,height = label
    x_left = (x_center - width / 2)*video_width
    x_right = (x_center + width / 2)*video_width
    y_up = (y_center - height / 2)*video_height
    y_down = (y_center + height / 2)*video_height
    
    print(x_left,y_up,x_right,y_down)

    return (x_left,y_up,x_right,y_down)

def create(dataset_frame_path,trackNet_test_folder_videos):
    
    metrics = Metrics()
    
    ball_positions = [(91, 149), (91, 149), (307, 104), None, None, (70, 152), (91, 149), None, None, (488, 194), (376, 136), (413, 79), (91, 149), (91, 149), (137, 101), (91, 149), None, (80, 251), None, (507, 181), (458, 161), (299, 142), (165, 179), None, None, None, (309, 229), (142, 241), None, None, (200, 94), (209, 264), None, (398, 152), None, None, None, None, None, None, None, None, (8, 94), (283, 187), None, (219, 8), None, None, None, (442, 200), (40, 55), (449, 252), (385, 187), None, (5, 51), None, (30, 62), None, (321, 141), None, (104, 155), (162, 65), (94, 83), (155, 254), (254, 68), None, None, (149, 117), None, None, (296, 184), None, (225, 93), (173, 69), (193, 168), (132, 128), None, None, None, (187, 219), (107, 267), None, (251, 203), None, (424, 213), None, None, (187, 184), (406, 97), None, (408, 168), None, None, (199, 157), (132, 209), (387, 206), (200, 149), None, None, None, None, (181, 197), (221, 231), None, None, (152, 81), None, None, None, (363, 104), (125, 229), None, None, None, (360, 71), None, (85, 241), None, None, (366, 225), (269, 168), (27, 165), None, None, None, None, (209, 104), (206, 99), (181, 86), (195, 74), (206, 65), (205, 65), (149, 37), (325, 177), (145, 56), (367, 215), None, (251, 112), (281, 221), (225, 280), None, None, None, (258, 154), (142, 123), (349, 132), None, (129, 190), (350, 225), None, None, (222, 56), (277, 103), None, None, (325, 133), None, None, None, (114, 206), None, None, (219, 235), None, None, (129, 152), (421, 219), None, (190, 170), None, None, (316, 152), None, (382, 216), (333, 141), (334, 113), None, None, (280, 104), (187, 200), (320, 139), None, (193, 190), None, (366, 225), None, (172, 241), (165, 229), None, (337, 235), (231, 108), None, None, None, (117, 168), None, None, (133, 106), None, None, (126, 107), None, None, (329, 184), (456, 184), None, None, None, None, None, (382, 120), None, (158, 104), (101, 197), (149, 155), None, (69, 97), None, None, (273, 24), (267, 113), (283, 54), (166, 136), (369, 60), None, (152, 181), (292, 110), (331, 90), (154, 141), (187, 73), (251, 82), (507, 193), (193, 136), (341, 193), None, (226, 234), None, (406, 33), (17, 238), (251, 145), None, None, (314, 257), (253, 59), (309, 241), (318, 206), (472, 174), (165, 53), None, (156, 193), None, None, (318, 117), (129, 200), (171, 145), None, None, None, (366, 196), (184, 33), (184, 33), (184, 33), None, None, (145, 169), (184, 32), None, (293, 120), (177, 147), None, None, (136, 245), (172, 145), (224, 75), None, None, (152, 49), None, None, (167, 139), None, (340, 137), (417, 262), (200, 161), (340, 179), None, (152, 149), (376, 235), (350, 145), (88, 229), None, None, (211, 104), (101, 225), (183, 108), (353, 141), (366, 221), (208, 78), None, (347, 219), None, None, None, (184, 108), (334, 147), (341, 75), None, (184, 108), (273, 76), None, None, (213, 40), (365, 170), (69, 213), (341, 189), None, (318, 78), (165, 126), (344, 140), None, (125, 187), (171, 85), (433, 219), (395, 178), (145, 162), None, None, None, (181, 122), (361, 173), (216, 125), None, (180, 106), (257, 83), (126, 133), None, (52, 238), (389, 219), (53, 268), (66, 133), (94, 59), (394, 212), None, (125, 111), None, (350, 163), None, None, (427, 246), (414, 197), None, (120, 270), (350, 172), (408, 181), (123, 200), None, (478, 263), (318, 103), (387, 172), (174, 125), (260, 91), (387, 172), (392, 145), (280, 229), (389, 171), (177, 62), None, None, (126, 165), (216, 232), (171, 193), (203, 155), (389, 171), (318, 177), (387, 172), None, (388, 171), (395, 280), None, None, (149, 139), None, None, (190, 61), (139, 117), None, None, None, (152, 229), None, (184, 203), None, None, None, None, (57, 203), (362, 197), (369, 187), (331, 155), None, (40, 261), None, None, None, None, None, None, (453, 187), None, (453, 187), (134, 152), (177, 170), None, (149, 91), (453, 187), None, (453, 187), None, (453, 187), None]
    
    for i,video_name in enumerate(os.listdir(trackNet_test_folder_videos)):
        video_path = os.path.join(trackNet_test_folder_videos,video_name)
        
        frame_name = get_frame_name(video_name)
        
        frame_path = os.path.join(dataset_frame_path,"images","test",f"{frame_name}.jpg")
        frame = cv2.imread(frame_path)
        
        height,width,_ = frame.shape
        
        last_frame_ball_position = None if ball_positions[i] is None else (int(ball_positions[i][0]*width/512),int(ball_positions[i][1]*height/288))

        hand_label = get_label_corners(dataset_frame_path,frame_name,height,width)
        
        test_videos(last_frame_ball_position,hand_label,metrics)
        
    precission = metrics.TP / (metrics.FP + metrics.TP)
    recall = metrics.TP / (metrics.FN + metrics.TP)
    
    print(metrics)
    print("precission", precission)
    print("recall",recall)
    print("f1",2*precission*recall / (precission + recall))
    print("accuracy:",(metrics.TP + metrics.TN) / (metrics.TP + metrics.TN + metrics.FP + metrics.FN))

def main():
    
    args = argument_parser.parse_args()
    
    dataset_frame_path = args.dataset_path
    dataset_video_path = args.videos_path
     
    create(dataset_frame_path,dataset_video_path)

if __name__ == "__main__":
    main()